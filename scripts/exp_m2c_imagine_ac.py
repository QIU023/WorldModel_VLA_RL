"""(M2.2+M2.3 合并) Imagination actor-critic in latent WM — Dreamer-style，标准 scaffold。
actor π(z) + critic V(z) 在 latent 世界模型的**想象 rollout** 里一起学（value 与 actor 耦合，
离线 value 冗余 → 在想象 on-policy 数据上学才有增益）。reward = M2.1 grounded r̂（接地、不可 hack）。

- WM(dynamics) **冻结**（RL-from-pretrained-WM / offline-MBRL）；标准 SDPA predictor（scaffold 不上创新，
  KDA 留给 finale）。可微穿过 WM → Dreamer reparam actor 梯度。
- 想象：z0 取自真 demo 帧（grounded init）；闭环 roll H 步：a_k=actor(z_k)，z_{k+1}=WM(z_k,a_k)，
  r_k=r̂(z_{k+1})，v_k=V(z_k)；λ-return 训 actor(max return)/critic(fit return)。
- 这是 finale 的核心：高效 WM → 廉价长想象 → actor 学长程行为。scaffold 先 smoke imagined-return 上升。
- 真 LIBERO success-rate eval（真指标）单独脚本（后续）。
"""
import os, sys, math, json, random, numpy as np, torch
import torch.nn as nn, torch.nn.functional as F
# autograd 下 SDPA 把 bool mask 广播成 stride-0 bias → flash/mem-eff 要求 contiguous 末维而报错；
# 强制 math backend（能处理 expanded bias，精确）。scaffold 用；大规模再换 contiguous-mask + flash。
import torch.backends.cuda as _cb
_orig_sdp = _cb.sdp_kernel
_cb.sdp_kernel = lambda *a, **k: _orig_sdp(enable_flash=False, enable_mem_efficient=False, enable_math=True)
sys.path.insert(0, "/home/wm-vla-rl/_ext/vjepa2")
sys.path.insert(0, "/home/wm-vla-rl/_ext/vjepa2/notebooks")
sys.path.insert(0, "/home/wm-vla-rl/module1_vjepa2ac")
sys.path.insert(0, "/home/wm-vla-rl/scripts")
from hubconf import vjepa2_ac_vit_giant
from rollout import step_predictor
from utils.mpc_utils import compute_new_pose
from kda_acblock import kda_ify_predictor


class RewardModel(nn.Module):                              # 与 exp_m2a 同构（内联避免拖入 libero）
    def __init__(self, dim, h=256):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, h), nn.GELU(),
                                 nn.Linear(h, h), nn.GELU(), nn.Linear(h, 1))

    def forward(self, z):
        return self.net(z).sigmoid()

DEV = "cuda"
RES = "/home/wm-vla-rl/module1_vjepa2ac/results"
CKPT = "/home/wm-vla-rl/_ext/ckpts"
H = int(os.environ.get("M2C_H", "5"))                 # 想象 horizon
B = int(os.environ.get("M2C_B", "8"))                 # 想象 batch（并行起点）
STEPS = int(os.environ.get("M2C_STEPS", "20"))
GAMMA = float(os.environ.get("M2C_GAMMA", "0.98"))
LAM = float(os.environ.get("M2C_LAMBDA", "0.95"))
MAXN = float(os.environ.get("ACT_MAXNORM", "1.0"))
WM = os.environ.get("M2C_WM", "sdpa_full_predictor_libero.pt")   # scaffold=标准 SDPA WM
ENT = float(os.environ.get("M2C_ENT", "0.001"))


class Actor(nn.Module):
    def __init__(self, dim, act=7):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 256), nn.GELU(),
                                 nn.Linear(256, 256), nn.GELU())
        self.mean = nn.Linear(256, act); self.logstd = nn.Parameter(torch.full((act,), -1.0))

    def forward(self, z):                                  # reparam action [B,act] + entropy
        h = self.net(z); mean = self.mean(h); std = self.logstd.clamp(-2.5, 0.5).exp()
        a = mean + std * torch.randn_like(mean)
        ent = (0.5 + 0.5 * math.log(2 * math.pi) + self.logstd.clamp(-2.5, 0.5)).sum()
        return a, ent


class Critic(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 256), nn.GELU(),
                                 nn.Linear(256, 256), nn.GELU(), nn.Linear(256, 1))

    def forward(self, z):
        return self.net(z).squeeze(-1)                     # [B]


def lambda_return(rewards, values, gamma, lam):
    # rewards: [H,B] (r_1..r_H); values: [H+1,B] (v_0..v_H) -> returns for v_0..v_{H-1}: [H,B]
    H = rewards.shape[0]; R = values[-1]; out = [None] * H
    for k in reversed(range(H)):
        R = rewards[k] + gamma * ((1 - lam) * values[k + 1] + lam * R)
        out[k] = R
    return torch.stack(out)


def main():
    enc, pred = vjepa2_ac_vit_giant(pretrained=True)
    if int(os.environ.get("M2C_KDA", "0")):                # finale：换 O(N) KDA WM（chunk 路径可微，长想象用）
        kda_ify_predictor(pred, use_attnres=bool(int(os.environ.get("M2C_ATTNRES", "0"))))
    pred.load_state_dict(torch.load(f"{CKPT}/{WM}", map_location="cpu"))
    enc = enc.to(DEV).eval(); pred = pred.to(DEV).eval()
    for p in list(enc.parameters()) + list(pred.parameters()): p.requires_grad_(False)
    pred.attn_mask = torch.zeros(1, 1, dtype=torch.bool, device=DEV)
    tok = pred.grid_height * pred.grid_width
    dim = enc.embed_dim if hasattr(enc, "embed_dim") else 1408

    reward = RewardModel(dim).to(DEV).eval()
    reward.load_state_dict(torch.load(f"{CKPT}/reward_model.pt", map_location="cpu"))
    for p in reward.parameters(): p.requires_grad_(False)

    actor = Actor(dim).to(DEV); critic = Critic(dim).to(DEV)
    critic_tgt = Critic(dim).to(DEV); critic_tgt.load_state_dict(critic.state_dict())
    for p in critic_tgt.parameters(): p.requires_grad_(False)
    TAU = float(os.environ.get("M2C_TAU", "0.02"))
    opt = torch.optim.AdamW(list(actor.parameters()) + list(critic.parameters()), lr=3e-4)

    data = torch.load("/home/wm-vla-rl/_ext/datasets/libero_latents.pt")
    print(f"M2C imagine actor-critic | WM={WM} H={H} B={B} steps={STEPS} γ={GAMMA} λ={LAM} | demos={len(data)}", flush=True)

    def mp(frame):                                         # [B,1,tok,dim] -> [B,dim] meanpool
        return frame[:, -1].mean(1)

    hist = []
    for step in range(STEPS):
        # 采样真 demo 起点（grounded init）
        z0, p0 = [], []
        for _ in range(B):
            d = random.choice(data); T = d["T"]; t0 = random.randint(0, max(0, T - H - 2))
            z0.append(d["latent"][t0].float()); p0.append(d["states"][t0].float())
        frame_traj = torch.stack(z0).unsqueeze(1).to(DEV)           # [B,1,tok,dim]
        pose_traj = torch.stack(p0).unsqueeze(1).to(DEV)            # [B,1,7]
        action_traj = None
        zs = [mp(frame_traj)]                                      # z_0；z_1..z_H 想象（带 actor 梯度）
        rewards, ents = [], 0.0
        for t in range(H):
            a, ent = actor(mp(frame_traj)); ents = ents + ent
            a_t = a.clamp(-MAXN, MAXN).unsqueeze(1)                 # [B,1,7]
            action_traj = a_t if action_traj is None else torch.cat([action_traj, a_t], 1)
            nxt = step_predictor(pred, frame_traj, action_traj, pose_traj, tok, normalize_reps=True)
            frame_traj = torch.cat([frame_traj, nxt], 1)
            new_pose = compute_new_pose(pose_traj[:, -1:], a_t.detach()).to(pose_traj.device)
            pose_traj = torch.cat([pose_traj, new_pose], 1)                  # pose 仅作条件，actor 梯度走 action→latent
            zk = nxt[:, 0].mean(1)                                  # [B,dim]
            rewards.append(reward(zk)); zs.append(zk)
        rewards = torch.stack(rewards)                             # [H,B]（actor 梯度经可微 WM 传入）
        with torch.no_grad():                                      # target critic bootstrap（detached → 抗发散）
            tgt_values = torch.stack([critic_tgt(z) for z in zs])  # [H+1,B]
        returns = lambda_return(rewards, tgt_values, GAMMA, LAM)   # 梯度仅经 reward（value 项 detached）
        onl_values = torch.stack([critic(z.detach()) for z in zs[:H]])      # [H,B] 拟合 online critic
        actor_loss = -returns.mean() - ENT * ents / H             # actor = reward-driven through differentiable WM
        critic_loss = F.mse_loss(onl_values, returns.detach())
        loss = actor_loss + 0.5 * critic_loss
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(list(actor.parameters()) + list(critic.parameters()), 1.0)
        opt.step()
        with torch.no_grad():                                     # EMA 更新 target critic
            for tp, p in zip(critic_tgt.parameters(), critic.parameters()):
                tp.mul_(1 - TAU).add_(TAU * p)
        imR = rewards.mean().item(); imRet = returns[0].mean().item()
        hist.append(dict(step=step, imag_reward=imR, imag_return=imRet, critic=critic_loss.item()))
        if step % max(1, STEPS // 10) == 0 or step == STEPS - 1:
            print(f"  step {step:3d} imag_reward(r̂)={imR:.3f} imag_return={imRet:.3f} "
                  f"critic_loss={critic_loss.item():.4f} actor_loss={actor_loss.item():.3f}", flush=True)

    json.dump(hist, open(f"{RES}/m2c_imagine_ac.json", "w"))
    torch.save(dict(actor=actor.state_dict(), critic=critic.state_dict()), f"{CKPT}/imagine_ac.pt")
    r0 = np.mean([h["imag_reward"] for h in hist[:3]]); r1 = np.mean([h["imag_reward"] for h in hist[-3:]])
    print(f"M2C_DONE imag_reward(r̂): {r0:.3f}->{r1:.3f}  (actor 是否在想象里把 grounded reward 拉高)", flush=True)


if __name__ == "__main__":
    main()
