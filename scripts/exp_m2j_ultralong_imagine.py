"""(task#12 step3-4 finale) Ultra-Long 想象 RL —— 3 个 WM(SDPA/KDA/KDA+AttnRes)当想象环境对比。
在长 horizon 想象里跑可微 actor-critic(Dreamer 式),reward=grounded r̂。每次跑一个 WM(env 选)。
目标证据:**SDPA WM 在长 horizon 想象 RL 上 OOM/跑不动,KDA(+AttnRes) 跑得起 + imag_reward 上升**
(= 用户"SDPA OOM 即正证据、KDA 能学就行")。policy 现用 BC-init latent actor(A1 stepping stone);
A2 后续换 OFT-蒸馏 latent 策略。
"""
import os, sys, math, json, random, numpy as np, torch
import torch.nn as nn, torch.nn.functional as F
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
from kda_acblock import kda_ify_predictor, attnres_ify_predictor
from src.models.utils.modules import build_action_block_causal_attention_mask

DEV = "cuda"; RES = "/home/wm-vla-rl/module1_vjepa2ac/results"; CKPT = "/home/wm-vla-rl/_ext/ckpts"
H = int(os.environ.get("M2J_H", "64"))                # 长想象 horizon（压 SDPA）
B = int(os.environ.get("M2J_B", "8"))
STEPS = int(os.environ.get("M2J_STEPS", "15"))
GAMMA = float(os.environ.get("M2J_GAMMA", "0.98")); LAM = float(os.environ.get("M2J_LAMBDA", "0.95"))
MAXN = float(os.environ.get("ACT_MAXNORM", "1.0")); ENT = float(os.environ.get("M2J_ENT", "0.001"))
WM = os.environ.get("M2J_WM", "kda_arlayerH10_sftfull.pt")
KDA = int(os.environ.get("M2J_KDA", "1")); AR = int(os.environ.get("M2J_AR", "1"))
NAME = os.environ.get("M2J_NAME", "kda_attnres")
DATA = os.environ.get("M2J_DATA", "/home/wm-vla-rl/_ext/datasets/libero_10_arlayer_latents.pt")
REWARD = os.environ.get("M2J_REWARD", "reward_model_arlayer.pt")
ACTOR_INIT = os.environ.get("M2J_ACTOR_INIT", "actor_bc_arlayer.pt")


class RewardModel(nn.Module):
    def __init__(self, dim, h=256):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, h), nn.GELU(),
                                 nn.Linear(h, h), nn.GELU(), nn.Linear(h, 1))
    def forward(self, z): return self.net(z).sigmoid()


class Actor(nn.Module):
    def __init__(self, dim, act=7):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 256), nn.GELU(),
                                 nn.Linear(256, 256), nn.GELU())
        self.mean = nn.Linear(256, act); self.logstd = nn.Parameter(torch.full((act,), -1.0))
    def forward(self, z):
        h = self.net(z); mean = self.mean(h); std = self.logstd.clamp(-2.5, 0.5).exp()
        a = mean + std * torch.randn_like(mean)
        ent = (0.5 + 0.5 * math.log(2 * math.pi) + self.logstd.clamp(-2.5, 0.5)).sum()
        return a, ent


class Critic(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 256), nn.GELU(),
                                 nn.Linear(256, 256), nn.GELU(), nn.Linear(256, 1))
    def forward(self, z): return self.net(z).squeeze(-1)


def lambda_return(rewards, values, gamma, lam):
    Hh = rewards.shape[0]; R = values[-1]; out = [None] * Hh
    for k in reversed(range(Hh)):
        R = rewards[k] + gamma * ((1 - lam) * values[k + 1] + lam * R); out[k] = R
    return torch.stack(out)


def main():
    enc, pred = vjepa2_ac_vit_giant(pretrained=True)
    if KDA:
        kda_ify_predictor(pred, use_attnres=False)
        if AR: attnres_ify_predictor(pred, num_blocks=8)
    sd = torch.load(f"{CKPT}/{WM}", map_location="cpu")
    enc.load_state_dict(sd["encoder"]); pred.load_state_dict(sd["predictor"])
    enc = enc.to(DEV).eval(); pred = pred.to(DEV).eval()
    for p in list(enc.parameters()) + list(pred.parameters()): p.requires_grad_(False)
    tok = pred.grid_height * pred.grid_width
    pred.attn_mask = build_action_block_causal_attention_mask(H + 2, pred.grid_height, pred.grid_width, add_tokens=2).to(DEV).contiguous()
    dim = enc.embed_dim if hasattr(enc, "embed_dim") else 1408

    reward = RewardModel(dim).to(DEV).eval(); reward.load_state_dict(torch.load(f"{CKPT}/{REWARD}", map_location="cpu"))
    for p in reward.parameters(): p.requires_grad_(False)
    actor = Actor(dim).to(DEV); critic = Critic(dim).to(DEV)
    if ACTOR_INIT: actor.load_state_dict(torch.load(f"{CKPT}/{ACTOR_INIT}", map_location="cpu")["actor"])
    critic_tgt = Critic(dim).to(DEV); critic_tgt.load_state_dict(critic.state_dict())
    for p in critic_tgt.parameters(): p.requires_grad_(False)
    TAU = 0.02; opt = torch.optim.AdamW(list(actor.parameters()) + list(critic.parameters()), lr=3e-4)
    data = torch.load(DATA)
    print(f"M2J [{NAME}] WM={WM} kda={KDA} ar={AR} | H={H} B={B} steps={STEPS} | demos={len(data)}", flush=True)

    def mp(frame): return frame[:, -1].mean(1)
    import time; t_start = time.time(); status = "ok"; hist = []
    try:
        for step in range(STEPS):
            z0, p0 = [], []
            for _ in range(B):
                d = random.choice(data); T = d["T"]; t0 = random.randint(0, max(0, T - 2))
                z0.append(d["latent"][min(t0, d["latent"].shape[0]-1)].float()); p0.append(d["states"][min(t0, d["states"].shape[0]-1)].float())
            frame_traj = torch.stack(z0).unsqueeze(1).to(DEV); pose_traj = torch.stack(p0).unsqueeze(1).to(DEV); action_traj = None
            zs = [mp(frame_traj)]; rewards, ents = [], 0.0
            for t in range(H):
                a, ent = actor(mp(frame_traj)); ents = ents + ent
                a_t = a.clamp(-MAXN, MAXN).unsqueeze(1)
                action_traj = a_t if action_traj is None else torch.cat([action_traj, a_t], 1)
                nxt = step_predictor(pred, frame_traj, action_traj, pose_traj, tok, normalize_reps=True)
                frame_traj = torch.cat([frame_traj, nxt], 1)
                new_pose = compute_new_pose(pose_traj[:, -1:], a_t.detach()).to(pose_traj.device)
                pose_traj = torch.cat([pose_traj, new_pose], 1)
                zk = nxt[:, 0].mean(1); rewards.append(reward(zk).squeeze(-1)); zs.append(zk)
            rewards = torch.stack(rewards)
            with torch.no_grad(): tgt_values = torch.stack([critic_tgt(z) for z in zs])
            returns = lambda_return(rewards, tgt_values, GAMMA, LAM)
            onl_values = torch.stack([critic(z.detach()) for z in zs[:H]])
            actor_loss = -returns.mean() - ENT * ents / H; critic_loss = F.mse_loss(onl_values, returns.detach())
            (actor_loss + 0.5 * critic_loss).backward()
            torch.nn.utils.clip_grad_norm_(list(actor.parameters()) + list(critic.parameters()), 1.0)
            opt.step(); opt.zero_grad()
            with torch.no_grad():
                for tp, p in zip(critic_tgt.parameters(), critic.parameters()): tp.mul_(1 - TAU).add_(TAU * p)
            imR = rewards.mean().item(); hist.append(imR)
            mem = torch.cuda.max_memory_allocated() / 1e9
            print(f"  [{NAME}] step {step:3d} imag_reward={imR:.3f} mem={mem:.1f}GB", flush=True)
    except RuntimeError as e:
        status = "OOM" if "out of memory" in str(e).lower() else f"ERR:{str(e)[:40]}"
        torch.cuda.empty_cache()
    dt = time.time() - t_start; mem = torch.cuda.max_memory_allocated() / 1e9
    r0 = np.mean(hist[:2]) if len(hist) >= 2 else (hist[0] if hist else 0.0)
    r1 = np.mean(hist[-2:]) if len(hist) >= 2 else (hist[-1] if hist else 0.0)
    out = dict(name=NAME, wm=WM, H=H, B=B, status=status, steps_done=len(hist), peak_mem_gb=round(mem, 1),
               wall_s=round(dt, 1), imag_reward_start=round(float(r0), 3), imag_reward_end=round(float(r1), 3))
    json.dump(out, open(f"{RES}/m2j_{NAME}.json", "w"))
    if ACTOR_INIT or status == "ok":
        torch.save(dict(actor=actor.state_dict()), f"{CKPT}/imagine_ultralong_{NAME}.pt")
    print(f"M2J_DONE [{NAME}] status={status} steps={len(hist)}/{STEPS} peak={mem:.1f}GB "
          f"imag_reward {r0:.3f}->{r1:.3f} ({dt:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
