"""(M2.1) Grounded reward model — TD-MPC2/Dreamer-style reward predictor; the STANDARD
(non-toy) replacement for the hacky progress head (writeup §9b). Foundation of imagination
RL in the latent WM (paradigm A): reward接地到真 LIBERO 成功结局，且训练时见过失败态 →
不可像 demo-only progress head 那样被钻空子。

Replay（正负同 encode 管线，避免 norm artifact 作弊；latent = per-token layer_norm 后 mean-pool）:
  POSITIVES = LIBERO demos（专家轨迹，成功）-> γ-return-to-success 目标 γ^(T-1-t)（末态~1）
  NEGATIVES = 随机策略 rollout（失败）       -> 目标 0
  HARD-NEG  = 被 hack 的 GRPO policy 状态（grpo_policy.pt）-> 仅作 held-out 鲁棒性检验：
              grounded head 应给低分（老 progress head 给 ~0.83 高分 → 这就是被 hack 的根因）。
验收: held-out AUC(成功导向 vs 失败) + 在 unseen hacked 状态上 new-reward vs old-progress 的均分对比。
"""
import os, sys, math, json, random, numpy as np, torch
import torch.nn as nn, torch.nn.functional as F
os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, "/home/wm-vla-rl/_ext/vjepa2")
sys.path.insert(0, "/home/wm-vla-rl/module1_vjepa2ac")
sys.path.insert(0, "/home/wm-vla-rl/scripts")
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from app.vjepa_droid.transforms import make_transforms
from hubconf import vjepa2_ac_vit_giant
from exp_p2b_train_progress_head import ProgressHead          # guarded import (old head, for comparison)

DEV = "cuda"
RES = "/home/wm-vla-rl/module1_vjepa2ac/results"
CKPT = "/home/wm-vla-rl/_ext/ckpts"
GAMMA = float(os.environ.get("M2_GAMMA", "0.98"))
N_NEG = int(os.environ.get("M2_NEG_TRAJ", "40"))        # 随机策略失败轨迹数
NEG_H = int(os.environ.get("M2_NEG_H", "60"))
N_POS = int(os.environ.get("M2_POS_DEMOS", "120"))      # demo 正轨迹数
STEPS = int(os.environ.get("M2_STEPS", "3000"))
MAXN = float(os.environ.get("ACT_MAXNORM", "0.6"))
TASK = int(os.environ.get("P2_TASK", "0"))
BENCH = os.environ.get("M2A_BENCH", "libero_spatial")
M2A_DATA = os.environ.get("M2A_DATA", "/home/wm-vla-rl/_ext/datasets/libero_latents.pt")
M2A_OUT = os.environ.get("M2A_OUT", "reward_model.pt")


class RewardModel(nn.Module):                              # grounded value/reward head
    def __init__(self, dim, h=256):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, h), nn.GELU(),
                                 nn.Linear(h, h), nn.GELU(), nn.Linear(h, 1))

    def forward(self, z):                                  # z:[...,dim] meanpool latent -> [...,1] in [0,1]
        return self.net(z).sigmoid()


class _Policy(nn.Module):                                  # 复刻 exp_p2c 的 Policy（不 import，那边 main() 未 guard）
    def __init__(self, dim, act=7):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 256), nn.GELU(),
                                 nn.Linear(256, 256), nn.GELU())
        self.mean = nn.Linear(256, act); self.logstd = nn.Parameter(torch.full((act,), -0.7))


def auc(pos, neg):                                         # Mann-Whitney AUC = P(score_pos > score_neg)
    pos, neg = np.asarray(pos), np.asarray(neg)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float((pos[:, None] > neg[None, :]).mean())


def main():
    enc, _ = vjepa2_ac_vit_giant(pretrained=True); torch.cuda.empty_cache()
    enc = enc.to(DEV).eval()
    for p in enc.parameters(): p.requires_grad_(False)
    dim = enc.embed_dim if hasattr(enc, "embed_dim") else 1408
    transform = make_transforms(crop_size=256, random_horizontal_flip=False,
                                random_resize_aspect_ratio=(1., 1.), random_resize_scale=(1., 1.),
                                reprob=0., auto_augment=False, motion_shift=False)

    @torch.no_grad()
    def encode(img):                                       # [H,W,3]uint8 -> meanpool latent [dim]（同 prep 归一化）
        clip = transform(img[None]).unsqueeze(0).to(DEV)
        B, C, T, Hh, Ww = clip.size()
        c = clip.permute(0, 2, 1, 3, 4).flatten(0, 1).unsqueeze(2).repeat(1, 1, 2, 1, 1)
        h = enc(c)[0]                                       # [tok,dim]
        return F.layer_norm(h.float(), (h.size(-1),)).mean(0)   # [dim]

    # env
    bench = benchmark.get_benchmark_dict()[BENCH]()
    task = bench.get_task(TASK)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=128, camera_widths=128)
    print(f"M2.1 grounded reward | task={task.name} | neg={N_NEG}x{NEG_H} pos_demos={N_POS} γ={GAMMA}", flush=True)

    # ---- POSITIVES: demo latents（已 encode，含 layer_norm）-> meanpool + γ-return target ----
    data = torch.load(M2A_DATA)
    random.seed(0); random.shuffle(data)
    Zpos, Tpos = [], []
    for d in data[:N_POS]:
        lat = d["latent"].float().mean(1)                  # [T,tok,dim]->[T,dim]
        T = d["T"]
        tgt = torch.tensor([GAMMA ** (T - 1 - t) for t in range(T)], dtype=torch.float32)  # γ-return-to-success
        Zpos.append(lat); Tpos.append(tgt)
    Zpos = torch.cat(Zpos).to(DEV); Tpos = torch.cat(Tpos).to(DEV)

    # ---- NEGATIVES: 随机策略 rollout（失败）-> target 0 ----
    Zneg = []
    for k in range(N_NEG):
        env.seed(5000 + k); obs = env.reset()
        for t in range(NEG_H):
            a = np.random.uniform(-MAXN, MAXN, 7).astype(np.float32)
            Zneg.append(encode(np.asarray(obs["agentview_image"])).cpu())
            obs, _, done, _ = env.step(a)
            if env.check_success() or done:               # 随机几乎不会成功；万一成功则丢弃该轨迹
                Zneg = Zneg[:-(t + 1)]; break
        if (k + 1) % 10 == 0:
            print(f"  collected {k+1}/{N_NEG} neg traj ({len(Zneg)} states)", flush=True)
    Zneg = torch.stack(Zneg).to(DEV)
    print(f"  pos states={len(Zpos)} neg states={len(Zneg)}", flush=True)

    # held-out split（正/负各留 20%）
    npos_h = max(64, len(Zpos) // 5); nneg_h = max(32, len(Zneg) // 5)
    Zp_tr, Tp_tr, Zp_h, Tp_h = Zpos[:-npos_h], Tpos[:-npos_h], Zpos[-npos_h:], Tpos[-npos_h:]
    Zn_tr, Zn_h = Zneg[:-nneg_h], Zneg[-nneg_h:]

    # ---- train（balanced minibatch：半正半负）----
    rm = RewardModel(dim).to(DEV)
    opt = torch.optim.AdamW(rm.parameters(), lr=1e-3)
    bs = 256
    for step in range(STEPS):
        ip = torch.randint(0, len(Zp_tr), (bs // 2,), device=DEV)
        ineg = torch.randint(0, len(Zn_tr), (bs // 2,), device=DEV)
        z = torch.cat([Zp_tr[ip], Zn_tr[ineg]])
        y = torch.cat([Tp_tr[ip], torch.zeros(bs // 2, device=DEV)]).unsqueeze(1)
        loss = F.mse_loss(rm(z), y)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 500 == 0 or step == STEPS - 1:
            print(f"  step {step:4d} loss {loss.item():.4f}", flush=True)

    # ---- eval: held-out grounding（AUC 成功导向 vs 失败）----
    rm.eval()
    with torch.no_grad():
        sp = rm(Zp_h[Tp_h > 0.5]).squeeze(-1).cpu().numpy()    # 成功导向（晚期 demo）态
        sn = rm(Zn_h).squeeze(-1).cpu().numpy()                # 失败态
    held_auc = auc(sp, sn)
    print(f"  HELD-OUT grounding AUC(成功导向 vs 失败)={held_auc:.3f}  "
          f"mean r̂: pos={sp.mean():.3f} neg={sn.mean():.3f}", flush=True)

    # ---- robustness: 在 unseen 被-hack 的 GRPO policy 状态上，new-reward vs old-progress ----
    rob = {}
    gpath = f"{CKPT}/grpo_policy.pt"
    if os.path.exists(gpath):
        pol = _Policy(dim).to(DEV).eval(); pol.load_state_dict(torch.load(gpath, map_location="cpu"))
        old = ProgressHead(dim).to(DEV).eval()
        old.load_state_dict(torch.load(f"{CKPT}/progress_head.pt", map_location="cpu"))
        Zhack, succ = [], False
        env.seed(1234); obs = env.reset()
        for t in range(NEG_H):
            z = encode(np.asarray(obs["agentview_image"]))
            Zhack.append(z)
            with torch.no_grad():
                h = pol.net(z[None]); a = pol.mean(h).clamp(-MAXN, MAXN)
            obs, _, done, _ = env.step(a[0].cpu().numpy().astype(np.float32))
            if env.check_success(): succ = True; break
            if done: break
        Zhack = torch.stack(Zhack)
        with torch.no_grad():
            new_r = rm(Zhack).mean().item()
            old_p = old(Zhack[:, None]).mean().item()        # old head mean-pools [.,tok,dim]; 这里已是 [.,dim] -> 加 tok=1
        rob = dict(hacked_states=len(Zhack), hacked_success=succ,
                   new_grounded_reward=new_r, old_progress_head=old_p)
        print(f"  ROBUSTNESS on被-hack态: new grounded r̂={new_r:.3f}  vs  old progress head={old_p:.3f}  "
              f"(成功={succ}) → 越低越说明不再被骗", flush=True)
    env.close()

    torch.save(rm.state_dict(), f"{CKPT}/{M2A_OUT}")
    json.dump(dict(held_auc=held_auc, pos_mean=float(sp.mean()), neg_mean=float(sn.mean()),
                   n_pos=len(Zpos), n_neg=len(Zneg), gamma=GAMMA, **rob),
              open(f"{RES}/m2a_grounded_reward.json", "w"))
    print(f"M2A_DONE auc={held_auc:.3f} reward_model.pt saved", flush=True)


if __name__ == "__main__":
    main()
