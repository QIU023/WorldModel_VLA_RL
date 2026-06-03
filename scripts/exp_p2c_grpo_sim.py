"""(1) sim-GRPO：小 policy 在 WM(V-JEPA2) latent 上，用 learned progress reward，GRPO 训。
- policy π(a|z)，z=mean-pool 编码器 latent；在真 LIBERO sim rollout（reward=progress head on 真实 obs）。
- GRPO：同 init K 轨迹一组 → group-normalized advantage + reverse-KL to ref。dense reward→组内有方差。
- RL 比 CEM 更抗单步 hacking（真实状态 + 轨迹平均 + KL anchor）。= latent-WM-reward VLA RL (M2)。
"""
import os, sys, math, random, numpy as np, torch
import torch.nn as nn, torch.nn.functional as F
os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, "/home/wm-vla-rl/_ext/vjepa2")
sys.path.insert(0, "/home/wm-vla-rl/module1_vjepa2ac")
sys.path.insert(0, "/home/wm-vla-rl/scripts")
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from app.vjepa_droid.transforms import make_transforms
from hubconf import vjepa2_ac_vit_giant
from exp_p2b_train_progress_head import ProgressHead

DEV = "cuda"
RES = "/home/wm-vla-rl/module1_vjepa2ac/results"
TASK = int(os.environ.get("P2_TASK", "0"))
GROUP = int(os.environ.get("GRPO_GROUP", "8"))      # K 轨迹/组
H = int(os.environ.get("GRPO_H", "40"))             # 轨迹步数
STEPS = int(os.environ.get("GRPO_STEPS", "60"))
KL_BETA = float(os.environ.get("GRPO_KL", "0.02"))
MAXN = float(os.environ.get("ACT_MAXNORM", "0.6"))   # 动作幅度（LIBERO ~[-0.9,0.9]）


class Policy(nn.Module):
    def __init__(self, dim, act=7):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 256), nn.GELU(),
                                 nn.Linear(256, 256), nn.GELU())
        self.mean = nn.Linear(256, act); self.logstd = nn.Parameter(torch.full((act,), -0.7))

    def dist(self, z):
        h = self.net(z); return self.mean(h), self.logstd.clamp(-2.0, 0.3)

    def logprob(self, z, a):
        m, ls = self.dist(z); std = ls.exp()
        return (-0.5 * ((a - m) / std) ** 2 - ls - 0.5 * math.log(2 * math.pi)).sum(-1)


def main():
    enc, _ = vjepa2_ac_vit_giant(pretrained=True); torch.cuda.empty_cache()
    enc = enc.to(DEV).eval()
    for p in enc.parameters(): p.requires_grad_(False)
    dim = enc.embed_dim if hasattr(enc, "embed_dim") else 1408
    head = ProgressHead(dim).to(DEV).eval()
    head.load_state_dict(torch.load("/home/wm-vla-rl/_ext/ckpts/progress_head.pt", map_location="cpu"))
    for p in head.parameters(): p.requires_grad_(False)
    transform = make_transforms(crop_size=256, random_horizontal_flip=False,
                                random_resize_aspect_ratio=(1., 1.), random_resize_scale=(1., 1.),
                                reprob=0., auto_augment=False, motion_shift=False)

    @torch.no_grad()
    def encode(img):                                  # [H,W,3]uint8 -> latent [tok,dim]
        clip = transform(img[None]).unsqueeze(0).to(DEV)
        B, C, T, Hh, Ww = clip.size()
        c = clip.permute(0, 2, 1, 3, 4).flatten(0, 1).unsqueeze(2).repeat(1, 1, 2, 1, 1)
        h = enc(c)[0]                                  # [tok,dim]
        return F.layer_norm(h.float(), (h.size(-1),))

    pi = Policy(dim).to(DEV); ref = Policy(dim).to(DEV); ref.load_state_dict(pi.state_dict())
    for p in ref.parameters(): p.requires_grad_(False)
    opt = torch.optim.AdamW(pi.parameters(), lr=3e-4)

    bench = benchmark.get_benchmark_dict()["libero_spatial"]()
    task = bench.get_task(TASK)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=128, camera_widths=128)
    print(f"sim-GRPO task={task.name} | group={GROUP} H={H} steps={STEPS} kl={KL_BETA}", flush=True)

    hist = []
    for gstep in range(STEPS):
        seed = 1000 + gstep                            # 同 init 一组（组内只变 policy 随机性）
        Z, A, R, succ = [], [], [], []                 # per-traj: [H,dim],[H,7],scalar
        for k in range(GROUP):
            env.seed(seed); obs = env.reset()
            zs, acts = [], []; s = False
            for t in range(H):
                z = encode(np.asarray(obs["agentview_image"]))      # [tok,dim]
                zp = z.mean(0, keepdim=True)                        # [1,dim]
                with torch.no_grad():
                    m, ls = pi.dist(zp); std = ls.exp()
                    a = (m + std * torch.randn_like(m)).clamp(-MAXN, MAXN)
                obs, _, done, _ = env.step(a[0].cpu().numpy().astype(np.float32))
                zs.append(zp[0]); acts.append(a[0])
                if env.check_success():
                    s = True; break
            with torch.no_grad():
                prog = head(encode(np.asarray(obs["agentview_image"]))[None]).item()
            Z.append(torch.stack(zs)); A.append(torch.stack(acts))
            R.append(prog + 2.0 * float(s)); succ.append(float(s))
        Rt = torch.tensor(R, device=DEV)
        adv = (Rt - Rt.mean()) / (Rt.std() + 1e-6)                   # group-normalized advantage
        # policy gradient + reverse-KL to ref
        pg, kl, ntok = 0.0, 0.0, 0
        for zs, acts, ad in zip(Z, A, adv):
            lp = pi.logprob(zs, acts)
            with torch.no_grad():
                lp_ref = ref.logprob(zs, acts)
            pg = pg - (lp.sum() * ad)
            kl = kl + (lp - lp_ref).sum()                           # reverse-KL 估计
            ntok += zs.shape[0]
        loss = pg / GROUP + KL_BETA * kl / max(ntok, 1)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(pi.parameters(), 1.0); opt.step()
        hist.append(dict(step=gstep, R=float(Rt.mean()), Rstd=float(Rt.std()), succ=float(np.mean(succ))))
        print(f"step {gstep:3d} R_mean={Rt.mean():.3f} R_std={Rt.std():.3f} succ={np.mean(succ):.2f} "
              f"loss={loss.item():.3f}", flush=True)
    env.close()
    import json; json.dump(hist, open(f"{RES}/p2c_grpo.json", "w"))
    torch.save(pi.state_dict(), "/home/wm-vla-rl/_ext/ckpts/grpo_policy.pt")
    r0 = np.mean([h["R"] for h in hist[:5]]); r1 = np.mean([h["R"] for h in hist[-5:]])
    s0 = np.mean([h["succ"] for h in hist[:5]]); s1 = np.mean([h["succ"] for h in hist[-5:]])
    print(f"GRPO_DONE  R: {r0:.3f}->{r1:.3f}  succ: {s0:.2f}->{s1:.2f}", flush=True)


if __name__ == "__main__":
    main()
