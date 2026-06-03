"""(finale) BC warm-start actor on libero_10 —— VLA-RL 标准从 BC/SFT 起步（from-scratch
imagination RL 在 LIBERO-LONG 上无望）。Actor(meanpool latent_t) -> demo action_t（MSE 监督）。
存 dict(actor=...) 兼容 exp_m2e eval / exp_m2c imagination 初始化。
"""
import os, sys, math, random, json, torch
import torch.nn as nn, torch.nn.functional as F
CKPT = "/home/wm-vla-rl/_ext/ckpts"
RES = "/home/wm-vla-rl/module1_vjepa2ac/results"
DATA = os.environ.get("BC_DATA", "/home/wm-vla-rl/_ext/datasets/libero_10_latents.pt")
OUT = os.environ.get("BC_OUT", "actor_bc_libero10.pt")
STEPS = int(os.environ.get("BC_STEPS", "3000"))


class Actor(nn.Module):                                    # 与 exp_m2c / exp_m2e 同构
    def __init__(self, dim, act=7):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 256), nn.GELU(),
                                 nn.Linear(256, 256), nn.GELU())
        self.mean = nn.Linear(256, act); self.logstd = nn.Parameter(torch.full((act,), -1.0))


def main():
    dev = "cuda"
    data = torch.load(DATA)
    dim = data[0]["latent"].shape[-1]
    # 预展平所有 (meanpool latent_t, action_t)
    Z, A = [], []
    for d in data:
        Z.append(d["latent"].float().mean(1))              # [T,dim]
        A.append(d["actions"].float())                     # [T,7]
    Z = torch.cat(Z).to(dev); A = torch.cat(A).to(dev)
    random.seed(0); nh = max(512, len(Z) // 10)
    perm = torch.randperm(len(Z)); Z, A = Z[perm], A[perm]
    Ztr, Atr, Zh, Ah = Z[:-nh], A[:-nh], Z[-nh:], A[-nh:]
    actor = Actor(dim).to(dev)
    opt = torch.optim.AdamW(actor.parameters(), lr=1e-3)
    print(f"BC actor | data={DATA.split('/')[-1]} pairs={len(Z)} steps={STEPS}", flush=True)
    for step in range(STEPS):
        i = torch.randint(0, len(Ztr), (256,), device=dev)
        pred = actor.mean(actor.net(Ztr[i]))
        loss = F.mse_loss(pred, Atr[i])
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 500 == 0 or step == STEPS - 1:
            with torch.no_grad():
                hmse = F.mse_loss(actor.mean(actor.net(Zh)), Ah).item()
            print(f"  step {step:4d} train {loss.item():.4f} held {hmse:.4f}", flush=True)
    torch.save(dict(actor=actor.state_dict()), f"{CKPT}/{OUT}")
    print(f"BC_DONE held-MSE={hmse:.4f} -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
