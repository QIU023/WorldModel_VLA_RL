"""(B) learned latent-WM reward：训 progress head（mean-pool latent → MLP → progress∈[0,1]）。
监督 = demo 内归一化进度 t/(T-1)。替掉手工 L1 作为 CEM 的稠密 reward。
输入用 encoder latent（libero_latents.pt；WM 预测的也是同空间的 latent，规划时可直接打分）。
"""
import sys, json, random, torch
import torch.nn as nn, torch.nn.functional as F
RES = "/home/wm-vla-rl/module1_vjepa2ac/results"
STEPS = 3000


class ProgressHead(nn.Module):
    def __init__(self, dim, h=256):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, h), nn.GELU(), nn.Linear(h, 1))

    def forward(self, z):                 # z: [..., tok, dim] -> [..., 1] progress
        return self.net(z.mean(-2)).sigmoid()


def sample_batch(demos, dev, bs=128):
    zs, ps = [], []
    for _ in range(bs):
        d = random.choice(demos); T = d["T"]; t = random.randint(0, T - 1)
        zs.append(d["latent"][t].float()); ps.append(t / max(T - 1, 1))
    return torch.stack(zs).to(dev), torch.tensor(ps, device=dev).unsqueeze(1)


@torch.no_grad()
def eval_head(head, demos, dev):
    mse, mono, n = 0.0, 0, 0
    for d in demos:
        z = d["latent"].float().to(dev)
        p = head(z).squeeze(-1)                                  # [T]
        tgt = torch.linspace(0, 1, d["T"], device=dev)
        mse += F.mse_loss(p, tgt).item()
        # 单调性：相邻递增比例
        mono += (p[1:] > p[:-1]).float().mean().item(); n += 1
    return mse / n, mono / n


def main():
    dev = "cuda"
    data = torch.load("/home/wm-vla-rl/_ext/datasets/libero_latents.pt")
    dim = data[0]["latent"].shape[-1]
    random.seed(0); random.shuffle(data); train, held = data[:-40], data[-40:]
    head = ProgressHead(dim).to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3)
    for step in range(STEPS):
        z, p = sample_batch(train, dev)
        loss = F.mse_loss(head(z), p)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 500 == 0 or step == STEPS - 1:
            print(f"step {step:4d} loss {loss.item():.4f}", flush=True)
    mse, mono = eval_head(head, held, dev)
    torch.save(head.state_dict(), "/home/wm-vla-rl/_ext/ckpts/progress_head.pt")
    json.dump({"held_mse": mse, "held_monotonicity": mono}, open(f"{RES}/p2b_progress_head.json", "w"))
    print(f"PROGRESS_HEAD_OK held: progress-MSE={mse:.4f} monotonicity={mono:.3f} "
          f"(1.0=完全单调递增)", flush=True)


main()
