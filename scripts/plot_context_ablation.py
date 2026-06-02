"""context 长度 ablation 图：held-out last-frame next-latent MSE vs context 窗口 T。"""
import json, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
RES = "/home/wm-vla-rl/module1_vjepa2ac/results"
d = json.load(open(f"{RES}/e1x_context_ablation.json"))
T = sorted(int(k) for k in d); mse = [d[str(t)] for t in T]
fig, ax = plt.subplots(figsize=(6, 4.2))
ax.plot(T, mse, "o-", color="C4")
ax.set(xlabel="context length T (frames in window)", ylabel="held-out last-frame next-latent MSE",
       title="Context-length ablation (KDA on LIBERO):\nno clear long-context benefit (≈flat, Markovian task)")
ax.set_ylim(0, max(mse) * 1.3); ax.grid(True, alpha=0.3)
for t, m in zip(T, mse):
    ax.annotate(f"{m:.3f}", (t, m), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)
fig.tight_layout(); out = f"{RES}/e1x_context_ablation.png"; fig.savefig(out, dpi=130)
print("PLOT_OK", out)
