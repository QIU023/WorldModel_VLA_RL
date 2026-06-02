"""三方 fair parity 柱状图：SDPA-full vs KDA+AttnRes(novel) vs KDA-full。"""
import json, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
RES = "/home/wm-vla-rl/module1_vjepa2ac/results"
m = {f: json.load(open(f"{RES}/{f}_finetune.json"))["finetuned_mse"]
     for f in ["e12_sdpa_full", "e12_kda_attnres", "e12_kda_full"]}
labels = ["SDPA-full\nO(N²)", "KDA+AttnRes\nO(N) novel", "KDA-full\nO(N)"]
vals = [m["e12_sdpa_full"], m["e12_kda_attnres"], m["e12_kda_full"]]
cols = ["#c44", "#2a8", "#48c"]
fig, ax = plt.subplots(figsize=(6.6, 4.3))
bars = ax.bar(labels, vals, color=cols, width=0.6)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.0005,
            f"{v:.4f}\n(+{(v/vals[0]-1)*100:.1f}%)" if v != vals[0] else f"{v:.4f}",
            ha="center", fontsize=10)
ax.set_ylabel("held-out next-latent MSE (↓ better)")
ax.set_title("Fair parity (LIBERO 200-demo, full-finetune 1500 steps):\n"
             "cross-frame Block-AttnRes halves the KDA→SDPA gap (4.3%→2.4%) at O(N)")
ax.set_ylim(0.095, 0.108); ax.grid(axis="y", alpha=0.3)
fig.tight_layout(); fig.savefig(f"{RES}/e12_parity_3way.png", dpi=130)
print("PLOT_OK")
