"""开环 rollout 保真度图：latent MSE vs rollout horizon（微调 KDA WM, held-out LIBERO）。"""
import json, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
RES = "/home/wm-vla-rl/module1_vjepa2ac/results"
d = json.load(open(f"{RES}/e1y_rollout_fidelity.json"))
mse = d["horizon_mse"]; hs = list(range(1, len(mse) + 1))
fig, ax = plt.subplots(figsize=(6.2, 4.2))
ax.plot(hs, mse, "o-", color="C1")
ax.set(xlabel="open-loop rollout horizon (steps)", ylabel="predicted-latent MSE vs ground truth",
       title=f"KDA WM open-loop rollout fidelity (LIBERO held-out, n={d['n_demos']})\n"
             "short-horizon accurate; expected open-loop drift")
ax.grid(True, alpha=0.3)
fig.tight_layout(); out = f"{RES}/e1y_rollout_fidelity.png"; fig.savefig(out, dpi=130)
print("PLOT_OK", out)
