import json, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
RES = "/home/wm-vla-rl/module1_vjepa2ac/results"
d = json.load(open(f"{RES}/e1z_rollout_fidelity_compare.json"))
hs = list(range(1, 17))
fig, ax = plt.subplots(figsize=(6.8, 4.4))
ax.plot(hs, d["KDA-full"], "s-", color="#48c", label="KDA-full  O(N)")
ax.plot(hs, d["KDA+AttnRes"], "^-", color="#2a8", label="KDA⊕AttnRes  O(N) novel")
imp = (1 - d["KDA+AttnRes"][-1] / d["KDA-full"][-1]) * 100
ax.annotate(f"h16: -{imp:.0f}%", (16, d["KDA+AttnRes"][-1]), textcoords="offset points",
            xytext=(-44, 6), fontsize=11, color="#2a8")
ax.set(xlabel="open-loop rollout horizon (steps)", ylabel="predicted-latent MSE vs GT (↓ better)",
       title="Long-horizon rollout fidelity (LIBERO held-out n=40):\n"
             "cross-frame Block-AttnRes drift advantage GROWS with horizon")
ax.grid(True, alpha=0.3); ax.legend()
fig.tight_layout(); fig.savefig(f"{RES}/e1z_rollout_compare.png", dpi=130); print("PLOT_OK")
