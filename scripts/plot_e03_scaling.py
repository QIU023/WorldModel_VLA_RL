"""E0.3 出图：SDPA predictor 延迟/显存 vs 时序长度 T。KDA(P1) 后续同轴叠加。"""
import json, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = "/home/wm-vla-rl/module1_vjepa2ac/results"
rows = [r for r in json.load(open(f"{RES}/e03_sdpa_scaling.json")) if not r.get("oom")]
T = [r["T"] for r in rows]
dt = [r["dt_ms"] / 1000 for r in rows]
mem = [r["mem_gb"] for r in rows]

fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
ax[0].plot(T, dt, "o-", color="C0", label="SDPA (V-JEPA2-AC, O(N²))")
ax[0].set(xlabel="temporal length T (tubelets)", ylabel="fwd latency (s)",
          title="Predictor fwd latency vs horizon")
ax[1].plot(T, mem, "o-", color="C3", label="SDPA")
ax[1].set(xlabel="temporal length T (tubelets)", ylabel="peak GPU mem (GB)",
          title="Predictor peak memory vs horizon")
for a in ax:
    a.grid(True, alpha=0.3); a.legend()
fig.suptitle("E0.3 — SDPA block-causal predictor: the O(N²) wall  (KDA/AttnRes P1 will overlay here)")
fig.tight_layout()
out = f"{RES}/e03_sdpa_scaling.png"
fig.savefig(out, dpi=130)
print("PLOT_OK", out)
