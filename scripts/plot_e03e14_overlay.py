"""E1.4 headline 图：SDPA(E0.3) vs KDA(E1.4) 延迟/显存 vs 时序长度 T 叠加。"""
import json, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = "/home/wm-vla-rl/module1_vjepa2ac/results"
sdpa = [r for r in json.load(open(f"{RES}/e03_sdpa_scaling.json")) if not r.get("oom")]
kda = [r for r in json.load(open(f"{RES}/e14_kda_scaling.json")) if not r.get("oom")]
Ts, dts, mems = [r["T"] for r in sdpa], [r["dt_ms"]/1000 for r in sdpa], [r["mem_gb"] for r in sdpa]
Tk, dtk, memk = [r["T"] for r in kda], [r["dt_ms"]/1000 for r in kda], [r["mem_gb"] for r in kda]

fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.4))
ax[0].plot(Ts, dts, "o-", color="C0", label="SDPA (O(N²))")
ax[0].plot(Tk, dtk, "s-", color="C2", label="KDA divided (O(N))")
ax[0].set(xlabel="temporal length T (tubelets)", ylabel="fwd latency (s)", title="Predictor fwd latency vs horizon")
ax[1].plot(Ts, mems, "o-", color="C3", label="SDPA")
ax[1].plot(Tk, memk, "s-", color="C2", label="KDA divided")
ax[1].set(xlabel="temporal length T (tubelets)", ylabel="peak GPU mem (GB)", title="Predictor peak memory vs horizon")
for a in ax:
    a.grid(True, alpha=0.3); a.legend()
fig.suptitle("E1.4 — KDA/AttnRes divided predictor: O(N) vs SDPA O(N²)  (SDPA OOMs at T=512; KDA continues)")
fig.tight_layout()
out = f"{RES}/e14_kda_vs_sdpa_scaling.png"
fig.savefig(out, dpi=130)
print("PLOT_OK", out)
