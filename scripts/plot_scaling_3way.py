"""三方 scaling 叠加图：SDPA O(N²) vs KDA O(N) vs KDA⊕AttnRes O(N)。"""
import json, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
RES = "/home/wm-vla-rl/module1_vjepa2ac/results"


def load(f):
    rows = [r for r in json.load(open(f"{RES}/{f}.json")) if not r.get("oom")]
    return [r["T"] for r in rows], [r["dt_ms"] / 1000 for r in rows], [r["mem_gb"] for r in rows]


s = load("e03_sdpa_scaling"); k = load("e14_kda_scaling"); a = load("e14_kda_attnres_scaling")
fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
for col, (T, dt, mem), lab, mk in [("#c44", s, "SDPA O(N²)", "o"),
                                    ("#48c", k, "KDA O(N)", "s"),
                                    ("#2a8", a, "KDA⊕AttnRes O(N) novel", "^")]:
    ax[0].plot(T, dt, mk + "-", color=col, label=lab)
    ax[1].plot(T, mem, mk + "-", color=col, label=lab)
ax[0].set(xlabel="temporal length T (tubelets)", ylabel="fwd latency (s)", title="latency vs horizon")
ax[1].set(xlabel="temporal length T (tubelets)", ylabel="peak GPU mem (GB)", title="peak memory vs horizon")
for x in ax:
    x.grid(True, alpha=0.3); x.legend()
fig.suptitle("P1.5 — both KDA and KDA⊕AttnRes are O(N) (run to T=1024); SDPA O(N²) OOMs at 768. "
             "AttnRes adds ~10% over KDA, still ~24×/8.6× vs SDPA at T=512.")
fig.tight_layout(); fig.savefig(f"{RES}/scaling_3way.png", dpi=130)
print("PLOT_OK")
