# Cosmos-Predict2-2B Linearization — Results (overnight, LIBERO video-WM track)

Downstream task: image+text → future-frame video prediction on held-out LIBERO (24 clips, 256px, 29 frames,
30 denoise steps). Teacher = pristine Cosmos-Predict2-2B-Video2World. Two linearization ratios (75%, 50% of
self-attn blocks → Hedgehog linear), distilled in two stages.

## Quality — 逼近教师 (lin vs teacher generation, identical conditioning+seed)

| model | method | vs_TEACHER PSNR↑ / SSIM↑ / LPIPS↓ | vs_GT PSNR / SSIM / LPIPS |
|---|---|---|---|
| teacher | — | — | 8.61 / 0.21 / 0.83 |
| r0.75 | Stage-1 attn-transfer | 9.49 / 0.29 / 0.85 | 15.52 / 0.556 / 0.72 |
| **r0.75** | **+ Stage-2 output-distill** | **12.24 / 0.44 / 0.50** | 9.63 / 0.248 / 0.82 |
| r0.50 | Stage-1 attn-transfer | 9.72 / 0.30 / 0.82 | 14.26 / 0.514 / 0.76 |
| **r0.50** | **+ Stage-2 output-distill** | **13.01 / 0.51 / 0.42** | 8.76 / 0.227 / 0.81 |

**Findings**
- **Output-distillation (match the teacher's velocity field) is the right 逼近教师 objective.** It roughly
  HALVES the lin-vs-teacher LPIPS (0.85→0.50) and adds +3 PSNR over attn-transfer alone. Flow-matching on
  data does NOT approach the teacher (it fits the data instead) — diagnosed and replaced.
- **Less linearization is closer:** r0.50 beats r0.75 on every vs-teacher metric (LPIPS 0.42 vs 0.50).
- **Honest ceiling:** vs-teacher LPIPS ~0.45 / PSNR ~12.5 is "much closer" but NOT a perfect reproduction
  (perfect ≈ PSNR>30). 75% linearization has a fidelity ceiling; the gap narrows with the ratio + distillation.
- **vs_GT drops after Stage-2** (15.5→9.6) because Stage-1 produced blurry/near-static video that
  accidentally scored high vs slow LIBERO GT; output-distill reproduces the teacher's *actual* (dynamic,
  out-of-domain) generation — so it tracks the teacher, not the static-blur artifact. vs_GT ≈ teacher's own.

## Efficiency — O(N) linear vs O(N²) softmax (full-model forward, 75% linear)

| T (latent frames) | N tokens | linear s / GB | softmax s / GB | speedup |
|---|---|---|---|---|
| 4 | 4 096 | 0.057 / 4.18 | 0.049 / 4.13 | 0.86× |
| 16 | 16 384 | 0.092 / 4.63 | 0.075 / 4.45 | 0.81× |
| 32 | 32 768 | 0.175 / 5.11 | 0.154 / 4.87 | 0.88× |
| 64 | 65 536 | 0.352 / 6.06 | 0.354 / 5.72 | 1.01× |

At 256px (N≤66k) linear only reaches PARITY (FlashAttention-softmax is fast + O(N)-mem; our φ is FP32). But
pushing N via resolution reveals the crossover + a growing O(N) win:

| resolution | T | N tokens | linear s | softmax s | **speedup** |
|---|---|---|---|---|---|
| 512px | 16 | 65 536 | 0.349 | 0.350 | 1.00× (crossover) |
| 512px | 32 | 131 072 | 0.740 | 0.897 | **1.21×** |
| 512px | 64 | 262 144 | 1.689 | 2.610 | **1.54×** |
| 768px | 16 | 147 456 | 0.828 | 1.035 | **1.25×** |
| 768px | 32 | 294 912 | 1.889 | 3.035 | **1.61×** |

**Crossover at N≈66k; beyond it linear scales better and the gap grows** (softmax time is super-linear in N =
O(N²); linear is ~linear = O(N)). At N=295k (768px, 32 frames) the 75%-linear model is **1.61× faster** and
climbing — exactly SANA-WM's high-res / minute-scale regime. (Memory is comparable, ~+1GB for linear: the
FP32 φ + KV state offset FlashAttention's O(N) memory; the win here is SPEED, not memory. bf16 φ would help.)

## Quality vs linearization ratio (output-distill, vs-teacher fidelity)

Sweeping the fraction of self-attn blocks linearized — a clean monotonic quality↔efficiency knob:

| linear % | vs_TEACHER PSNR↑ | SSIM↑ | LPIPS↓ |
|---|---|---|---|
| 25% | 14.32 | 0.590 | 0.351 |
| 50% | 13.01 | 0.509 | 0.421 |
| 75% | 12.24 | 0.439 | 0.497 |
| 87.5% | 10.99 | 0.352 | 0.619 |

Strictly monotonic on every metric: more linearization → further from the teacher, smoothly. The attn-transfer
floor (attn_mse 0.057 / 0.066 / 0.099 / 0.13) and the output-distill floor (0.0011 / 0.0020 / 0.0025 / 0.0037)
are likewise monotonic — the per-layer mismatch propagates predictably to generation. Practically, 50–75% (the
SANA-WM hybrid range) trades ~0.08–0.15 LPIPS-vs-teacher for the O(N) speedup that kicks in past N≈66k.

## Takeaways
- The **distillation recipe works**: output-distillation + lower ratio brings a linearized 2B video DiT
  measurably toward its softmax teacher's generation, no pretraining (~0.4–0.6% trainable, LoRA+AttnRes+φ).
- The **efficiency win is not yet demonstrated** at the tested scale — it requires much longer context, which
  is exactly SANA-WM's regime. This is the next experiment, not a refutation.
