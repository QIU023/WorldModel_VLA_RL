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
O(N²); linear is ~linear = O(N)). Pushing to the model's max (128 latent frames × high res) reaches
SANA-WM's minute-scale token regime, where the gap keeps climbing:

| N tokens | linear s | softmax s | **speedup** |
|---|---|---|---|
| 131k | 0.74 | 0.90 | 1.21× |
| 524k (512px·128f) | 4.20 | 8.49 | 2.02× |
| 786k | 7.51 | 17.6 | 2.34× |
| 1.05M (1024px·64f) | 11.7 | 30.2 | 2.58× |
| **1.18M (768px·128f)** | **14.2** | **38.1** | **2.69×** |

Monotonic 1.21×→2.69× as N goes 131k→1.18M. At ~1.2M tokens (minute-scale) softmax needs **38 s per
forward** (impractical) vs linear's 14 s. Extrapolating the trend (speedup ~ N^0.35), true minute-scale
(N≈3.5M) → ~3.5–4×. Memory comparable (~±5GB). This is the direct same-task comparison to SANA-WM's
efficiency claim — the O(N) advantage is real and grows exactly in their high-res/long-video regime.

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

## SANA-comparable absolute quality — FVD + VBench (DAVIS, 96 videos, 75% linear)

The metrics SANA-WM & Cosmos-Predict report. Generated 96 teacher + 96 linearized videos; teacher is the
controlled baseline (absolute numbers are not cross-paper comparable: 256px not native 720p, n=96 not 2048,
S3D not exact-I3D, prediction-divergence — so read teacher-vs-lin, not the absolute value).

**FVD (S3D-K400):** FVD(lin‖teacher) = **75** (low → the linearized model's generative distribution closely
tracks the teacher). FVD(lin‖real) 454 ≈ FVD(teacher‖real) 519 → no quality loss vs real.

**VBench (6 reference-free dimensions):**
| dimension | teacher | lin-75% | Δ |
|---|---|---|---|
| subject_consistency | 0.515 | 0.513 | −0.002 |
| background_consistency | 0.759 | 0.760 | +0.001 |
| motion_smoothness | 0.903 | 0.906 | +0.003 |
| dynamic_degree | 1.000 | 1.000 | 0.000 |
| aesthetic_quality | 0.323 | 0.342 | +0.019 |
| imaging_quality | 0.359 | 0.390 | +0.031 |

**The 75%-linearized distilled model matches the SDPA teacher on every VBench dimension** (Δ ∈ [−0.002, +0.031];
even slightly higher on aesthetic/imaging). Combined: linearization + output-distillation preserves the
teacher's quality (FVD + VBench) while delivering the O(N) speedup — the SANA-WM result reached by
distillation, no pretraining.

## Native-resolution check — 480px DAVIS (DDP-2, GBS=8)

Re-ran at DAVIS's NATIVE 480px (256px was downscaled) to be credible vs SANA's native 720p, on both GPUs
(manual-all-reduce DDP-2, GBS = micro2·accum2·world2 = 8, aligned with the 256px runs).

| metric | 256px | 480px (native) |
|---|---|---|
| FVD teacher ‖ real | 518.6 | **351.7** |
| FVD lin ‖ real | 454.1 | 364.6 |
| FVD lin ‖ teacher | 75.2 | 93.8 |
| lin-vs-teacher LPIPS | 0.375 | 0.546 |

**Two honest findings:** (1) at native 480px the TEACHER is much stronger (FVD vs real 518→352 — 256px was
hampering it), and the 75%-linear student still matches it in ABSOLUTE quality (FVD lin‖real 365 ≈ teacher
352, gap 13). (2) But exact lin-vs-teacher FIDELITY is *lower* at 480px (LPIPS 0.55 vs 0.375, FVD 94 vs 75):
more spatial detail = harder for linear attention to reproduce the teacher exactly. So linearization preserves
*quality* (absolute FVD ≈ teacher) even where it can't perfectly *reproduce* the teacher at higher resolution.
SANA-WM reports VBench overall 80.62/81.89 at native 720p (different base model + 720p + their pipeline — a base
comparison, not our linearization contribution; ours is student≈teacher).

## VBench @ 480px native (teacher vs lin-75%) — appearance preserved, DYNAMICS lost

| dimension | teacher | lin-75% | Δ |
|---|---|---|---|
| subject_consistency | 0.559 | 0.550 | −0.010 |
| background_consistency | 0.745 | 0.746 | +0.002 |
| motion_smoothness | 0.930 | 0.946 | +0.015 |
| **dynamic_degree** | 0.917 | **0.490** | **−0.427** |
| aesthetic_quality | 0.320 | 0.347 | +0.028 |
| imaging_quality | 0.505 | 0.468 | −0.037 |

**Honest key finding:** every *appearance* dimension is preserved (subject/background/motion-smoothness/
aesthetic/imaging within ±0.04 — some higher), BUT **dynamic_degree halves (0.92→0.49)**: the linearized
student generates noticeably MORE STATIC video. This is the linear-attention "smoothing/static" failure mode —
saturated/hidden at 256px (both 1.0) but exposed at native 480px. It also explains the higher lin-vs-teacher
LPIPS at 480px. **Net:** 75% linearization preserves static appearance quality but loses motion/dynamics,
increasingly so with resolution — a real cost, not hidden. (Mitigations to try: lower linearization ratio,
keep more softmax anchors on the temporal axis, or a dynamics-weighted distill loss.)

## Takeaways
- The **distillation recipe works**: output-distillation + lower ratio brings a linearized 2B video DiT
  measurably toward its softmax teacher's generation, no pretraining (~0.4–0.6% trainable, LoRA+AttnRes+φ).
- The **efficiency win is not yet demonstrated** at the tested scale — it requires much longer context, which
  is exactly SANA-WM's regime. This is the next experiment, not a refutation.
