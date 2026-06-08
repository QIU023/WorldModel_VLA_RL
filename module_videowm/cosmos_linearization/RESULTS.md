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

**Honest negative at these scales:** linear attention does NOT beat FlashAttention-softmax until ~T=64
(N≈66k), and only at parity there. Reasons: (1) FlashAttention softmax is already O(N) memory + very fast;
(2) our Hedgehog φ runs in FP32 for stability (constant overhead); (3) N≤66k is too small — the O(N) win is
asymptotic (SANA-WM's efficiency claim is at *minute-scale* video = millions of tokens). **Next:** push N via
higher resolution / longer sequences to locate the crossover; optionally bf16 the linear kernel.

## Takeaways
- The **distillation recipe works**: output-distillation + lower ratio brings a linearized 2B video DiT
  measurably toward its softmax teacher's generation, no pretraining (~0.4–0.6% trainable, LoRA+AttnRes+φ).
- The **efficiency win is not yet demonstrated** at the tested scale — it requires much longer context, which
  is exactly SANA-WM's regime. This is the next experiment, not a refutation.
