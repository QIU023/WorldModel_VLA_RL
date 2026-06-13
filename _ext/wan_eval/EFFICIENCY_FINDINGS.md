# Efficiency F-sweep — honest findings (Causal-Forcing KDA(75%)+AttnRes student vs softmax teacher)

Two complementary micro-benchmarks. **Bottom line: linearization does NOT win on efficiency for THIS
teacher, because Causal Forcing already uses bounded (21-frame) attention.** The linear-vs-quadratic win
only exists at long *unbounded* sequences, a regime this checkpoint never enters.

## A. Full-sequence training-path forward (`wan_eval.py --mode efficiency`)
KDA uses its parallel `chunk_kda` form; softmax uses memory-efficient block-sparse `flex_attention`.

| latent F | px  | KDA      | softmax  | speedup | mem  |
|----------|-----|----------|----------|---------|------|
| 21       | 81  | 0.466s / 14.6GB | 0.539s / 13.6GB | 1.16x | 0.93x |
| 41       | 161 | 0.995s / 44.3GB | 1.478s / 44.2GB | 1.49x | 1.00x |
| ≥81      | ≥321| **OOM**  | **OOM**  | —       | —    |

→ KDA modestly faster (1.16–1.49x) but **memory parity**; BOTH OOM beyond F=41. The "linear vs quadratic
memory" sell does not appear — because flex_attention is already memory-efficient and this mode can't reach
the long lengths where it would matter.

## B. Streaming / autoregressive deployment path (`fsweep_stream.py`, the real deployment mode)
KV-cache for softmax (bounded `max_attention_size=32760` = 21 frames), recurrent state for KDA.

| latent F | px  | softmax teacher        | KDA student            |
|----------|-----|------------------------|------------------------|
| 21 (native) | 81 | gen 2.50s / 119ms-frame / 20.8GB | gen 6.33s / 302ms-frame / 50.2GB |
| ≥41 (>21f)  | ≥161 | — (config cap)       | — (config cap)         |

Two hard findings:
1. **At the native 21-frame length the KDA student is 2.5x SLOWER and 2.4x heavier** than the softmax
   teacher. At 21 frames softmax attention is cheap, so the KDA-chunk + naive cross-layer **AttnRes
   aggregation** (4 taps) + **LoRA** operator overhead is a net LOSS. (A minor dtype mismatch in the
   student also disables fused rms_norm — secondary.)
2. **Generation hard-caps at 21 frames.** `local_attn_size = -1` → the KV-cache is a fixed 32760-token
   (21-frame) buffer with NO rolling window (the rolling code path at causal_model.py:210 is gated on
   `local_attn_size != -1`). Rolling out to F=41 indexes past the cache → zero-width write → crash:
   `kv_cache["k"][:, local_start:local_end] = roped_key  # local_end-local_start == 0`.
   So this checkpoint cannot demonstrate the long-horizon regime without re-distilling under a
   rolling-window (local_attn_size) config — at which point softmax stays bounded-cost too, so KDA would
   still not win. The crossover only exists vs an *unbounded global-attention* teacher (not this one).

## Why ARL²'s 2.26x / 54% does not reproduce here
ARL²'s headline efficiency is almost certainly measured against an unbounded / longer-horizon baseline.
Causal Forcing's bounded 21-frame attention removes the very cost that linearization amortizes. The honest
contribution of this work is therefore the **quality** result (Quality 84.0, no dimension collapse) + the
**REPA-for-linearization** data point + the two KDA inference-state bug fixes — **not** an efficiency win.
