# WorldVLA-7B KDA+AttnRes — eval & fitting-gap (LIBERO-Long, 256px, his=2, ck=10)

All numbers from our harness on `libero_10` (LIBERO-Long). Eval = closed-loop success in MuJoCo
(osmesa render); loss/action metrics = teacher-forced on the libero_10 train data.

## Closed-loop success rate (validation of harness + naive port)
| model | trials | success | note |
|---|---|---|---|
| **SDPA SOTA** (`model_256/libero_10`, ~40 ep) | 17 | **10/17 = 59%** | == paper 59.0% → **harness validated** |
| **KDA naive port** (1 ep, no decode cache) | 7 | 0/7 = 0% | HF DynamicCache ≠ KDA dict → init_state=None → garbage |
| **KDA + decode-cache fix** (1 ep) | 2 | 0/2 | still ~0 — dominated by undertraining (see below) |

(per-task trials are low (5–10) → coarse; paper uses 50/task. Enough to tell broken-vs-working.)

## Fitting gap — teacher-forced (THE diagnostic)
| metric | SDPA SOTA (40 ep) | KDA epoch0 (1 ep) | read |
|---|---|---|---|
| closs (combined) | ~1.0–1.1 | ~1.16 | near-parity — **dominated by 256 image tokens** |
| **acc_image_9** (world-model) | 0.66 | **0.66** | **PARITY** — KDA matches WM image prediction |
| **acc_action_0** | 0.64 | 0.43 | gap |
| **l1_loss_action_0** | **0.018** | **0.14** | **8× worse** — the control-relevant gap |

## Interpretation
- KDA (linearized, 1-epoch adapt) **already matches SDPA on the world-model/image tokens** — the
  surgery + AttnRes preserve perceptual/dynamics modeling.
- The gap is **concentrated in action tokens** (L1 0.14 vs 0.018). closs hides it because action
  tokens are ~7/263 of the sequence. This small-but-critical gap **compounds over 500-step closed
  loops** → 0% success despite near-parity loss.
- Root cause = **undertraining: 1 epoch vs the paper's 40**. Fix = continue adapt (resume from epoch0,
  `train_wvla_kda_resume.sh`), drive action-L1 0.14 → ~0.018, re-eval success.

## Two real bugs fixed along the way (both in the eval/inference path)
1. **EGL SIGABRT** under 2 concurrent render procs → switched to `MUJOCO_GL=osmesa` (CPU render, stable).
2. **KDA decode cache** (`kda_llama.py`): HF `generate()` passes a `DynamicCache` (not the dict
   KDAttention checked) → `init_state=None` every step → KDA lost all context. Fix: keep the
   recurrent state on the module across decode steps (prefill `cache_position[0]==0` resets) →
   correct AND O(1)/step (the KDA decode-efficiency selling point). Training path unaffected (use_cache=False → chunk_kda).
