# WorldVLA cont-head KV-cache fast-path — implementation, validation, and decision

Goal: a KV-cached, OPEN-LOOP chunked continuous-action decode for the WorldVLA Chameleon-7B
KDA(75%)+AttnRes + continuous-action-head model, to speed up the LIBERO-10 closed-loop eval.

## What was implemented
- `get_action_Chameleon_cont_awm_ck_cached` in `worldvla/libero_util/Chameleon_utils.py`
  (see `worldvla_contcache.patch`): PREFILLs prompt+images+first `action_start` once
  (`use_cache=True`, fresh `DynamicCache`, `cache_position[0]==0` which RESETS the KDA recurrent
  state), reads the cont head at the `action_start` -> action 0, then per chunk INCREMENTALLY
  appends `[discretized action(7), action_end, next action_start]` (advancing `cache_position`;
  KDA carries its O(1) recurrent state via `fused_recurrent_kda`, full-attn layers reuse cached
  K/V) and reads the cont head at each new `action_start`. Calls the base `ChameleonModel.forward`
  directly (NOT the XLLMX training wrapper, which hardcodes `use_cache=False`).
- Attention masking faithfully replicates `model.generate_att_mask_3` (the training-time mask):
  each action block masks ALL previous action blocks for the FULL-ATTENTION layers, while the KDA
  layers are a pure causal recurrence that ignore the mask (`module_videowm/kda_llama.py`). Rebuilt
  per step as a 4D additive mask from per-position block ids (`-1`=prompt, `k`=action block k):
  query i attends key j iff `j<=i` (causal) AND (`block(j)==-1` OR `block(j)==block(i)`). At an
  `action_start` read position this means it sees only prompt+images+itself in the full-attn layers
  — exactly the full-seq path. (Model uses eager/SDPA additive masks — both consume the 4D mask.)
- `CONT_CACHED` env flag wired into `worldvla/eval_solver_libero_256.py`: `=1` routes the cont
  branch to the cached fn; default (unset/`0`) uses the deterministic full-seq OPEN-LOOP path.
- `eval_conthead_openloop_sharded.sh`: 2-GPU sharded LIBERO-10 launcher, `CONT_NCHUNKS=25`
  (open-loop chunk = fair protocol), `CONT_CACHED` switch.

## Validation (3 standalone tests, run on the merged ckpt)
1. `test_cont_cache_isolate.py` — TEACHER-FORCED identical tokens through both paths (removes the
   open-loop feedback loop) to test mask+cache correctness in isolation:
   **mean rel 0.76%, max|Δ| 0.045 on raw head logits, NON-growing/bounded** across 10 blocks.
   => the mask replicates `generate_att_mask_3` and the KDA cache is CORRECT (deviation is pure
   bf16 `chunk_kda`(prefill)-vs-`fused_recurrent_kda`(decode) rounding, as in `test_kda_cache.py`).
2. `test_cont_cache.py` — end-to-end open-loop, cached vs full-seq, NCHUNKS=8.
3. `test_cont_cache_timing.py` — NCHUNKS=25 (real eval setting):
   - **Speedup = 1.6x** (full-seq 1.84s vs cached 1.19s for 25 actions), NOT 25x.
   - **Both paths are bit-deterministic** (full-vs-full = cache-vs-cache = 0.0000).
   - **cache-vs-full = 18% end-to-end rel divergence** (per-chunk max|Δ| grows 0.002 -> ~0.10-0.17).

## Why only 1.6x (not 25x), and why the cache is not worth shipping here
- The cont head needs **25 sequential forwards** regardless (one per action, to read the cont head
  at each `action_start`; the next `action_start` position depends on the fed-back discretized
  previous action -> inherently autoregressive). KV-caching only removes prompt-reprocessing WITHIN
  those 25 forwards. At these sequence lengths (P~566, +9/chunk) the **per-forward FIXED overhead**
  (32 layers x KDA/AttnRes kernel launches + DynamicCache update) dominates, not prompt reprocessing
  — so each tiny 9-token incremental forward (~0.047s) costs nearly as much as a full 567-token
  forward (~0.074s). Ceiling ~1.6x.
- The "~25x" premise was that prompt-reprocessing / decode is the bottleneck. It is NOT: (a) the real
  ~5x win comes from OPEN-LOOP CHUNKING (NCHUNKS=25 -> 1 decode call per 25 env steps, vs the old
  closed-loop NCHUNKS=1 at ~0.33s decode EVERY step), which the existing full-seq path already does;
  (b) the eval is **rendering-bound** (osmesa CPU, minutes/episode; ~0.3-0.8s/step render vs
  open-loop decode amortized to ~0.074s/step), so KV-caching the decode saves essentially nothing in
  wall-clock.
- Meanwhile the cached path introduces **18% open-loop trajectory divergence** (the implementation is
  correct — teacher-forced 0.76% — but the greedy discretized-action feedback amplifies the tiny
  bf16 chunk_kda-vs-fused_recurrent difference: action_{k-1} differs by ~0.003-0.01, occasionally
  crossing a 1/256 bin boundary -> a different fed-back token -> a different but equally-valid
  rollout, compounding). This exceeds the ~3% match bar.

## Decision (per the task's fallback)
Run the **full-seq OPEN-LOOP eval** (`CONT_NCHUNKS=25`, `CONT_CACHED=0`): deterministic, the fair
reference protocol (matches the discrete head's `generate_dis_ma` action-chunking and the teacher),
and — because the eval is rendering-bound and open-loop already amortizes decode 25x — at
essentially the same wall-clock as the cached path would be. The cached path is preserved (validated,
gated behind `CONT_CACHED=1`) for settings where decode IS the bottleneck (e.g. GPU-rendered, or
much longer prompts), but ships OFF here. No wrong fast eval is shipped.

## Eval result
(populated after the run — `eval_conthead_openloop_sharded.sh`, 2-GPU sharded, LIBERO-10, vs
teacher 59% / discrete-KDA 0%)
