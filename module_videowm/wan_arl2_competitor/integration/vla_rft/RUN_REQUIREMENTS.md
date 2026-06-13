# RUN_REQUIREMENTS — running VLA-RFT with OUR KDA-linearized iVideoGPT world model

Status: **integration code is written and the KDA swap is compile/forward/generate/recurrent-decode
validated** (see `sanity_check_kda_swap.py`, all checks pass on `/home/venv/libero`, torch 2.12 +
fla 0.5.0 + transformers 4.46.3). What is below is everything still needed to do a *real* train/eval
run. **No run was attempted** (verl/vLLM not installed here; no checkpoints exist).

---

## 0. What this integration ships (and where)

Canonical copies live in this dir; `apply_patches.sh` also drops them into the VLA-RFT tree.

| File | Role |
|---|---|
| `wm_kda_surgery.py` | WM-build hook: `apply_wm_kda_surgery(world_module, world_model_cfg, rank)` — calls `kdaify_llama` + `attnres_ify_llama`, gated by config flags. Lands at `train/verl/verl/workers/rollout/wm_kda_surgery.py`. |
| `hf_world_model_rollout.py` | `HFWorldModelRollout` — HF interact rollout that replaces the vLLM WM rollout (KDA can't run on vLLM). Lands at `train/verl/verl/workers/rollout/hf_world_model_rollout.py`. |
| `kda_llama.py` | The KDA/AttnRes surgery primitives (copy of `/home/wm-vla-rl/module_videowm/kda_llama.py`). |
| `patches/01_fsdp_workers.patch` | Inserts the surgery hook after `from_pretrained` (fsdp_workers.py:996) **and** rewires the WM `rollout.name=='hf'` branch to `HFWorldModelRollout`. |
| `patches/02_rollout_init.patch` | Exports `HFWorldModelRollout` from `verl/workers/rollout/__init__.py`. |
| `patches/03_trainer_yaml.patch` | Adds config flags: `world_model.{attn_type,kda_layers,init_from_softmax,attnres_groups}` and `rollout.{hf_recurrent,hf_gt_grows}`. |
| `apply_patches.sh` | `git apply` the 3 patches + copy the 3 modules into the verl tree. |
| `sanity_check_kda_swap.py` | Mechanical smoke test of the hook + decode primitives (no verl needed). |

All 3 patches were verified to `git apply --check` cleanly against the pristine VLA-RFT checkout.

### Launch deltas (add to `train/verl/examples/grpo_trainer/run_vla_rft.sh`)
Replace the WM rollout block (`world_model_rollout.rollout.name=vllm`) with:
```
world_model_rollout.rollout.name=hf \
world_model_rollout.world_model.attn_type=kda \
world_model_rollout.world_model.kda_layers=3 \          # Kimi 3:1 hybrid (null = pure-KDA)
world_model_rollout.world_model.init_from_softmax=True \
world_model_rollout.world_model.attnres_groups=4 \      # 0 to disable depth-axis AttnRes
world_model_rollout.rollout.hf_recurrent=True \         # true O(N) decode; False = per-frame generate
world_model_rollout.rollout.interact=True \
world_model_rollout.rollout.interact_max_tokens=64 \
```
(Drop `world_model_rollout.rollout.gpu_memory_utilization`, `enforce_eager`, `free_cache_engine`,
`load_format`, `tensor_model_parallel_size` for the WM — those are vLLM-only and are ignored by HF.)

---

## B1 (HARD blocker) — a VLA-RFT-format KDA world-model checkpoint

VLA-RFT ships **no** checkpoints (README TODO: "Release pre-trained World Model weights" and VLA policy
weights both unchecked; `checkpoints/` is empty). So a WM ckpt must be produced. The integration needs
**three** artifacts (paths from `run_vla_rft.sh`): `checkpoints/libero/WorldModel/<task>` (the KDA WM),
`checkpoints/libero/WorldModel/Tokenizer` (FSQ), `checkpoints/libero/Base/<task>` (the VLA policy).

### The matching released base ckpt (found)
RLVR-World (thuml, the exact lineage VLA-RFT's WM code is copied from) has released the format-identical
base on Hugging Face:

- **`thuml/rt1-world-model-multi-step-base`** — `LlamaForCausalLM`, **`vocab_size=9008`, `bos=9006`,
  `eos=9007`, `max_position_embeddings=8192`, `tie_word_embeddings=false`** (hidden 768 / 12 layers / 12
  heads). The **vocab + special-token IDs are byte-identical to VLA-RFT's WM** (`processor.bos=9006`,
  `eos=9007`, vocab 9008 = 2·4375 visual + 256 action bins + bos/eos). It is **already a discrete-action
  multi-step world model**, i.e. actions live as token ids in the vocab — exactly VLA-RFT's contract.
  (`-single-step-base` and the RLVR variants `-multi-step-rlvr` / `-single-step-rlvr` are siblings.)
- **`thuml/rt1-compressive-tokenizer`** (+ `thuml/rt1-frame-tokenizer`) — the **FSQ** `CompressiveVQModelFSQ`
  tokenizer for that 9008-vocab layout (covers B3/B4 below).

> Domain caveat: these are **RT1** (Google robot) checkpoints, **not LIBERO**. They are the right *format*
> for validating the surgery + HF-rollout mechanics end-to-end, and the right *starting point* for a
> LIBERO WM, but the reward (pixel/LPIPS distance to LIBERO reference frames) needs a LIBERO-trained WM
> for a faithful VLA-RFT result. Two options:

### Recipe A — surgery + short attention-transfer distillation on the released RT1 base (fast, recommended first)
1. `from_pretrained('thuml/rt1-world-model-multi-step-base')` → `LlamaForCausalLM` (vocab 9008).
2. `apply_wm_kda_surgery(model, cfg(attn_type='kda', kda_layers=3, init_from_softmax=True, attnres_groups=4))`.
   `init_from_softmax` copies q/k/v/o so the KDA layers start ≈ softmax → cheap to adapt.
3. **Short attention-transfer / output distillation**: teacher = the frozen SDPA RT1 WM, student = the KDA
   WM; minimise per-layer hidden-state MSE + output-logit KL on RT1 (or LIBERO RLDS) token sequences for a
   few k steps. Reuse `/home/wm-vla-rl/module_videowm/kda_distill.py` (the validated KDA-LLaMA distill;
   2.47× @ 65k in prior work). Save with HF `save_pretrained` so verl `from_pretrained` loads it.
4. Save the FSQ tokenizer (`thuml/rt1-compressive-tokenizer`) into `checkpoints/.../WorldModel/Tokenizer`.

This yields a **VLA-RFT-format KDA WM** with no I/O-contract changes (discrete-action tokens already
match). Sufficient to A/B the HF-KDA rollout vs the vLLM-softmax baseline on RT1 dynamics.

### Recipe B — LIBERO WM from scratch then surgery (needed for the actual VLA-RFT LIBERO result)
1. Train a LIBERO WM with `train/verl/ivideogpt` (same `CompressiveVQModelFSQ` tokenizer arch as
   `rt1-compressive-tokenizer`, fit on LIBERO RLDS at 224→256 px) producing the 9008-vocab discrete-action
   LlamaForCausalLM (the embedding/lm_head must match `processor.py:191-199` layout).
   *Or* wait for VLA-RFT to release its LIBERO WM (README TODO) and skip straight to surgery.
2. Apply Recipe A steps 2-4 on that LIBERO base.

---

## B2 (HARD) — rollout backend: vLLM → HF (DONE in code; runtime accounting still needed)
KDA's `fla` Triton kernels + module-level `_kda_state` (kda_llama.py:77-88) cannot run under vLLM's
paged-attention executor, so the WM rollout moves to HF. `HFWorldModelRollout` implements this:
- **Default path** (`hf_recurrent=False`): per-frame `model.generate(max_new_tokens=64, use_cache=True)`,
  re-prefilling the grown prefix each frame. Robust, literally "HF generate(use_cache=True)", correct for
  pure-KDA **and** hybrid (HF manages the softmax KV cache + KDA threads `_kda_state` within a frame).
  Cost ≈ O(N²) in the (tiny, N≈600) prefix.
- **Opt-in O(N) path** (`hf_recurrent=True`): manual token-by-token decode that threads KDA `_kda_state`
  across the interleaved 7 action tokens (integration-plan option (a)) and never re-prefills. Threads a
  `DynamicCache` so residual softmax layers in a hybrid stay correct; for pure-KDA the cache is empty.
- It mirrors `vllm_rollout.py:231-242` exactly: loop `range(action_ids.shape[1]-1)` (=7 for
  `segment_length=9`), emit 64 dynamics tokens + append `action_ids[:, t+1]` (7 ids); pad responses to
  `response_length=568` → reshape `[B, 8, 71]` in `msp_reward_fn`. Output DataProto keys match vLLM
  (`prompts, responses, input_ids, attention_mask, position_ids [, gt_responses]`).

**Still needed at run time:** a *fair fixed-compute comparison*. The KDA selling point (longer
`segment_length` / larger GRPO `n` per wall-clock) must be measured against vLLM's highly-optimized
softmax decode, which the baseline keeps. Benchmark both on the same GPU before claiming a win
(cf. the "no efficiency win below ~N66k" lesson in cosmos-linearization).

---

## B3 — discrete-action vs continuous-action representation bridge
- VLA-RFT WM = **discrete action tokens** in the vocab (action discretised to 256 bins/dim, offset +2·V,
  band `[8750, 9006)`; `processor.py:146-159,194-195`). The policy outputs **continuous** 7-DoF actions;
  the *processor* discretises them before the WM (`_discretize_actions`, `libero_action_ranges.pth`). The
  WM↔policy bridge therefore lives entirely in VLA-RFT's processor and is **unchanged** by us.
- **Good news:** the released RLVR-World RT1 base (`thuml/rt1-world-model-multi-step-base`) is *already*
  discrete-action — so basing the KDA WM on it means **no continuous→discrete head conversion is needed**.
  The continuous-head mismatch only affects OUR pre-existing iVideoGPT-KDA checkpoint
  (`HeadModelWithAction`, `action_model.py:36`), which we therefore do **not** reuse. Surgery is applied to
  a discrete-action base instead.

## B4 — FSQ tokenizer adoption
Adopt VLA-RFT's frozen `CompressiveVQModelFSQ` (`ctx_tokenizer/compressive_vq_model.py:35`), loaded via
`TOKENIZER['ctx_cnn'].from_pretrained(processor.tokenizer.path)`. Use `thuml/rt1-compressive-tokenizer`
(format match) for RT1, or a LIBERO-trained FSQ tokenizer of the same arch for LIBERO. The WM only ever
sees token ids, so **no pixel/latent shim is required** once the tokenizer is in place. Nothing in our
code touches the tokenizer; just point `processor.tokenizer.path` at it.

## B5 — environment / pins (build a dedicated VLA-RFT venv)
VLA-RFT needs (README + `third_party/README.md`): python 3.10, CUDA 12.2+, **torch 2.4+**,
`flash-attn==2.6.0.post1+cu122torch2.4`, `vllm` (still needed for the *policy* actor rollout, not the WM),
`verl[gpu]`, `tensordict`, `LIBERO`, `openvla-oft`, `dlimp`. Then add the KDA stack: **`fla`** (Triton
kernels) — present in `/home/venv/libero`, must be installed into the VLA-RFT venv too.
- `import verl` currently fails here (`ModuleNotFoundError: tensordict`) — verl is **not built** in any
  existing env. Build it per the README `uv venv` flow.
- Version skew: our KDA was validated on torch 2.12 + transformers 4.46.3 + fla 0.5.0. VLA-RFT pins
  torch 2.4. The KDAttention.forward signature targets transformers 4.46 LlamaAttention (the version
  VLA-RFT also uses — `ivideogpt/configs/llama.json` says `transformers_version 4.32`, but verl runs 4.4x);
  confirm `fla` builds against the torch in the VLA-RFT venv (it has cu122/cu130 wheels). Re-run
  `sanity_check_kda_swap.py` inside the VLA-RFT venv before training.

## B6 — KDA recurrent-state threading across interleaved action tokens (DONE in code)
Handled in `HFWorldModelRollout._rollout_recurrent`: prefill resets `_kda_state`
(`cache_position[0]==0`); each subsequent step (frame token or the 7 action tokens) is fed with
`cache_position>0` so `_kda_state` carries forward. Validated by sanity-check step [5]/[6] (response len
497 → reshape [7,71]; `_kda_state` populated post-rollout).

## B7 (verify) — gt-action rollout parity quirk
The shipped vLLM gt-action loop (`vllm_rollout.py:216-229`, active because `processor.use_img_gt_ac=True`)
generates each gt-reference frame from the **original prompt** (it never grows the gt sequence) — which
looks like an upstream bug (a coherent autoregressive gt rollout should grow). `HFWorldModelRollout`
**faithfully mirrors** that behaviour by default (`hf_gt_grows=False`) for numerical parity; set
`hf_gt_grows=True` to instead roll out gt autoregressively. **Verify which is intended** against the
shipped vLLM output once a checkpoint exists, since `gt_responses` is the reward reference.

---

## Ranked blockers to a real run
1. **B1 — no VLA-RFT checkpoints.** Mitigated: `thuml/rt1-world-model-multi-step-base` (+`-compressive-tokenizer`)
   are format-identical (9008 vocab, discrete-action, FSQ) and usable as the surgery base. Still need
   (a) the KDA-distilled WM produced via Recipe A/B, and (b) a **Base VLA policy** ckpt (also unreleased —
   train OpenVLA-OFT/miniVLA on LIBERO, or wait for the VLA-RFT release).
2. **B5 — environment not built.** verl/vLLM/flash-attn(cu122 torch2.4)/LIBERO venv must be created and
   `fla` added; verl import currently fails (`tensordict` missing).
3. **B2 — fair fixed-compute benchmark** of HF-KDA rollout vs vLLM-softmax baseline (code done; numbers needed).
4. **Domain gap (RT1 vs LIBERO).** RT1 base validates mechanics; a LIBERO WM (Recipe B / VLA-RFT release)
   is required for the actual LIBERO reward to be meaningful.
5. **B7 — gt-rollout parity** to confirm against shipped vLLM behaviour.

(B3/B4/B6 are resolved in the shipped code + by choosing the discrete-action RT1 base.)

---

## Sanity-check result (this run)
`/home/venv/libero/bin/python sanity_check_kda_swap.py` → **ALL SANITY CHECKS PASSED**:
- `apply_wm_kda_surgery` on a 9008-vocab `LlamaForCausalLM`: Kimi 3:1 → KDA layers {0,1,2,4,5,6},
  softmax {3,7}; surgery copied 24 q/k/v/o; AttnRes added (3 aggregation points).
- forward → logits `[2, 40, 9008]`.
- `generate(use_cache=True)` → `[1, 104]` (prompt 40 + 64 frame tokens).
- recurrent interact loop (prefill + per-token decode, 7×(64 frame + 7 action)) → response len **497**,
  reshape **[7, 71]** (= `[segment_length-1 step, tokens_per_frame+action_dim]`); KDA `_kda_state` populated.

## Sources
- [VLA-RFT (GitHub)](https://github.com/OpenHelix-Team/VLA-RFT) — README TODO: WM & VLA weights unreleased.
- [RLVR-World (GitHub)](https://github.com/thuml/RLVR-World) / [paper arXiv:2505.13934](https://arxiv.org/abs/2505.13934) — FSQ over VQ; iVideoGPT + RLVR lineage.
- [RLVR-World HF collection](https://huggingface.co/collections/thuml/rlvr-world-682f331c75a904b8febc366a)
- [thuml/rt1-world-model-multi-step-base](https://huggingface.co/thuml/rt1-world-model-multi-step-base) — `config.json`: LlamaForCausalLM, vocab 9008, bos 9006, eos 9007.
- [thuml/rt1-compressive-tokenizer](https://huggingface.co/thuml/rt1-compressive-tokenizer) — FSQ tokenizer.
- [iVideoGPT (GitHub)](https://github.com/thuml/iVideoGPT) — released action-cond ckpts are continuous-action/VQ/64px (wrong format; not used).
