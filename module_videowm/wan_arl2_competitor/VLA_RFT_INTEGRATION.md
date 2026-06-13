# VLA-RFT Integration Plan — swapping in OUR world model as the dense-reward simulator

Date: 2026-06-13
Repo analyzed: `https://github.com/OpenHelix-Team/VLA-RFT` cloned to `/home/wm-vla-rl/_ext/VLA-RFT` (clone succeeded, no fallback URL needed).
Paper: arXiv 2510.00406 "VLA-RFT: Vision-Language-Action Reinforcement Fine-tuning with Verified Rewards in World Simulators".

TL;DR headline finding: **VLA-RFT's world model IS iVideoGPT** (same thuml RLVR-World lineage; the WM code lives verbatim in `train/verl/ivideogpt/`). So OUR-B (iVideoGPT+KDA) is the same model family and is the correct candidate. OUR-A (Causal-Forcing, text/image-to-video Wan2.1) does **NOT** fit — it has no per-step action conditioning. However VLA-RFT did **not** use stock iVideoGPT's action mechanism: it rewrote action conditioning from *continuous embeddings* to *discrete action tokens* so the WM can roll out on **vLLM**. KDA cannot run on vLLM, so the rollout path needs to move to HF generate. And **no checkpoints are shipped yet**, which blocks any real smoke run.

---

## STEP 2 — The VLA-RFT world-model / reward / RL contract (concrete, cited)

### (a) World-model contract — ACTION-CONDITIONED discrete-token iVideoGPT

**It is action-conditioned, not just a video predictor.** The policy's per-step actions are fed into the WM and determine the next predicted frame.

- **Architecture / load path.** The WM is a plain HF `LlamaForCausalLM` loaded with `AutoModelForCausalLM.from_pretrained(...)` in `WorldModelRolloutWorker._build_model` — `train/verl/verl/workers/fsdp_workers.py:975-984`. Template config `train/verl/ivideogpt/configs/llama.json` = LLaMA, `hidden_size=1024`, `num_hidden_layers=24`, 16 heads (~0.3B); the run script overrides `world_model_rollout.world_model.vocab_size=9008` (`run_vla_rft.sh`). So the WM is a small LLaMA over a custom fused vocab.

- **Token / latent space (fused discrete vocab, size 9008).** Built by `ContextMultiStepPredictionProcessor` in `train/verl/ivideogpt/processor.py:172-225`:
  - dynamics visual tokens `[0, visual_token_num)` (per-frame, `tokens_per_frame=64`),
  - context-frame visual tokens, offset `+visual_token_num` (`processor.py:191`),
  - action tokens, discretized to 256 bins/dim and offset `+2*visual_token_num` (`processor.py:194-195`, `_discretize_actions` at `processor.py:146-159`),
  - plus `bos=9006, eos=9007, pad=9007` (run script).
  Sequence layout per sample: `[ctx_tokens] + for each frame t: [64 dyn tokens] + [7 action tokens]` (`processor.py:198-199`).

- **Visual tokenizer (frozen, separate model).** `ctx_cnn` = `CompressiveVQModelFSQ` (finite-scalar-quantization), `init_resolution=256`, `context_length=1` — `train/verl/ivideogpt/ctx_tokenizer/compressive_vq_model.py:35-119`, registry `train/verl/ivideogpt/tokenizer/__init__.py`. Loaded in `TokenizerWorker.init_model` via `TOKENIZER['ctx_cnn'].from_pretrained(config.tokenizer.path)` — `fsdp_workers.py:1723`. Raw video frames are 224×224 (`data.video.resolution=[224,224]`), `segment_length=9` → 1 context + 8 predicted frames; `action_dim=7`, `action_bins=256`.

- **Where the WM forward is called during RL rollout.**
  - Generation (imagined rollout): `RayVLARFTGRPOTrainer.fit` → `self.wm_rollout_wg.generate_sequences(wm_gen_batch)` at **`train/verl/verl/trainer/ppo/ray_trainer.py:1685`** → `WorldModelRolloutWorker.generate_sequences` (`fsdp_workers.py:1055-1097`) → `vLLMRollout.generate_sequences` interact loop (`train/verl/verl/workers/rollout/vllm_rollout/vllm_rollout.py:160-242`).
  - **Action injection** happens at **`vllm_rollout.py:231-242`**: for each of `T-1` steps, vLLM generates `interact_max_tokens=64` visual tokens for the next frame, then the policy's discretized action tokens `actions[j, t+1]` (7 ids) are appended, then the loop continues. This is the literal action-conditioning during rollout.
  - Log-probs for the GRPO ratio: `DataParallelWorldModel.compute_log_prob` / `_forward_micro_batch` — `train/verl/verl/workers/world_model/dp_world_model.py:122-133, 168-258` (HF FSDP forward, `world_module(input_ids=..., use_cache=False)`).
  - The WM is **frozen** during RFT — `fit()` only calls `update_critic`/`update_actor`; there is no WM optimizer.

### (b) Dense reward — reference-distance verifier (no learned reward head)

Computed in `RayVLARFTGRPOTrainer.msp_reward_fn` (processor_type `ctx_msp`) — **`ray_trainer.py:1297-1402`**:
1. Take WM responses, reshape to `[B, segment_length-1, tokens_per_frame+action_dim]`, slice the 64 dynamics tokens (`ray_trainer.py:1306-1308`).
2. Detokenize predicted dyn tokens (+ctx tokens) → pixels via `self.tokenizer_wg.detokenize(...)` (`ray_trainer.py:1317-1325`; impl `fsdp_workers.py:1792-1839`).
3. **Reward = negative distance to the ground-truth reference frames**: `recon_loss` = pixel MAE or MSE (`ray_trainer.py:1332-1337`) + `perceptual_loss` = LPIPS (`ray_trainer.py:1342`; `_perceptual_loss` `fsdp_workers.py:1730-1741`). Weighted by `loss_weight.{mae,mse,lpips}`, aggregated over the segment (`mean`/`last`/`discount`, `ray_trainer.py:1348-1357`).
4. `reward = -loss` placed on the last valid response token (`ray_trainer.py:1389-1398`).

So the reward is a **verifier / goal-reference pixel+perceptual distance**, NOT a learned reward model. An alternative `ac_reward_fn` exists (`ray_trainer.py:1404-1469`) that scores L1/MSE/Huber directly between `predicted_actions` and `gt_actions`, gated by `trainer.use_ac_reward` (the shipped run script sets it `False`, so the WM-rollout reward is the default).

### (c) RL algorithm / policy / action space / rollout loop

- **Algorithm: GRPO** (`algorithm.adv_estimator=grpo`; `compute_advantage` `ray_trainer.py:182-`). Trainer class `RayVLARFTGRPOTrainer(RayPPOTrainer)` — `ray_trainer.py:1018`; training loop `fit()` `ray_trainer.py:1526-1781`. Actor update is PPO-clip with optional MSE/KL aux losses (`run_vla_rft.sh`: `use_mse_loss`, `entropy_coeff`, `mse_kl_*`). `n=16` rollouts/group.
- **Policy fine-tuned: VLA-Adapter / OpenVLA-OFT (miniVLA)** with a **flow-matching action head** (`eval_libero.sh`: `--use_flow_matching True --use_minivla True`; `sample_noisy_actions` `fsdp_workers.py:621-642`, `generate_actions` `fsdp_workers.py:646-676`). Policy lives in `train/verl/vla-adapter/openvla-oft`.
- **Action space: 7-DoF continuous** (LIBERO: dx,dy,dz,droll,dpitch,dyaw,gripper). Ranges in `train/verl/ivideogpt/configs/libero_action_ranges.pth` (verified `torch.Size([7,2])`). Continuous actions are discretized to 256 bins for the WM, but the policy outputs/optimizes them continuously.
- **Rollout loop per step** (`ray_trainer.py:1561-1744`): load batch (policy `pixel_values`, `proprio`, `input_ids`, `gt_actions`, raw `raw_pixel_values`) → policy samples `n` action trajectories (`generate_actions`) → tokenizer `process()` turns raw pixels + predicted actions into WM token sequence + ctx tokens (`fsdp_workers.py:1842-1869`) → WM imagines the next 8 frames conditioned on those actions (`wm_rollout_wg.generate_sequences`) → reward = -dist(imagined frames, gt frames) → GRPO advantage → `update_actor`.

### (d) Shipped WM checkpoint format/arch, env, data, launch

- **Format/arch:** HF `LlamaForCausalLM` (`config.json` + `*.safetensors`), ~hidden 1024 × 24 layers, custom 9008 vocab; plus a separate frozen `CompressiveVQModelFSQ` tokenizer (HF `from_pretrained`). Expected dirs (from `run_vla_rft.sh`): `checkpoints/libero/WorldModel/<task>`, `checkpoints/libero/WorldModel/Tokenizer`, `checkpoints/libero/Base/<task>` (Base VLA policy + `config.json`).
- **HARD blocker — nothing is shipped.** `checkpoints/` contains only an empty `README.md`; README TODO list has "Release pre-trained World Model weights" and VLA weights **unchecked**. So none of the three required checkpoints exist in-repo.
- **Env/deps:** python 3.10, CUDA 12.2+, torch 2.4+, `flash-attn 2.6.0.post1+cu122torch2.4`, `vllm`, `verl[gpu]`, `LIBERO`, `openvla-oft`, `dlimp`. Install via `uv venv` per README / `third_party/README.md`.
- **Dataset:** LIBERO RLDS (`openvla/modified_libero_rlds`), loaded by `RLDSDataset` + `RLDSBatchTransform_V1` (`ray_trainer.py:1157-1194`), `data/README.md`.
- **Launch:** train `scripts/libero/post_train_rlvr.sh` → `train/verl/examples/grpo_trainer/run_vla_rft.sh` → `python -m verl.trainer.main_vla_rft_grpo` (`train/verl/verl/trainer/main_vla_rft_grpo.py`). Eval `scripts/libero/eval_libero.sh`.

---

## STEP 3 — Fit assessment of OUR two candidate world models

### OUR-A = Causal-Forcing (linearized Wan2.1-1.3B) — DOES NOT FIT (reject as simulator)

- It is a **text-to-video / image-to-video** AR diffusion model. `inference.py` consumes `text_prompts` (+ optional `initial_latent` for I2V) — `/home/wm-vla-rl/_ext/Causal-Forcing/inference.py:179-204`; README explicitly: T2V + I2V (`README.md:31,71,147-156`). An action-conditioned variant is only referenced as a **separate external model** (minWM, `README.md:57-58`) — not what we have.
- It has **no per-step action input**: you cannot feed the policy's 7-DoF action at each frame to advance the prediction. Latent is continuous diffusion `[21,16,60,104]`, not a discrete token stream the VLA-RFT harness consumes.
- **Verdict:** cannot serve as VLA-RFT's action-conditioned rollout simulator. At most it could provide a goal-distance / latent-similarity reward against a target video — but that is NOT VLA-RFT's mechanism (which requires autoregressive action-conditioned frame prediction → detokenize → pixel/LPIPS distance). **Reject OUR-A for the simulator role.**

### OUR-B = iVideoGPT + KDA — FITS (correct candidate), with bridgeable mismatches

- iVideoGPT is exactly the right family: action-conditioned interactive video world model = `CompressiveVQ` tokenizer + LLaMA transformer + action conditioning (`/home/wm-vla-rl/_ext/iVideoGPT/README.md`, action-cond bair/robonet models).
- Our KDA work is a **drop-in LLaMA attention replacement**: `kda_llama.kdaify_llama` / `attnres_ify_llama` (`/home/wm-vla-rl/module_videowm/kda_llama.py:178-211, 161-176`), already wired into iVideoGPT's `train_gpt.py:610-624` and `mbrl/video_predictor.py:77-83`. KDA = fla gated delta-rule linear attention (`KDAttention`, `kda_llama.py:23-98`) with **O(1)/step recurrent-state decode** for long rollouts (`kda_llama.py:77-88`).
- **Smoke-validated** (see Step 5): applied KDA(kimi 3:1)+AttnRes to a 9008-vocab LLaMA mimicking the VLA-RFT WM and ran both `forward` and HF `generate(use_cache=True)`.

**Two mechanism mismatches to bridge (honest):**
1. **Action conditioning representation.** Stock iVideoGPT (`HeadModelWithAction`) uses **continuous** action embeddings: `action_linear = nn.Linear(action_dim, embed_dim)` added to `inputs_embeds` at each frame-block start (`/home/wm-vla-rl/_ext/iVideoGPT/ivideogpt/transformer/action_model.py:36, 171-177`), and generates via `self.llm.generate(inputs_embeds=...)`. VLA-RFT instead uses **discrete action tokens** in the vocab + pure-token vLLM generation (`processor.py:194-195`, `vllm_rollout.py:231-242`). These I/O contracts are incompatible — our existing iVideoGPT-KDA checkpoint (continuous head) is **not directly loadable** into VLA-RFT.
2. **Tokenizer.** VLA-RFT = FSQ `CompressiveVQModelFSQ` @ res 256 (`compressive_vq_model.py:35`); stock iVideoGPT = VQ `CompressiveVQModel` (`/home/wm-vla-rl/_ext/iVideoGPT/ivideogpt/vq_model/compressive_vq_model.py:33`). Tokenizer is frozen+separate, so we adopt VLA-RFT's and its 9008-vocab layout.

**Hard runtime blocker for OUR-B too:** VLA-RFT rolls the WM on **vLLM** (`run_vla_rft.sh: world_model_rollout.rollout.name=vllm`; the interact loop is vLLM-only, `vllm_rollout.py:204-245`, and `hf_rollout.py` has no interact). KDA uses custom `fla` kernels + a module-level recurrent state (`kda_llama.py:88`) that vLLM's paged-attention executor does not support. So the rollout must move to **HF generate**.

---

## STEP 4 — Minimal-change integration plan (OUR-B)

Strategy: keep VLA-RFT's GRPO + FSQ tokenizer + discrete-token + reference-distance-reward harness **unchanged**; change only (i) the WM backbone (apply KDA surgery), (ii) the rollout backend (vLLM → HF interact). Do **not** try to reuse OUR-B's exact existing checkpoint (continuous head, different tokenizer/data); reuse the **KDA surgery code** and produce a VLA-RFT-format KDA WM.

Smallest set of edits:

1. **WM build hook (small).** In `WorldModelRolloutWorker._build_model` (after `from_pretrained`, `fsdp_workers.py:984`) and the `DataParallelWorldModel` constructor path, call:
   ```python
   import kda_llama
   kda_llama.kdaify_llama(world_module, layers=<kimi 3:1 set>, init_from_softmax=True)
   kda_llama.attnres_ify_llama(world_module, groups=cfg.attnres_groups)  # optional
   ```
   Add config flags under `world_model_rollout.world_model`: `attn_type`, `kda_layers`, `attnres_groups` (mirror iVideoGPT `train_gpt.py:311-624`). This is the same hook iVideoGPT already uses; KDA's `init_from_softmax` keeps near-softmax behavior at init → cheap to adapt.

2. **Rollout backend (the one real new module, ~80 lines).** Set `world_model_rollout.rollout.name=hf` and implement an **interact loop in `HFRollout`** (`train/verl/verl/workers/rollout/hf_rollout.py`) mirroring `vllm_rollout.py:231-242`:
   ```
   for t in range(segment_length-1):
       out = model.generate(input_ids=seq, max_new_tokens=tokens_per_frame=64, use_cache=True, ...)
       seq = cat([seq, out_frame_tokens, action_ids[:, t+1]])   # append 7 discrete action ids
   ```
   KDA's module recurrent state (`_kda_state`) gives O(1)/step decoding (`kda_llama.py:77-88`).
   - **Implementation care (B6):** `_kda_state` resets on prefill (`cache_position[0]==0`). Between frame blocks we append 7 action tokens that must advance the state before the next `generate`. Cleanest correct impls: (a) run one extra forward over the 7 appended action tokens to update `_kda_state`, or (b) re-prefill the growing prefix each frame (N≈600 tokens, cheap). Pick (a) for true O(N).

3. **Weights — produce a VLA-RFT-format KDA WM.** Options:
   - (a) When VLA-RFT releases its SDPA WM ckpt: `kdaify_llama(ckpt, init_from_softmax=True)` then short KDA distillation/fine-tune on LIBERO RLDS to recover quality (cheap because init≈softmax).
   - (b) Train from scratch using `train/verl/ivideogpt` + KDA surgery on LIBERO. Either way the WM embedding/head must match the 9008-vocab discrete-action layout (`processor.py:191-199`).

4. **Tokenizer:** reuse VLA-RFT's frozen FSQ tokenizer (`processor.tokenizer.path`). No change.

5. **Data:** reuse the LIBERO RLDS pipeline unchanged (`ray_trainer.py:1157-1194`); 224×224 frames → tokenizer res 256.

6. **Latent/resolution shim:** none needed — once we adopt VLA-RFT's tokenizer the WM only ever sees token ids; no pixel/latent shim required.

**Selling point realized:** KDA WM = O(N) attention + O(1)/step recurrent decode. The rollout is `T-1` sequential `generate` calls (the throughput bottleneck), so at fixed compute KDA buys **longer horizons (larger `segment_length`) and/or more GRPO samples (`n`)** for the same wall-clock — the cheaper-simulator argument. (Caveat: fair comparison must account for losing vLLM's optimized softmax decode on the baseline — see B2.)

**Concrete blockers (ranked):**
- **B1 (hard):** No released VLA-RFT checkpoints (WM, FSQ tokenizer, Base VLA policy). Cannot reproduce/AB-test without them or full self-training.
- **B2 (hard):** vLLM ≠ KDA → must implement HF interact rollout; need fair fixed-compute accounting vs the vLLM softmax baseline.
- **B3 (medium):** Action-conditioning representation mismatch (continuous embeddings vs discrete tokens) → OUR-B's existing checkpoint not directly loadable; need a VLA-RFT-format KDA WM via surgery+distill.
- **B4 (medium):** Tokenizer/vocab mismatch (FSQ 9008 vs our VQ) → adopt VLA-RFT's tokenizer, retrain WM embeddings/head for that vocab.
- **B5 (low):** Env pins (torch 2.4 / flash-attn cu122) vs our torch 2.12 libero venv → build a dedicated VLA-RFT venv; `fla`/KDA already installed there.
- **B6 (low):** KDA recurrent-state threading across interleaved action tokens during interactive decode (impl detail in edit #2).

---

## STEP 5 — Env setup / smoke (best-effort; no long training started)

- **Clone:** OK (`/home/wm-vla-rl/_ext/VLA-RFT`).
- **Cannot run their training or eval smoke:** `post_train_rlvr.sh`/`run_vla_rft.sh` and `eval_libero.sh` all reference `checkpoints/libero/{Base,WorldModel,WorldModel/Tokenizer}` which **do not exist** (only an empty `checkpoints/README.md`). Hard blocker (B1).
- **verl not installed** in any existing env: `import verl` → `ModuleNotFoundError: No module named 'tensordict'` (needs the documented `uv` build with torch 2.4 / vllm / flash-attn cu122). Not built here (heavy; no training requested).
- **Smokes that DID pass:**
  - **Hydra config composes cleanly:** `vla_rft_grpo_trainer.yaml` loads, all top-level sections present (`data, processor, actor_rollout_ref, world_model_rollout, critic, reward_model, algorithm, trainer`).
  - **KDA surgery primitive validated end-to-end:** in `/home/venv/libero` (torch 2.12 + `fla`), applied `kdaify_llama(kimi 3:1)` + `attnres_ify_llama(groups=4)` to a `LlamaForCausalLM(vocab=9008)`; `forward` → logits `[1,71,9008]`; HF `generate(max_new_tokens=64, use_cache=True)` → `[1,135]`. Confirms the core WM swap (KDA backbone + autoregressive HF decode) is mechanically sound for the VLA-RFT WM.

### Bottom line
OUR-B (iVideoGPT+KDA) is the right and viable candidate — it is literally the same model family VLA-RFT uses, and our KDA surgery applies cleanly to that LLaMA backbone. The integration is **not** a checkpoint drop-in: it requires (1) a small WM-build surgery hook, (2) a new HF interact rollout to replace vLLM, and (3) producing a VLA-RFT-format (discrete-action-token, FSQ-vocab) KDA WM via surgery + short distillation. OUR-A (Causal-Forcing) is unsuitable as a simulator (no action conditioning) and is at most a goal-distance reward source. The biggest external blocker is that VLA-RFT has released no checkpoints yet.
