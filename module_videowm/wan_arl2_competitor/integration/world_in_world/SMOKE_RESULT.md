# OUR linearized Causal-Forcing as a World-in-World `text`/I2V world model — SMOKE RESULT

Status: **REAL-CKPT SMOKE PASS.** OUR linearized Causal-Forcing student (not a stub) runs inside the
World-in-World closed-loop worker contract and returns valid `pred_frames` uint8 `[1, 14, 3, 480, 480]`.

Date: 2026-06-13. Box: H200 (143 GB), env `/home/venv/cosmos_predict2` (the env that already runs Causal-Forcing).

---

## 1. What OUR world model is (verified loading)

Base: `/home/causal_forcing_ckpt/framewise/causal_forcing.pt` (EMA-only: top-level key is `generator_ema`, 825 tensors).
Surgery (in-place on `pipeline.generator.model`, identical to `Causal-Forcing/inference.py:107-123`):
- `factorize_causal_wan(ratio=0.75)` → **22/30 blocks** become intra-frame softmax + inter-frame causal-KDA; softmax anchors @ [0,4,8,12,17,21,25,29].
- `attnres_ify_wan(groups=4)` → 4 cross-layer AttnRes aggregation points (lpb=7).
- `add_lora_wan(r=8, include_ffn=True)` → 5.86M LoRA params on Q/K/V/O+FFN of the 22 linearized blocks.
- Trained delta: `/home/wm-vla-rl/_ext/wan_lin_out/jepa_l025_T8/wanlin_s2_step6000.safetensors`
  → **446 trained tensors loaded, unexpected=0** (perfect key match against the surgically-modified generator).

I2V path: first-frame PIL → `transforms.Resize((480,832))+ToTensor+Normalize([0.5],[0.5])` → `[1,3,1,480,832]`
→ `vae.encode_to_latent` → `initial_latent` → `pipeline.inference(noise=[1,N-1,16,60,104], text_prompts, initial_latent)`
(frame-wise: `num_frame_per_block==1`, `batch_size==1`). Output video `[1,T,3,480,832]` in [0,1] →
per-frame PIL → base-class `images_to_tensor(uni_samp=14, save_size=(480,480))`.

Native gen resolution is **fixed at 480×832** because the pipeline hardcodes `frame_seq_length=1560`
(= patchified 60×104 latent). The harness output size (480×480) is applied by the base class afterward.
`num_latent_frames=5` → VAE 4× temporal → **17 pixel frames** → subsampled to 14 for the harness.

---

## 2. Files added / edited (the clone = `/home/wm-vla-rl/_ext/world-in-world`)

New wrapper:
- `downstream/api_models/causalforcing_model.py` — subclasses `DiffuserModel`, overrides `_load_pipe`,
  `_load_pipe_args`, `generate_pipe_images`; `__main__` calls `worker_main(pipe_fd, navigator.inference_batch)`.
  Has a `--smoke` mode that drives the exact `task_fn` worker_main runs.

The 4 registration edits:
1. `downstream/vlm.py:28` — added `"causalforcing"` to `WORLD_MODEL_TYPES["text"]`.
2. `downstream/utils/worker_manager.py:752` — added `"causalforcing"` to the `worker_type` launch list.
3. `downstream/utils/workers_cfg.py` — added `COMMON_ARGS["Host3"]["causalforcing_worker"]` (this box → `Host3`):
   ```python
   "causalforcing_worker": [
       "/home/venv/cosmos_predict2/bin/python",
       "downstream/api_models/causalforcing_model.py",
       "--config_path=/home/wm-vla-rl/_ext/Causal-Forcing/configs/causal_forcing_dmd_framewise_2step.yaml",
       "--checkpoint_path=/home/causal_forcing_ckpt/framewise/causal_forcing.pt",
       "--surgery_ckpt=/home/wm-vla-rl/_ext/wan_lin_out/jepa_l025_T8/wanlin_s2_step6000.safetensors",
       "--surgery_ratio=0.75", "--surgery_groups=4", "--surgery_lora_r=8",
       "--num_latent_frames=5", "--num_output_frames=14", "--width=480", "--height=480",
   ],
   ```
4. (docs/05_add_new_WM.md = the spec followed; env_config txt is optional and skipped.)

Env reconcile: in `/home/venv/cosmos_predict2`, `pip install jaxtyping json_repair` (imageio already present).
**torch untouched** (2.12.0+cu130). No other deps needed.

Import-ordering gotcha (handled in the wrapper): both repos ship a top-level `utils` package;
world-in-world's has `__init__.py` (regular package → always shadows CF's namespace `utils`). The wrapper
imports Causal-Forcing's `pipeline` (claiming `utils`) BEFORE world-in-world's `downstream`, `chdir`s to
`CF_DIR` (CF's wrappers use relative `wan_models/`, `configs/` paths), and does NOT import world-in-world's
`utils.logger`.

---

## 3. Commands

Real-ckpt smoke (one rollout, prints shapes + timing):
```bash
cd /home/wm-vla-rl/_ext/world-in-world
CUDA_VISIBLE_DEVICES=0 /home/venv/cosmos_predict2/bin/python \
  downstream/api_models/causalforcing_model.py --smoke --device=cuda:0
```

As a managed worker (what the harness does — registered as `causalforcing`):
```bash
cd /home/wm-vla-rl/_ext/world-in-world
CUDA_VISIBLE_DEVICES=1 bash downstream/scripts/init_worldmodel_manager.sh <exp> 1 causalforcing &
```

---

## 4. Results (REAL checkpoint)

Conditioning frame: frame 0 of `/home/wan_distill_data_mp4/clip_00000.mp4` (480×832 uint8).
Action text (nav): `Follow this sequence of camera motions: ['forward 0.2m', 'turn_left 22.5°', 'forward 0.2m', 'turn_right 22.5°'].`

Output: `pred_frames` shape **(1, 14, 3, 480, 480)**, dtype **uint8**, range [0,255]. `check_outputdict` passes.

Model load (text-encoder umt5-xxl 11GB + base ckpt 5.6GB + surgery + VAE): **~69-71 s** (one-time per worker).

Per-rollout wall-clock (single I2V rollout = text-encode + VAE-encode + 4-frame AR gen + VAE-decode of 17 frames),
warm GPU mean over 3 runs (first run dropped as warmup), one model load:

| denoising schedule        | per-rollout (s) | vs 2-step |
|---------------------------|-----------------|-----------|
| **2-step (first-chunk 4)**| **2.09**        | 1.00×     |
| 4-step uniform            | 2.62            | 1.25× slower |
| 8-step uniform            | 4.06            | 1.94× slower |

(First, cold rollout incl. CUDA autotune ~3.6-4.4 s; the `--smoke` single-shot reported 4.40 s wall / 3.99 s GPU.)

Efficiency selling point (finding #3): the 2-step linearized student is ~1.25× faster than 4-step and ~1.94×
faster than 8-step at identical output, i.e. closed-loop look-ahead with `TTS_ratio=1.0, query_num=2` costs
~2.1 s × (#candidates) × (#steps/episode) — affordable, where a 30-step Wan2.1 teacher would be ~15× the 2-step cost.

---

## 5. Blockers / remaining steps to a full IGNav run

The WM half is done and validated. A full closed-loop IGNav episode additionally needs (all per
`WORLD_IN_WORLD_INTEGRATION.md`, none are WM bugs):
1. **Habitat-sim 0.2.5 (py3.9) env** — runs the solver. Not installed here.
2. **HM3D scenes + `episodes_IGNav.json.gz` + `igdataset_goal_imgs.zip`** — multi-GB, gated (Matterport).
   NOT downloaded (task constraint: no gated datasets).
3. **VLM policy server** (or `--use_heur` to drop it and isolate WM-speed) + **WM manager** (vllm env).
   IGNav needs no detection model (lightest Habitat task).
4. Then: launch VLM (8000) → WM manager `... 1 causalforcing` (7000) → `init_solvers.sh downstream.solver_IGNav <exp> 4 localhost:8000 localhost:7000 --use_WM` → `evaluator.py <exp> --task IGNav`.

Quality caveat (unchanged from analysis): zero-shot Wan2.1-1.3B following discrete nav text is weak; for real
success numbers add a small navigation LoRA (treat as `FTtext`). API-compatibility & speed are proven now.

Minor: a benign `rms_norm` dtype warning (KDA `kda_o_norm` is bf16, input float → fused-kernel fallback, no
correctness impact). Could cast the norm to match for a tiny speedup; not required.
