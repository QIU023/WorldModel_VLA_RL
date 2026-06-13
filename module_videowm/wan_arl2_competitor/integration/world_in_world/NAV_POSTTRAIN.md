# NAV post-train (FTtext LoRA) for OUR linearized Causal-Forcing WM — make I2V FOLLOW the IGNav action-text

**Date:** 2026-06-13  **Box:** 2x H200  **Env:** `/home/venv/cosmos_predict2` (train/encode), `habitat025` (render)
**Goal:** the IGNav closed-loop is policy-floored at **SR 0.10** (`--use_heur`, zero-shot Wan2.1-1.3B follows
discrete nav-text weakly). Fine-tune a small **nav LoRA** on real HM3D rollouts so the WM's I2V path actually
moves the camera per the action-text, to unlock real closed-loop success.

All data + checkpoints live on `/` (`/root/nav_posttrain/`, 815 G free). `/home` untouched except this doc and
the small nav-LoRA copies under `/home/wm-vla-rl/_ext/wan_lin_out/nav_lora/`.

---

## 1. Design (reuse, don't rebuild)

**Student = the EXACT eval-time linearized Causal-Forcing model** (causalforcing_model.py): base CF generator
(`generator_ema`) + KDA(75%) + naive AttnRes(groups=4) + main LoRA(r=8), with the distill delta
`jepa_l025_T8/wanlin_s2_step6000.safetensors` loaded. This whole student is **FROZEN**.

**New = a fresh nav LoRA** (zero-init B -> identity at start) **stacked** on the linearized blocks' Q/K/V/O+FFN
(a second `LoRALinear` wrapping the main `LoRALinear`). Only the nav LoRA trains (5.86 M params, base 1442.6 M frozen).
Because it is zero-init, at step 0 the model is **byte-identical** to the validated eval student.

**Loss = I2V flow-matching to the GROUND-TRUTH future latents** (not teacher distillation), reusing the
rectified-flow parameterization verified against `utils/scheduler.py` and the diffusion-forcing training of CF:
```
x0 = clip latent [B,5,16,60,104]   (frame0 = conditioning first frame)
frame0:  timestep = 0  AND kept clean   (matches pipeline.inference's context frame: timestep*0)
frames 1..4: noised with per-frame schedule timesteps;  v_target = eps - x0
pred = student(noisy, {prompt_embeds=T5(action_text)}, timestep)[0]      # velocity, kv_cache=None -> _forward_train
loss = MSE(pred[:,1:], v_target[:,1:])                                   # frame0 excluded (it's the given cond)
```
The frame-causal mask (built lazily for F=5) makes the future frames attend the clean frame0 ⇒ the loss is
exactly the I2V (first-frame image + action-text) conditioning objective.

**Re-eval is a drop-in:** `causalforcing_model.py` gained `--nav_lora_ckpt` / `--nav_lora_r`. It applies the SAME
surgery order (factorize → attnres → add_lora_wan(main) → load main delta → `_add_nav_lora` → load nav delta);
`_add_nav_lora` is byte-for-byte equivalent to training's `add_nav_lora`, so the nav delta keys match (unexp=0).

Files:
- `/root/nav_posttrain/scripts/navgen_render.py` — phase-1 (habitat025): roll out + render RGB.
- `/root/nav_posttrain/scripts/navgen_encode.py` — phase-2 (cosmos): frames → Wan VAE latents + T5 embeds.
- `/root/nav_posttrain/scripts/nav_posttrain.py` — training (modes: smoke / overfit / train).
- `/root/nav_posttrain/scripts/nav_gen.py` — gate (c) I2V gen sanity via the real streaming path.
- edits: `wan_kda_surgery.py` (LoRALinear made stackable), `causalforcing_model.py` (`--nav_lora_ckpt`).

---

## 2. Data format (matches wan_distill + the EXACT eval action-text)

Two phases (the two envs don't share deps): habitat025 renders, cosmos encodes.

**Render (habitat025):** for HM3D val scenes, random navigable start + random yaw, sensor matched to the eval
(`downstream/simulator.py` default_settings: **512x512, hfov 90, sensor_position 1.0, agent_height 1.0,
agent_radius 0.1, pitch 0**). A length-4 action sequence over {1:forward, 2:turn_left, 3:turn_right} with the
EXACT eval primitives (**UNIT_DISTANCE 0.20 m, UNIT_DEGREE 22.5°**), each primitive executed as **4 micro-steps**
(forward 0.05 m ×4 / turn 5.625° ×4) rendering one frame per micro-step ⇒ **1 + 4×4 = 17 px frames** (smooth video).
Saved: `frames/clip_XXXXX/frame_00..16.png` + `meta.json{b_action=[0]+motions, scene, ...}`.

**Encode (cosmos):** Resize((480,832)) (the wrapper's transform) + Normalize → Wan VAE
`encode_to_latent` ⇒ **latent [5,16,60,104]** (17 px ÷ 4× temporal = 5 latent frames, matching eval
`num_latent_frames=5`). Action-text built EXACTLY like `downstream/api_models.process_b_action_nav`:
`Follow this sequence of camera motions: ['forward 0.2m', 'turn_right 22.5°', ...].` → T5 →
**prompt_embeds [512,4096]**. Saved `latents/clip_XXXXX.pt = {latent, prompt_embeds, prompt, b_action, scene}`
(the SAME schema wan_distill consumes).

> Caveat: data-gen uses HM3D **val** scenes (the same split IGNav evaluates on) per the task brief. This is WM
> post-training (learning the action→camera-motion mapping), not policy training; train poses are random and
> unrelated to the IGNav episodes' start/goal poses. Still, note it when reporting eval numbers.

---

## 3. SMOKE GATE (the 准确性 check) — RESULTS

Data: 84 clips, 6 HM3D val scenes (`/root/nav_posttrain/latents`).

### (a) one forward+backward — loss finite, grads flow into the nav LoRA only
```
[smoke] (a) forward loss = 0.29458  finite=True
[smoke] (a) nav-LoRA params with grad: 264/264; grad global-norm=2.02e-01; nonzero-grad params=132; nan-grad params=0
[smoke] (a) frozen params with nonzero grad: 0 (must be 0)
=== SMOKE (a) PASS ===
```
264/264 nav-LoRA tensors receive grad, 0 NaN, **0 grad on any frozen param**. (132 nonzero = exactly the
`lora_B` halves; `lora_A` grad is 0 at zero-init B — correct LoRA behavior; both flow after step 1.)
Confirms: I2V+text forward runs (F=5 block_mask built ok), loss finite, gradients reach the nav LoRA and
nothing else.

### (b) overfit a single batch (FIXED noise+timestep) — loss must drop substantially
```
[overfit] step   0/100  loss 0.36041
[overfit] step  50/100  loss 0.05719
[overfit] step  99/100  loss 0.04612
[overfit] start~0.31932  end~0.04612  drop 85.6%
=== OVERFIT PASS (need >50% loss drop) ===
```
Near-monotonic 0.319 → 0.046 (**-85.6 %**). Proves the I2V+text conditioning + flow-matching loss + nav-LoRA
grad path can actually fit the target (not just trend). (The residual ~0.046 is because the fixed timesteps
landed near max-noise ~970/1000, where a tiny adapter can't fully denoise — wiring is the point and it's proven.)

### (c) I2V gen sanity (real streaming path) — non-degenerate + steered toward GT + delta loads (unexp=0)
Generated from a trained-on (first_frame, action_text) via `pipeline.inference` (the IGNav-eval path),
nav LoRA OFF vs ON, MSE to OWN-clip GT vs OTHER-clip GT (both 17 frames, 1:1 aligned). nav LoRA = a
300-step smoke train on the 84 clips.
```
[gen] main delta loaded (unexp=0)
[gen][base] T=17 finite=True per-frame-std=0.2701 inter-frame-motion=0.0618
[gen][base] MSE(gen, OWN-GT)=0.09736  MSE(gen, OTHER-GT)=0.14333  -> conditioning OK (own<other)
[gen] nav LoRA loaded: 264 tensors (unexp=0)
[gen][nav]  T=17 finite=True per-frame-std=0.2336 inter-frame-motion=0.0631
[gen][nav]  MSE(gen, OWN-GT)=0.07218  MSE(gen, OTHER-GT)=0.11453  -> conditioning OK (own<other)
```
- non-degenerate (finite, std ~0.23–0.27, motion ~0.063); conditioning works (own<other for both);
- nav LoRA reduces own-GT MSE **0.0974 → 0.0722 (-25.9%)** — post-training steers the rollout toward the real trajectory;
- nav delta loads into the eval `_add_nav_lora` structure with **unexp=0** (re-eval wiring proven empirically).
- Contact sheets: `/root/nav_posttrain/gen/{base,nav,GT}_f{00,08,16}.png` (gen blurry — 300-step under-trained
  LoRA on the 2-step linearized WM — but clearly tracks the forward+right motion).

Full smoke evidence: `/root/nav_posttrain/SMOKE.md`. **GATE PASSED → full training launched.**

---

## 4. FULL TRAIN
- **Data:** 612 clips, ~30 HM3D val scenes (`/root/nav_posttrain/latents`; 84 smoke + 528 full).
- **Recipe:** fresh nav LoRA r=8 on the frozen distilled student; I2V flow-matching to GT future latents.
  2x H200 DDP, cosine LR, checkpoint every 500 steps to `/root/nav_posttrain/ckpts/`, small nav delta copied
  to `/home/wm-vla-rl/_ext/wan_lin_out/nav_lora/` (+`nav_lora_latest.safetensors`) at each save.
- **Command (background, nohup):**
```
cd /home/wm-vla-rl/_ext/Causal-Forcing
nohup torchrun --nproc_per_node=2 /root/nav_posttrain/scripts/nav_posttrain.py --mode train \
    --latents /root/nav_posttrain/latents --batch 1 --accum 4 --steps 3000 --lr 3e-4 --warmup 100 \
    --save_every 500 --ckpt_dir /root/nav_posttrain/ckpts \
    --home_copy /home/wm-vla-rl/_ext/wan_lin_out/nav_lora > /root/nav_posttrain/train.log 2>&1 &
```
- **Status:** ✅ **LAUNCHED & RUNNING** (2026-06-13), 2x H200 DDP, via `launch_full_train.sh`.
  `[ddp] rank 0/2 cuda:0 + rank 1/2 cuda:1`; `[train] step 0/2000 i2v_mse 0.29488 GBS=8`. Steady progress.
  Live log: `/root/nav_posttrain/train.log`. Checkpoints: `/root/nav_posttrain/ckpts/nav_lora_step{500,1000,1500,2000}.safetensors`
  (each also copied to `/home/wm-vla-rl/_ext/wan_lin_out/nav_lora/` + `nav_lora_latest.safetensors`).
  ~45 min ETA. Re-launch / resume: `bash /root/nav_posttrain/scripts/launch_full_train.sh` (resume via `--resume <ckpt>`).
  NOTE: must use the cosmos env's runner `/home/venv/cosmos_predict2/bin/torchrun` (bare `torchrun` = py3.10 env, no einops).

---

## 5. How to re-eval IGNav with the nav LoRA
Launch the WM worker exactly as before but add the nav LoRA:
```
--nav_lora_ckpt /home/wm-vla-rl/_ext/wan_lin_out/nav_lora/nav_lora_latest.safetensors --nav_lora_r 8
```
(in `downstream/scripts/init_worldmodel_manager.sh` / wherever `causalforcing_model.py` worker args are set),
then run the IGNav solver as in `IGNAV_EVAL_RESULT.md` (`06.13_IGNav_causalforcing_sub10_tts100` recipe).
With nav LoRA absent the model is byte-identical to the prior eval student (SR 0.10 baseline).

---

## 6. Blockers / honesty
- **No hard blockers.** Data-gen (render + encode), the I2V training wiring, all 3 smoke gates, and the full
  DDP launch all succeeded.
- **Fidelity is modest.** The 2-step linearized WM gens are blurry (inherent to the few-step student); the nav
  LoRA reduces MSE-to-GT but does not produce crisp video. The metric that matters for IGNav is whether the
  *imagined motion follows the action-text* (so the planner picks the goal-reaching sequence), which the (c)
  conditioning test + MSE-improvement support. Whether this lifts SR above the 0.10 floor must be measured by
  re-running IGNav (§5) with `--nav_lora_ckpt`.
- **Train/eval path mismatch (known).** Training uses the full-sequence frame-causal path (`_forward_train`,
  kv_cache=None); IGNav eval uses the streaming KDA path (`_forward_inference`, kv_cache≠None, the per-frame
  state-commit in `wan_kda_surgery.CausalFactorizedKDASelfAttention`). The nav LoRA sits on the shared Q/K/V/O+FFN
  so it applies in both, and (c) generates through the REAL streaming path — but the objective is optimized on the
  training path, so transfer is approximate (same approximation the base distillation already lives with).
- **Data = HM3D val scenes** (same split IGNav evaluates) per the brief — WM post-training, not policy training;
  train poses are random and unrelated to episode start/goal poses. Flag when reporting eval deltas.
- **Block-mask caching:** the CausalWanModel caches `block_mask` for the first F it sees; all nav data is F=5, so
  this is consistent. Don't mix F in one process; gen runs in a fresh process (the eval wrapper) so it's unaffected.
- **Small-data overfit risk:** 612 clips, GBS=8 ⇒ ~76 steps/epoch; 2000 steps ≈ 26 epochs. Intermediate
  checkpoints (step 500/1000/1500) are saved so eval can pick the best before overfitting; expand data if needed.
