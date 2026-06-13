# NAV post-train — SMOKE EVIDENCE (accuracy/correctness gate)

**Date:** 2026-06-13  **Env:** `/home/venv/cosmos_predict2` (train), `habitat025` (render)
**Verdict: ALL THREE GATES PASS.** The I2V + action-text conditioning + flow-matching loss + nav-LoRA grad
path are wired correctly; the saved nav delta loads into the eval wrapper with `unexp=0`. Full training launched.

Smoke data: **84 clips, 6 HM3D val scenes** (`/root/nav_posttrain/latents`), latent [5,16,60,104],
prompt_embeds [512,4096], action-text identical to `process_b_action_nav`.

---

## (a) one forward+backward — loss finite, grads into the nav LoRA ONLY
```
[smoke] (a) forward loss = 0.29458  finite=True
[smoke] (a) nav-LoRA params with grad: 264/264; grad global-norm=2.0230e-01; nonzero-grad params=132; nan-grad params=0
[smoke] (a) frozen params with nonzero grad: 0 (must be 0)
=== SMOKE (a) PASS ===
```
- loss finite; 264/264 nav-LoRA tensors get grad; **0 NaN**; grad-norm 0.20 (nonzero).
- 132 nonzero = exactly the `lora_B` halves (`lora_A` grad is 0 at zero-init B — correct LoRA behavior; both flow after step 1).
- **0 grad on ANY frozen param** ⇒ only the nav LoRA trains. F=5 block_mask built correctly (no crash).

## (b) overfit a single batch (FIXED noise+timestep) — loss must drop > 50%
```
[overfit] step   0/100  loss 0.36041
[overfit] step  20/100  loss 0.10876
[overfit] step  50/100  loss 0.05719
[overfit] step  99/100  loss 0.04612
[overfit] start~0.31932  end~0.04612  drop 85.6%
=== OVERFIT PASS (need >50% loss drop) ===
```
Near-monotonic **0.319 → 0.046 (-85.6%)** ⇒ the conditioning + loss + grad path can actually FIT the target
(not just trend). (Residual ~0.046: the fixed sampled timesteps landed near max-noise ~970/1000, where a tiny
rank-8 adapter can't fully denoise — irrelevant to the wiring proof.)

## (c) I2V gen sanity (REAL streaming path) — non-degenerate + steered toward GT + delta loads
Generated from a trained-on (first_frame, action_text) via `pipeline.inference` (the IGNav-eval path),
comparing the distilled student with nav LoRA OFF vs ON, MSE to OWN-clip GT vs OTHER-clip GT
(both 17 frames, 1:1 aligned). nav LoRA = `nav_lora_step300.safetensors` (300-step smoke train on the 84 clips).
```
[gen] clip clip_00000 | Follow this sequence of camera motions: ['forward 0.2m', 'turn_right 22.5°', 'forward 0.2m', 'forward 0.2m'].
[gen] main delta loaded (unexp=0)
[gen][base] T=17 finite=True mean=0.550 per-frame-std=0.2701 inter-frame-motion=0.0618
[gen][base] MSE(gen, OWN-GT)=0.09736  MSE(gen, OTHER-GT)=0.14333  -> conditioning OK (own<other)
[gen] nav LoRA loaded: 264 tensors (unexp=0)
[gen][nav]  T=17 finite=True mean=0.528 per-frame-std=0.2336 inter-frame-motion=0.0631
[gen][nav]  MSE(gen, OWN-GT)=0.07218  MSE(gen, OTHER-GT)=0.11453  -> conditioning OK (own<other)
```
- **non-degenerate**: finite, per-frame-std ~0.23–0.27 (not uniform), inter-frame-motion ~0.063 (dynamic).
- **conditioning works**: MSE(gen, OWN-GT) < MSE(gen, OTHER-GT) for both ⇒ first-frame + action-text steer the rollout.
- **nav LoRA helps**: own-GT MSE **0.0974 → 0.0722 (-25.9%)** ⇒ even 300 steps moves generation toward the real nav trajectory.
- **re-eval wiring proven**: nav delta loads into the eval-time `_add_nav_lora` structure with **unexp=0** (264 tensors).
- Contact sheets: `/root/nav_posttrain/gen/{base,nav,GT}_f{00,08,16}.png` (gen blurry, as expected for the
  300-step under-trained LoRA on the 2-step linearized WM, but clearly tracks the forward+right motion).

---

## How the smoke was run (reproduce)
```
cd /home/wm-vla-rl/_ext/Causal-Forcing
# (a)
/home/venv/cosmos_predict2/bin/python /root/nav_posttrain/scripts/nav_posttrain.py --mode smoke   --latents /root/nav_posttrain/latents
# (b)
/home/venv/cosmos_predict2/bin/python /root/nav_posttrain/scripts/nav_posttrain.py --mode overfit --latents /root/nav_posttrain/latents --steps 100 --lr 1e-3
# (c)
/home/venv/cosmos_predict2/bin/python /root/nav_posttrain/scripts/nav_gen.py --clip clip_00000 --other_clip clip_00010 \
    --nav_lora_ckpt /root/nav_posttrain/ckpts/nav_lora_step300.safetensors --compare_base
```
