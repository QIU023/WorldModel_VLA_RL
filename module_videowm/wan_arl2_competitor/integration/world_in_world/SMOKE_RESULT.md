# World-in-World × OUR linearized Causal-Forcing WM — SMOKE PASS (2026-06-13)

Our linearized Causal-Forcing student (Wan2.1-1.3B AR; 75% KDA+AttnRes+LoRA r8; ckpt
jepa_l025_T8/wanlin_s2_step6000.safetensors) runs END-TO-END inside World-in-World's
closed-loop worker protocol as a `text`/I2V world model.

- wrapper: `downstream/api_models/causalforcing_model.py` (4 registration edits done in the clone).
- env: /home/venv/cosmos_predict2 (+ jaxtyping json_repair); resolved the dual-`utils`-package
  clash (import CF pipeline before world-in-world `downstream`).
- model load 70.6s; surgery 446 trained tensors loaded (unexpected=0, ratio=0.75 groups=4 lora_r=8).
- **1 rollout, 17 px frames, 3.99s/rollout (2-step)**; returns pred_frames (1,14,3,480,480) uint8 [0,255].
- EXIT 0. This is the efficiency baseline for the finding-#3 demo (our 2-step vs Wan 30-step at fixed TTS compute).

## REMAINING to a full IGNav success/SPL number (separate agent doing A = HM3D)
1. Habitat-sim 0.2.5 + HM3D (gated) + IGNav episodes/goal-imgs.
2. VLM proposer (or `--use_heur` to isolate the WM-speed effect).
3. sweep `TTS_ratio` 0.25→1.0; controlled comparison: our linearized WM vs Wan2.1 30-step on the same I2V backbone.
4. zero-shot nav-text following is weak → likely a small nav LoRA for real numbers.
