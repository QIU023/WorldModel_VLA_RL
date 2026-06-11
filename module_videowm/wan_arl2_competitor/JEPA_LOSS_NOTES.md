# §3a — REPA-style V-JEPA2 alignment loss in Wan Stage-2 distillation

Headline experiment: add a REPA-style representation-alignment loss to the MSE-only Stage-2 output-distillation of
the linearized Wan2.1-1.3B (KDA-75% + AttnRes) student, to close the consistency + dynamics gap vs the teacher
(MSE-only Stage-2 → VBench Quality 83.84; teacher 88.09).

**Loss:** `L = velocity_MSE + λ · mean(1 − cos(proj(DiT_hidden), VJEPA2_target))`, λ = 0.25.
The velocity-MSE term is byte-for-byte the existing Stage-2 loss; REPA only ADDS the align term + a projection MLP.
(λ = 0.25, not REPA's 0.5, because our MSE ≈ 0.065 is ~10× smaller than REPA's diffusion loss.)

## Geometry (why spatial aligns 1:1, only temporal interpolates)
- Wan latent `[F=21, C=16, H=60, W=104]`, patch (1,2,2) → DiT token grid **(21, 30, 52)** = 32760 tokens, dim 1536.
  Token order is (F,H,W) row-major (Conv patch → flatten → transpose). Reshape hidden `[B,32760,1536] → [B,21,30,52,1536]`.
- V-JEPA2 ViT-g/16, tubelet (2,16,16), fed the FULL **480×832** target frame → spatial patch grid **30×52**, which
  EXACTLY matches Wan's (30,52). Same (T,H,W) row-major token order. So spatial alignment is 1:1 — no spatial interp.
- Only the temporal axis differs (V-JEPA2 tubelets vs Wan's 21 frames) → `F.interpolate` time-axis only (trilinear,
  H,W preserved) from the stored `T_vjepa` up to 21 inside the loss.

## STEP 1 — V-JEPA2 target precompute  (`jepa_preproc_encode.py`, venv `/home/venv/vjepa2`)
- Encoder = **ENCODER half** of `vjepa2_ac_vit_giant` (`vit_giant_xformers_rope`, embed_dim **1408**, RoPE,
  `handle_nonsquare_inputs=True`, `use_sdpa=True` so no xformers needed). Weights from the AC ckpt's `["encoder"]`
  (`/home/.cache/torch/hub/checkpoints/vjepa2-ac-vitg.pt`, re-downloaded from dl.fbaipublicfiles.com, 11.76 GB).
- Per clip: read mp4 (`/home/wan_distill_data_mp4/clip_XXXXX.mp4`, 81 frames) → **uniformly subsample 81→64**
  (encoder native length → 32 tubelets, in-distribution temporal RoPE) → full-res 480×832 (spatial RoPE
  extrapolated, the only OOD axis) → imagenet-normalized → encoder → `[1, 32×30×52, 1408]`.
- Reshape `[32,30,52,1408]`, **mean-pool the 32 tubelets into `T_VJEPA=2` temporal groups** (coarse start..end
  dynamics anchors), store **fp16**: `/home/wan_jepa_feats/clip_XXXXX.pt` = tensor **`[2, 30, 52, 1408]`**.
- **Disk:** 8.78 MB/clip → ~**52.7 GB** for 6000 clips. dim=1408 is fixed by the giant encoder and 30×52 is required
  to match Wan, so `T_VJEPA` is the only knob: `T_VJEPA=2` keeps coarse temporal/dynamics signal (T=1 would kill it)
  while fitting the ~113 GB-free disk with headroom. (Per-token, NOT global-pooled — REPA needs per-token.)
- 2-GPU sharded: `CUDA_VISIBLE_DEVICES=0 python jepa_preproc_encode.py 0 2` (+ `1 2`). One-time. Atomic writes
  (`.tmp`→rename) so it is resumable. Self-test: `... 0 1 --probe`.

## STEP 2 — REPA loss in `/home/wm-vla-rl/_ext/wan_distill.py`  (venv `/home/venv/cosmos_predict2`)
- `RepaProjector`: 2-layer MLP `1536 → 2048 (SiLU) → 1408`, fp32, ~6M params. NEW optimizer param group; saved as a
  companion `wanlin_s2_step{N}.repa.pt`. It is an AUX head — discarded at inference (VBench uses only the DiT).
- Forward hook on **`base.blocks[15]`** (mid; 15 of 30) captures the post-block hidden `[B,32760,1536]` (grad-carrying).
- `data_batches` now also yields `clip_ids` (filename stems) so the loss fetches the **clip-matched**
  `/home/wan_jepa_feats/clip_XXXXX.pt`. `load_jepa_target` interpolates `T_vjepa→21` on the time axis.
- `repa_align_loss`: reshape hidden → `[B,21,30,52,1536]`, project → `[B,21,30,52,1408]`, `mean(1 − cosine)` per token
  vs the interpolated target. Combined `L = MSE + 0.25·L_align`. Logged every step: `distill_mse`, `l_align`,
  `0.25·l_align/mse` ratio. Enabled only via `--jepa_feats <dir>` (default OFF — backward compatible).
- DDP: projector grads are all-reduced with the rest; projector is broadcast from rank-0 at Stage-2 start.

## SMOKE GATE — `jepa_smoke.py` (run before the full run; AUTO-PROCEED iff all pass)
~100 steps, single GPU, real batch=1 regime. Checks: (a) both terms finite; (b) proj shape == target shape
`[B,21,30,52,1408]`; (c) `0.25·l_align/mse` within 0.3–3×; (d) **matched-vs-shuffled** — l_align with the correctly
matched clip target must be clearly lower than with a mismatched clip's target (here the previous step's, a different
clip). Matched ≈ shuffled ⇒ clip mapping broken / features uninformative ⇒ STOP.

## FULL RUN — `wan_s2_jepa_run.sh`
Resume `wanlin_s1_step4000.safetensors`, DDP-2 torchrun, ratio 0.75 / groups 4, lr 1e-4, 6000 steps, GBS 8, λ 0.25,
block 15. Saves to NEW dir `/home/wm-vla-rl/_ext/wan_lin_out/jepa_l025/` (MSE-only ckpts untouched). ~10 h.
