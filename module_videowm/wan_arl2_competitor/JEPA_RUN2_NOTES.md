# §3a JEPA Run-2: fix T=2 dynamics-suppression with more temporal resolution

## Motivation (run-1 result)
Vanilla REPA V-JEPA2 alignment (λ=0.25, tap Wan DiT block15, resume S1 → 6000 steps, lr 1e-4) with the target
encoded as `[T_VJEPA=2, 30, 52, 1408]` (full spatial, temporally mean-pooled to 2 slices).

VBench Quality-7: **MSE-only 83.84 → JEPA-T2 83.90** (net +0.06, FLAT). Per-dim: spatial/perceptual all UP
(subject_consistency, background, aesthetic, imaging) but **dynamic_degree DOWN −5.55** (83.33 → 77.78).

Root cause: T=2 → the target interpolated 2→21 on the time axis is temporally SMOOTH → it pushes the student's
per-frame features to look alike across time → suppresses motion.

## Run-2 design (the fix)
Re-encode the V-JEPA2 target with **T_VJEPA=8** temporal slices so the target is NOT artificially smooth.
To keep disk equal, spatial 2×2 avg-pool the patch grid 30×52 → 15×26 BEFORE the temporal mean-pool.
Note 8·(15·26) == 2·(30·52) == 3120, so T=8@15×26 stores at the SAME 8.79 MB/clip (~52.7 GB total) as run-1's
T=2@30×52 — but with 4× the temporal slices.

Everything else identical to run-1 to isolate the T effect: λ=0.25, block15, resume S1 step4000, 6000 steps,
lr 1e-4, DDP-2, centering OFF (vanilla; `JEPA_CENTER` opt-in default off), velocity-MSE unchanged.

### Code changes
- `jepa_preproc_encode.py`: `JEPA_T` default 8; add spatial 2×2 avg-pool (`F.avg_pool2d`, env `JEPA_SPATIAL_POOL=2`)
  → stored `[8,15,26,1408]` fp16.
- `wan_distill.py`: new shared helper `pool_hidden_spatial()` 2×2-avg-pools the Wan DiT hidden
  `[B,21,30,52,dim]→[B,21,15,26,dim]`; `repa_align_loss` now aligns at the 15×26 grid (project AFTER pooling).
  `load_jepa_target` is grid-agnostic (reads spatial dims from the stored tensor) → loads `[8,15,26,1408]`,
  interpolates temporal 8→21, spatial 15×26 preserved 1:1. Projector dim_wan→1408 unchanged.
- `jepa_smoke.py`: uses the shared `pool_hidden_spatial()` (single code path with training).
- `wan_s2_jepa_run.sh`: OUT → `jepa_l025_T8/` (run-1's `jepa_l025/` untouched); `--log_every 1`.
- `jepa_encode_t8.sh`: frees old T=2 feats, 2-GPU sharded T=8 re-encode.
- `wan_eval/rescore_jepa_t8.sh`: regen 944 from T8 ckpt + Quality-7 + compare vs teacher/MSE-only/JEPA-T2.

## Milestones
- **Re-encode (probe)**: `(8, 15, 26, 1408)` fp16, 8.79 MB/clip, est total **52.7 GB** for 6000 clips. ✓
- **Re-encode (full)**: 2-GPU sharded from `/home/wan_distill_data_mp4` → `/home/wan_jepa_feats`. (in progress)
- **Smoke**: TBD
- **Training**: TBD
- **Eval (Quality-7)**: TBD — teacher 88.09 / MSE-only 83.84 / JEPA-T2 83.90 / **JEPA-T8 ?**
  KEY: did dynamic_degree recover while keeping the spatial gains?
