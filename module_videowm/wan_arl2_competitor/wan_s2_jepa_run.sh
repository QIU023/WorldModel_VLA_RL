#!/bin/bash
# §3a HEADLINE (RUN-2): Stage-2 output-distill + REPA V-JEPA2 alignment (lambda=0.25), DDP-2, resume Stage-1, 6000 steps.
# Velocity-MSE is UNCHANGED from the MSE-only Stage-2; we only ADD 0.25 * (1 - cos(proj(pool(DiT_block15)), VJEPA2_target)).
# RUN-2: target re-encoded with T_VJEPA=8 (vs run-1's T=2) @ 2x2-pooled 15x26 grid -> fixes run-1's dynamic_degree
# suppression. Saves to a NEW dir jepa_l025_T8/ (does NOT touch run-1's jepa_l025/). log_every=1 (distill_mse+l_align
# each step, per task). Usage: nohup bash wan_s2_jepa_run.sh & ; tail -f wan_s2_jepa_T8.log
set -u
EXT=/home/wm-vla-rl/_ext; CF=$EXT/Causal-Forcing; PY=/home/venv/cosmos_predict2/bin/python
DATA=/home/wan_distill_data; JEPA=/home/wan_jepa_feats
S1=$EXT/wan_lin_out/wanlin_s1_step4000.safetensors
OUT=$EXT/wan_lin_out/jepa_l025_T8
RATIO=0.75; NGRP=4; STEPS=6000; BATCH=1; ACCUM=4; LR=1e-4; LAM=0.25; BLOCK=15
mkdir -p "$OUT"; cd "$CF"
echo "[jepa-T8 $(date +%H:%M:%S)] S2+REPA(T=8) lam=$LAM block=$BLOCK resume=$S1 steps=$STEPS lr=$LR GBS=$((BATCH*ACCUM*2)) -> $OUT"
$PY -m torch.distributed.run --nproc_per_node=2 --master_port=29571 $EXT/wan_distill.py \
    --stage 2 --latents "$DATA" --ratio $RATIO --groups $NGRP --resume "$S1" \
    --jepa_feats "$JEPA" --jepa_lambda $LAM --jepa_block $BLOCK --log_every 1 \
    --steps $STEPS --batch $BATCH --accum $ACCUM --lr $LR --save "$OUT"
echo "[jepa-T8 $(date +%H:%M:%S)] DONE -> $(ls -t "$OUT"/wanlin_s2_step*.safetensors 2>/dev/null | head -1)"
