#!/bin/bash
# §3a HEADLINE: Stage-2 output-distill + REPA V-JEPA2 alignment (lambda=0.25), DDP-2, resume Stage-1, 6000 steps.
# Velocity-MSE is UNCHANGED from the MSE-only Stage-2; we only ADD 0.25 * (1 - cos(proj(DiT_block15), VJEPA2_target)).
# Saves to a NEW dir (does NOT touch the MSE-only ckpts). Usage: nohup bash wan_s2_jepa_run.sh & ; tail -f wan_s2_jepa.log
set -u
EXT=/home/wm-vla-rl/_ext; CF=$EXT/Causal-Forcing; PY=/home/venv/cosmos_predict2/bin/python
DATA=/home/wan_distill_data; JEPA=/home/wan_jepa_feats
S1=$EXT/wan_lin_out/wanlin_s1_step4000.safetensors
OUT=$EXT/wan_lin_out/jepa_l025
RATIO=0.75; NGRP=4; STEPS=6000; BATCH=1; ACCUM=4; LR=1e-4; LAM=0.25; BLOCK=15
mkdir -p "$OUT"; cd "$CF"
echo "[jepa $(date +%H:%M:%S)] S2+REPA lam=$LAM block=$BLOCK resume=$S1 steps=$STEPS lr=$LR GBS=$((BATCH*ACCUM*2)) -> $OUT"
$PY -m torch.distributed.run --nproc_per_node=2 --master_port=29571 $EXT/wan_distill.py \
    --stage 2 --latents "$DATA" --ratio $RATIO --groups $NGRP --resume "$S1" \
    --jepa_feats "$JEPA" --jepa_lambda $LAM --jepa_block $BLOCK \
    --steps $STEPS --batch $BATCH --accum $ACCUM --lr $LR --save "$OUT"
echo "[jepa $(date +%H:%M:%S)] DONE -> $(ls -t "$OUT"/wanlin_s2_step*.safetensors 2>/dev/null | head -1)"
