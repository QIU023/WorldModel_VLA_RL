#!/bin/bash
# Hands-off ARL2-competitor launch: WAIT for the GPU1 teacher data-gen to reach N_DATA clips, then run the
# 2-stage distillation on BOTH GPUs (DDP-2). No own generation (avoids racing the running generator).
# Usage: nohup bash wan_arl2_launch.sh > wan_arl2_launch.log 2>&1 &
set -u
EXT=/home/wm-vla-rl/_ext; CF=$EXT/Causal-Forcing; PY=/home/venv/cosmos_predict2/bin/python
DATA=/home/wan_distill_data; OUT=$EXT/wan_lin_out
RATIO=0.75; NGRP=4; S1_STEPS=6000; S2_STEPS=12000; BATCH=1; ACCUM=4; S1_LR=1e-4; S2_LR=1e-4; N_DATA=512
WD=0.1; CLIP=4   # sweep-selected: lr 1e-4 (1e-3 plateaued, 1e-5 too small); WorldVLA-KDA wd/clip; cosine default
mkdir -p "$OUT"; cd "$CF"; ts() { date +%H:%M:%S; }
TR="$PY -m torch.distributed.run --nproc_per_node=2 --master_port=29561 $EXT/wan_distill.py"

echo "[launch $(ts)] waiting for $N_DATA teacher clips (GPU1 data-gen in progress)..."
while [ "$(ls "$DATA"/clip_*.pt 2>/dev/null | wc -l)" -lt "$N_DATA" ]; do sleep 60; done
# ensure no data-gen process is still holding GPU memory before claiming both GPUs for DDP-2
pkill -f wan_data.py 2>/dev/null; sleep 5
echo "[launch $(ts)] data ready: $(ls "$DATA"/clip_*.pt | wc -l) clips. Starting DDP-2 distillation."

echo "[launch $(ts)] Stage-1 attn-transfer (DDP-2, $S1_STEPS steps)"
$TR --stage 1 --latents "$DATA" --ratio $RATIO --groups $NGRP --wd $WD --clip $CLIP \
    --steps $S1_STEPS --batch $BATCH --accum $ACCUM --lr $S1_LR --save "$OUT" > "$EXT/wan_s1.log" 2>&1
S1=$(ls -t "$OUT"/wanlin_s1_step*.safetensors 2>/dev/null | head -1)
if [ -z "$S1" ]; then echo "[launch $(ts)] ABORT: Stage-1 produced no checkpoint (see wan_s1.log)"; exit 1; fi
echo "[launch $(ts)] Stage-1 done -> $S1"

echo "[launch $(ts)] Stage-2 output-distill (DDP-2, $S2_STEPS steps), warm-start $S1"
$TR --stage 2 --latents "$DATA" --ratio $RATIO --groups $NGRP --resume "$S1" --wd $WD --clip $CLIP \
    --steps $S2_STEPS --batch $BATCH --accum $ACCUM --lr $S2_LR --save "$OUT" > "$EXT/wan_s2.log" 2>&1
S2=$(ls -t "$OUT"/wanlin_s2_step*.safetensors 2>/dev/null | head -1)
echo "=== [launch $(ts)] DISTILL COMPLETE. student ckpt: $S2 ==="
