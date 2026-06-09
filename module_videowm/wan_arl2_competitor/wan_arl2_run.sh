#!/bin/bash
# ARL2-competitor full run: linearize Causal Forcing (Wan2.1-1.3B AR) with intra-softmax + inter-KDA + AttnRes,
# distilled with ARL2's 2-stage recipe (attn-transfer 6K -> output-distill 12K), DDP-2, then VBench eval.
# Usage: nohup bash wan_arl2_run.sh > wan_arl2_run.log 2>&1 &
set -u
EXT=/home/wm-vla-rl/_ext; CF=$EXT/Causal-Forcing; PY=/home/venv/cosmos_predict2/bin/python
DATA=/home/wan_distill_data; OUT=$EXT/wan_lin_out
RATIO=${RATIO:-0.75}; NGRP=${NGRP:-4}
S1_STEPS=${S1_STEPS:-6000}; S2_STEPS=${S2_STEPS:-12000}
BATCH=${BATCH:-1}; ACCUM=${ACCUM:-4}          # GBS = batch * accum * world(2) = 8 (ARL2/LoLCATs-scale)
S1_LR=${S1_LR:-1e-3}; S2_LR=${S2_LR:-2e-4}
N_DATA=${N_DATA:-512}
mkdir -p "$OUT"; cd "$CF"; ts() { date +%H:%M:%S; }
TR="$PY -m torch.distributed.run --nproc_per_node=2 --master_port=29561 $EXT/wan_distill.py"
echo "[arl2 $(ts)] ratio=$RATIO groups=$NGRP  S1=$S1_STEPS/lr$S1_LR  S2=$S2_STEPS/lr$S2_LR  GBS=$((BATCH*ACCUM*2))"

# ---- Phase 0: ensure data (teacher-synthesized latents) ----
have=$(ls "$DATA"/clip_*.pt 2>/dev/null | wc -l)
if [ "$have" -lt "$N_DATA" ]; then
  echo "[arl2 $(ts)] data: have $have / $N_DATA -> generating remainder (GPU0)"
  CUDA_VISIBLE_DEVICES=0 $PY $EXT/wan_data.py --config configs/causal_forcing_dmd_framewise.yaml \
      --prompts prompts/demos.txt --n $N_DATA --frames 21 --seeds 6 --out "$DATA" > "$EXT/wan_data_gen2.log" 2>&1
fi
echo "[arl2 $(ts)] data ready: $(ls "$DATA"/clip_*.pt | wc -l) clips"

# ---- Phase 1: Stage-1 per-layer attn-transfer (DDP-2) ----
echo "[arl2 $(ts)] Phase1: Stage-1 attn-transfer (DDP-2)"
$TR --stage 1 --latents "$DATA" --ratio $RATIO --groups $NGRP \
    --steps $S1_STEPS --batch $BATCH --accum $ACCUM --lr $S1_LR --save "$OUT" > "$EXT/wan_s1.log" 2>&1
S1=$(ls -t "$OUT"/wanlin_s1_step*.safetensors 2>/dev/null | head -1)
echo "[arl2 $(ts)] Stage-1 done -> $S1"

# ---- Phase 2: Stage-2 output-distill (DDP-2), warm-started from Stage-1 ----
echo "[arl2 $(ts)] Phase2: Stage-2 output-distill (DDP-2)"
$TR --stage 2 --latents "$DATA" --ratio $RATIO --groups $NGRP --resume "$S1" \
    --steps $S2_STEPS --batch $BATCH --accum $ACCUM --lr $S2_LR --save "$OUT" > "$EXT/wan_s2.log" 2>&1
S2=$(ls -t "$OUT"/wanlin_s2_step*.safetensors 2>/dev/null | head -1)
echo "[arl2 $(ts)] Stage-2 done -> student ckpt: $S2"

echo "=== [arl2 $(ts)] DISTILL COMPLETE. student=$S2 ==="
echo "next: VBench eval (wan_eval.py) — generate teacher vs student videos @832x480, frame lengths 81/165/501/921"
