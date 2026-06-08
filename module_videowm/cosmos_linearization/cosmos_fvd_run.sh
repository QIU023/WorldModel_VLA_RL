#!/bin/bash
# FVD eval (SANA/Cosmos-comparable metric): generate N teacher videos (GPU0) ∥ N linearized r0.75 videos
# (GPU1) on DAVIS, then score S3D-Fréchet. Distillation judged complete -> evaluate.
# Usage: nohup bash cosmos_fvd_run.sh > cosmos_fvd_run.log 2>&1 &
set -u
EXT=/home/wm-vla-rl/_ext; PY=/home/venv/cosmos_predict2/bin/python
SNAP=$(ls -d /home/cosmos_predict2_model/models--nvidia--Cosmos-Predict2-2B-Video2World/snapshots/*/ | head -1)
S2D="$EXT/cosmos_lin_davis/r75/linstate_s2d_step2000.safetensors"   # DAVIS-distilled 75% ckpt
OUT="$EXT/cosmos_fvd_out"; N=${N:-96}; DS=${DS:-davis}
cd "$EXT"; ts() { date +%H:%M:%S; }
echo "[FVD $(ts)] gen $N teacher(GPU0) ∥ $N lin-r0.75(GPU1) on $DS"
CUDA_VISIBLE_DEVICES=0 $PY cosmos_fvd.py --mode teacher --model "$SNAP" --dataset $DS --out "$OUT" \
    --tag teacher --n $N > "$EXT/fvd_gen_teacher.log" 2>&1 & A=$!
CUDA_VISIBLE_DEVICES=1 $PY cosmos_fvd.py --mode lin --model "$SNAP" --dataset $DS --resume "$S2D" --ratio 0.75 \
    --out "$OUT" --tag r75_lin --n $N > "$EXT/fvd_gen_lin.log" 2>&1 & B=$!
wait $A; wait $B
echo "[FVD $(ts)] generation done -> scoring"
CUDA_VISIBLE_DEVICES=0 $PY cosmos_fvd.py --mode score --dataset $DS --out "$OUT" --tag r75_lin > "$EXT/fvd_score.log" 2>&1
echo "=== [FVD $(ts)] DONE ==="; grep -A4 "=== FVD" "$EXT/fvd_score.log" 2>/dev/null
