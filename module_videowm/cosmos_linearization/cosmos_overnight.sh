#!/bin/bash
# Overnight Cosmos-Predict2-2B linearization chain: data -> Stage-1 attn-transfer -> Stage-2 flow finetune -> eval.
# Run AFTER the LIBERO eval frees the GPU. Logs each stage; aborts the chain if a stage fails.
# Usage: CUDA_VISIBLE_DEVICES=0 nohup bash cosmos_overnight.sh > cosmos_overnight.log 2>&1 &
set -u
EXT=/home/wm-vla-rl/_ext
PY=/home/venv/cosmos_predict2/bin/python
SNAP=$(ls -d /home/cosmos_predict2_model/models--nvidia--Cosmos-Predict2-2B-Video2World/snapshots/*/ | head -1)
TF="${SNAP}transformer"
DATA=/home/cosmos_lin_data
DATA_EVAL=/home/cosmos_lin_data_eval
OUT=$EXT/cosmos_lin_out
RATIO=${RATIO:-0.75}
S1_STEPS=${S1_STEPS:-2000}
S2_STEPS=${S2_STEPS:-3000}
BATCH=${BATCH:-4}            # clips/step (H200 has the room); OOM-skip backstops if a batch is too big
N_TRAIN=${N_TRAIN:-1024}     # train clips (multiple windows/traj) for a credible Stage-2 finetune
mkdir -p "$OUT"
cd "$EXT"
echo "[overnight] $(date) snapshot=$SNAP ratio=$RATIO"

run() { echo "=== [overnight] $1 :: $(date) ==="; shift; "$@"; rc=$?; if [ $rc -ne 0 ]; then echo "[overnight] FAILED rc=$rc, aborting chain"; exit $rc; fi; }

# 1. data (train + held-out eval)
if [ ! -f "$DATA/clip_$(printf %05d $((N_TRAIN-1))).pt" ]; then
  run "data-train" $PY cosmos_data.py --model "$SNAP" --out "$DATA" --split train --n_clips $N_TRAIN --frames 29 --size 256
else echo "[overnight] train data exists, skip"; fi
if [ ! -f "$DATA_EVAL/clip_00031.pt" ]; then
  run "data-eval" $PY cosmos_data.py --model "$SNAP" --out "$DATA_EVAL" --split eval --n_clips 32 --frames 29 --size 256
else echo "[overnight] eval data exists, skip"; fi

# 2. Stage-1 attn-transfer (train Hedgehog phi to match softmax)
run "stage1" $PY cosmos_distill.py --stage 1 --transformer "$TF" --latents "$DATA" --ratio "$RATIO" \
    --steps $S1_STEPS --batch $BATCH --lr 1e-3 --save "$OUT"
S1=$(ls -t "$OUT"/linstate_s1_step*.safetensors 2>/dev/null | head -1)
echo "[overnight] Stage-1 ckpt: $S1"

# 3. Stage-2 flow-matching finetune (warm-start from Stage-1)
run "stage2" $PY cosmos_distill.py --stage 2 --transformer "$TF" --latents "$DATA" --ratio "$RATIO" \
    --resume "$S1" --steps $S2_STEPS --batch $BATCH --lr 2e-4 --save "$OUT"
S2=$(ls -t "$OUT"/linstate_s2_step*.safetensors 2>/dev/null | head -1)
echo "[overnight] Stage-2 ckpt: $S2"

# 4. eval: efficiency (O(N) vs O(N^2)) + quality (velocity-MSE vs teacher), Stage-1 and Stage-2
run "eval-efficiency" $PY cosmos_eval.py --mode efficiency --transformer "$TF" --ratio "$RATIO" \
    --resume "$S2" --frames 4,8,16,24,32,48,64
run "eval-quality-s1" $PY cosmos_eval.py --mode quality --transformer "$TF" --ratio "$RATIO" \
    --resume "$S1" --latents "$DATA_EVAL"
mv cosmos_eval_out/quality.json cosmos_eval_out/quality_s1.json 2>/dev/null
run "eval-quality-s2" $PY cosmos_eval.py --mode quality --transformer "$TF" --ratio "$RATIO" \
    --resume "$S2" --latents "$DATA_EVAL"
mv cosmos_eval_out/quality.json cosmos_eval_out/quality_s2.json 2>/dev/null

echo "=== [overnight] CHAIN COMPLETE $(date) ==="
echo "S1=$S1"; echo "S2=$S2"
echo "--- efficiency ---"; cat cosmos_eval_out/efficiency.json
echo "--- quality s1 ---"; cat cosmos_eval_out/quality_s1.json
echo "--- quality s2 ---"; cat cosmos_eval_out/quality_s2.json
