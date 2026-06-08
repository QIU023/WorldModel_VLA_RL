#!/bin/bash
# Reoriented overnight (per directive): downstream video-gen task, teacher baseline -> linearize-distill sweep
# -> distilled ckpt approaches teacher (逼近). Both GPUs saturated throughout (parallel jobs).
# Task: image+text->future-frame prediction on held-out LIBERO. Metric: lin-vs-teacher fidelity + vs-GT.
# Usage: nohup bash cosmos_overnight2.sh > cosmos_overnight2.log 2>&1 &
set -u
EXT=/home/wm-vla-rl/_ext; PY=/home/venv/cosmos_predict2/bin/python
SNAP=$(ls -d /home/cosmos_predict2_model/models--nvidia--Cosmos-Predict2-2B-Video2World/snapshots/*/ | head -1)
TF="${SNAP}transformer"
DATA=/home/cosmos_lin_data; OUT=$EXT/cosmos_lin_out; GEN=$EXT/cosmos_gen_out
N_TRAIN=${N_TRAIN:-1024}; N_EVAL=${N_EVAL:-24}
S1_STEPS=${S1_STEPS:-1500}; S2_STEPS=${S2_STEPS:-2000}
S1_BATCH=${S1_BATCH:-8}; S2_BATCH=${S2_BATCH:-16}
GEN_STEPS=${GEN_STEPS:-30}; NF=${NF:-29}; SZ=${SZ:-256}
mkdir -p "$OUT" "$GEN"; cd "$EXT"
ts() { date +%H:%M:%S; }
echo "[ON2 $(ts)] snapshot=$SNAP  N_TRAIN=$N_TRAIN N_EVAL=$N_EVAL  S1=$S1_STEPS/b$S1_BATCH S2=$S2_STEPS/b$S2_BATCH"

# ---- Phase 0: teacher baseline gen (GPU1)  ∥  train-data encode (GPU0) ----
echo "[ON2 $(ts)] Phase0: teacher-gen (GPU1) ∥ data-encode (GPU0)"
CUDA_VISIBLE_DEVICES=1 $PY cosmos_generate.py --mode teacher --model "$SNAP" --out "$GEN" \
    --n $N_EVAL --num_frames $NF --steps $GEN_STEPS --size $SZ > "$EXT/on2_teacher.log" 2>&1 &
PT=$!
if [ ! -f "$DATA/clip_$(printf %05d $((N_TRAIN-1))).pt" ]; then
  CUDA_VISIBLE_DEVICES=0 $PY cosmos_data.py --model "$SNAP" --out "$DATA" --split train \
      --n_clips $N_TRAIN --frames $NF --size $SZ > "$EXT/on2_data.log" 2>&1 &
  PD=$!; wait $PD; echo "[ON2 $(ts)] data-encode done rc=$?"
else echo "[ON2 $(ts)] data exists, skip encode"; fi
wait $PT; echo "[ON2 $(ts)] teacher-gen done rc=$?  -> $(grep vs_GT "$EXT/on2_teacher.log" | tail -1)"

# ---- Phase 1: distill sweep, ratio 0.75 (GPU0) ∥ ratio 0.50 (GPU1), Stage-1 attn-transfer ----
distill_s1() { # $1=gpu $2=ratio $3=outdir
  CUDA_VISIBLE_DEVICES=$1 $PY cosmos_distill.py --stage 1 --transformer "$TF" --latents "$DATA" \
    --ratio $2 --steps $S1_STEPS --batch $S1_BATCH --lr 1e-3 --save "$3" > "$EXT/on2_s1_r$2.log" 2>&1
}
echo "[ON2 $(ts)] Phase1: Stage-1 attn-transfer  r0.75(GPU0) ∥ r0.50(GPU1)"
distill_s1 0 0.75 "$OUT/r75" & A=$!
distill_s1 1 0.50 "$OUT/r50" & B=$!
wait $A; wait $B
S1_75=$(ls -t "$OUT/r75"/linstate_s1_step*.safetensors 2>/dev/null | head -1)
S1_50=$(ls -t "$OUT/r50"/linstate_s1_step*.safetensors 2>/dev/null | head -1)
echo "[ON2 $(ts)] Stage-1 done. r75=$S1_75  r50=$S1_50"

# ---- Phase 2: lineval Stage-1 vs teacher (逼近教师) ----
lineval() { # $1=gpu $2=ratio $3=linstate $4=tag
  CUDA_VISIBLE_DEVICES=$1 $PY cosmos_generate.py --mode lineval --model "$SNAP" --teacher_dir "$GEN/teacher" \
    --resume "$3" --ratio $2 --tag $4 --out "$GEN" --n $N_EVAL --num_frames $NF --steps $GEN_STEPS --size $SZ \
    > "$EXT/on2_eval_$4.log" 2>&1
}
echo "[ON2 $(ts)] Phase2: lineval Stage-1"
lineval 0 0.75 "$S1_75" r75_s1 & A=$!
lineval 1 0.50 "$S1_50" r50_s1 & B=$!
wait $A; wait $B

# ---- Phase 3: Stage-2 flow finetune (resume S1) ----
distill_s2() { # $1=gpu $2=ratio $3=resume $4=outdir
  CUDA_VISIBLE_DEVICES=$1 $PY cosmos_distill.py --stage 2 --transformer "$TF" --latents "$DATA" \
    --ratio $2 --resume "$3" --steps $S2_STEPS --batch $S2_BATCH --lr 2e-4 --save "$4" > "$EXT/on2_s2_r$2.log" 2>&1
}
echo "[ON2 $(ts)] Phase3: Stage-2 flow-finetune  r0.75(GPU0) ∥ r0.50(GPU1)"
distill_s2 0 0.75 "$S1_75" "$OUT/r75" & A=$!
distill_s2 1 0.50 "$S1_50" "$OUT/r50" & B=$!
wait $A; wait $B
S2_75=$(ls -t "$OUT/r75"/linstate_s2_step*.safetensors 2>/dev/null | head -1)
S2_50=$(ls -t "$OUT/r50"/linstate_s2_step*.safetensors 2>/dev/null | head -1)
echo "[ON2 $(ts)] Stage-2 done. r75=$S2_75  r50=$S2_50"

# ---- Phase 4: lineval Stage-2 ----
echo "[ON2 $(ts)] Phase4: lineval Stage-2"
lineval 0 0.75 "$S2_75" r75_s2 & A=$!
lineval 1 0.50 "$S2_50" r50_s2 & B=$!
wait $A; wait $B

# ---- Phase 5: efficiency O(N) vs O(N^2) (GPU0) ----
echo "[ON2 $(ts)] Phase5: efficiency curve"
CUDA_VISIBLE_DEVICES=0 $PY cosmos_eval.py --mode efficiency --transformer "$TF" --ratio 0.75 \
    --resume "$S2_75" --frames 4,8,16,24,32,48,64 > "$EXT/on2_efficiency.log" 2>&1

# ---- report ----
echo "=== [ON2 $(ts)] COMPLETE ==="
echo "--- teacher vs GT ---";        grep "=== teacher" "$EXT/on2_teacher.log" | tail -1
for t in r75_s1 r50_s1 r75_s2 r50_s2; do
  echo "--- $t ---"; grep "===" "$EXT/on2_eval_$t.log" | tail -1
done
echo "--- efficiency ---"; tail -3 "$EXT/on2_efficiency.log"
