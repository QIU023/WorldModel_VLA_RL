#!/bin/bash
# General-domain (DAVIS natural video) linearization — the SANA-comparable quality track. Teacher is in its
# native strength domain here (unlike OOD LIBERO). Validated recipe: S1 attn-transfer -> S2d output-distill
# (逼近教师), r0.75 ∥ r0.50. Both GPUs saturated, no AC. Run after /home/davis_ready.txt exists.
# Usage: nohup bash cosmos_davis_overnight.sh > cosmos_davis_overnight.log 2>&1 &
set -u
EXT=/home/wm-vla-rl/_ext; PY=/home/venv/cosmos_predict2/bin/python
SNAP=$(ls -d /home/cosmos_predict2_model/models--nvidia--Cosmos-Predict2-2B-Video2World/snapshots/*/ | head -1)
TF="${SNAP}transformer"; DATA=/home/cosmos_davis_data; OUT=$EXT/cosmos_lin_davis; GEN=$EXT/cosmos_gen_davis
N_TRAIN=${N_TRAIN:-512}; N_EVAL=${N_EVAL:-16}
S1_STEPS=${S1_STEPS:-1200}; S2_STEPS=${S2_STEPS:-2000}; S1_BATCH=${S1_BATCH:-8}; S2_BATCH=${S2_BATCH:-8}
GEN_STEPS=${GEN_STEPS:-30}; NF=29; SZ=256
mkdir -p "$OUT" "$GEN"; cd "$EXT"; ts() { date +%H:%M:%S; }
echo "[DAV $(ts)] DAVIS general-domain linearization  N_TRAIN=$N_TRAIN N_EVAL=$N_EVAL S1=$S1_STEPS S2=$S2_STEPS"

# ---- Phase 0: DAVIS data encode (GPU0) ∥ teacher baseline gen (GPU1) ----
echo "[DAV $(ts)] Phase0: data-encode (GPU0) ∥ teacher-gen (GPU1)"
CUDA_VISIBLE_DEVICES=1 $PY cosmos_generate.py --mode teacher --model "$SNAP" --dataset davis --out "$GEN" \
    --n $N_EVAL --num_frames $NF --steps $GEN_STEPS --size $SZ > "$EXT/dav_teacher.log" 2>&1 & PT=$!
if [ ! -f "$DATA/clip_$(printf %05d $((N_TRAIN-1))).pt" ]; then
  CUDA_VISIBLE_DEVICES=0 $PY cosmos_data.py --model "$SNAP" --dataset davis --out "$DATA" --split train \
      --n_clips $N_TRAIN --frames $NF --size $SZ > "$EXT/dav_data.log" 2>&1
  echo "[DAV $(ts)] data-encode rc=$?"
fi
wait $PT; echo "[DAV $(ts)] teacher-gen done -> $(grep '=== teacher' "$EXT/dav_teacher.log" | tail -1)"

# ---- Phase 1: S1 attn-transfer  r0.75(GPU0) ∥ r0.50(GPU1) ----
s1() { CUDA_VISIBLE_DEVICES=$1 $PY cosmos_distill.py --stage 1 --transformer "$TF" --latents "$DATA" \
    --ratio $2 --steps $S1_STEPS --batch $S1_BATCH --lr 1e-3 --save "$3" > "$EXT/dav_s1_r$2.log" 2>&1; }
echo "[DAV $(ts)] Phase1: S1 attn-transfer"
s1 0 0.75 "$OUT/r75" & A=$!; s1 1 0.50 "$OUT/r50" & B=$!; wait $A; wait $B
S1_75=$(ls -t "$OUT/r75"/linstate_s1_step*.safetensors 2>/dev/null | head -1)
S1_50=$(ls -t "$OUT/r50"/linstate_s1_step*.safetensors 2>/dev/null | head -1)
echo "[DAV $(ts)] S1 done. r75=$S1_75 r50=$S1_50"

# ---- Phase 2: S2d output-distill (逼近教师) ----
s2d() { CUDA_VISIBLE_DEVICES=$1 $PY cosmos_distill.py --stage 2 --method distill --transformer "$TF" \
    --latents "$DATA" --ratio $2 --resume "$3" --steps $S2_STEPS --batch $S2_BATCH --lr 2e-4 --save "$4" \
    > "$EXT/dav_s2d_r$2.log" 2>&1; }
echo "[DAV $(ts)] Phase2: S2d output-distill"
s2d 0 0.75 "$S1_75" "$OUT/r75" & A=$!; s2d 1 0.50 "$S1_50" "$OUT/r50" & B=$!; wait $A; wait $B
S2_75=$(ls -t "$OUT/r75"/linstate_s2d_step*.safetensors 2>/dev/null | head -1)
S2_50=$(ls -t "$OUT/r50"/linstate_s2d_step*.safetensors 2>/dev/null | head -1)
echo "[DAV $(ts)] S2d done. r75=$S2_75 r50=$S2_50"

# ---- Phase 3: lineval vs teacher on DAVIS ----
lineval() { CUDA_VISIBLE_DEVICES=$1 $PY cosmos_generate.py --mode lineval --model "$SNAP" --dataset davis \
    --teacher_dir "$GEN/teacher" --resume "$3" --ratio $2 --tag $4 --out "$GEN" --n $N_EVAL \
    --num_frames $NF --steps $GEN_STEPS --size $SZ > "$EXT/dav_eval_$4.log" 2>&1; }
echo "[DAV $(ts)] Phase3: lineval vs teacher"
lineval 0 0.75 "$S2_75" r75_s2d & A=$!; lineval 1 0.50 "$S2_50" r50_s2d & B=$!; wait $A; wait $B

echo "=== [DAV $(ts)] COMPLETE ==="
echo "--- teacher vs GT (general domain) ---"; grep "=== teacher" "$EXT/dav_teacher.log" | tail -1
for t in r75_s2d r50_s2d; do echo "--- $t ---"; grep "逼近教师" "$EXT/dav_eval_$t.log" 2>/dev/null | tail -1; done
