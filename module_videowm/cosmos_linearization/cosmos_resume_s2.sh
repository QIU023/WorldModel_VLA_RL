#!/bin/bash
# Resume from saved Stage-1 ckpts (teacher baseline + Stage-1 + Stage-1 lineval already done & preserved).
# Replaces the broken batch-16 flow Stage-2 with OUTPUT-DISTILLATION (逼近教师: match pristine teacher velocity)
# at batch 8 (fits; Stage-1 fit at 8). r0.75(GPU0) ∥ r0.50(GPU1) -> lineval vs teacher -> efficiency.
# Usage: nohup bash cosmos_resume_s2.sh > cosmos_resume_s2.log 2>&1 &
set -u
EXT=/home/wm-vla-rl/_ext; PY=/home/venv/cosmos_predict2/bin/python
SNAP=$(ls -d /home/cosmos_predict2_model/models--nvidia--Cosmos-Predict2-2B-Video2World/snapshots/*/ | head -1)
TF="${SNAP}transformer"; DATA=/home/cosmos_lin_data; OUT=$EXT/cosmos_lin_out; GEN=$EXT/cosmos_gen_out
S2_STEPS=${S2_STEPS:-2000}; S2_BATCH=${S2_BATCH:-8}; GEN_STEPS=${GEN_STEPS:-30}; NF=29; SZ=256; N_EVAL=24
S1_75=$OUT/r75/linstate_s1_step1500.safetensors
S1_50=$OUT/r50/linstate_s1_step1500.safetensors
cd "$EXT"; ts() { date +%H:%M:%S; }
echo "[RS2 $(ts)] output-distill Stage-2  S2=$S2_STEPS/b$S2_BATCH  S1_75=$S1_75 S1_50=$S1_50"

# ---- Phase A: output-distill Stage-2  r0.75(GPU0) ∥ r0.50(GPU1) ----
s2d() { CUDA_VISIBLE_DEVICES=$1 $PY cosmos_distill.py --stage 2 --method distill --transformer "$TF" \
    --latents "$DATA" --ratio $2 --resume "$3" --steps $S2_STEPS --batch $S2_BATCH --lr 2e-4 --save "$4" \
    > "$EXT/on2_s2d_r$2.log" 2>&1; }
echo "[RS2 $(ts)] PhaseA: output-distill  r0.75(GPU0) ∥ r0.50(GPU1)"
s2d 0 0.75 "$S1_75" "$OUT/r75" & A=$!
s2d 1 0.50 "$S1_50" "$OUT/r50" & B=$!
wait $A; wait $B
S2D_75=$(ls -t "$OUT/r75"/linstate_s2d_step*.safetensors 2>/dev/null | head -1)
S2D_50=$(ls -t "$OUT/r50"/linstate_s2d_step*.safetensors 2>/dev/null | head -1)
echo "[RS2 $(ts)] Stage-2 done. r75=$S2D_75  r50=$S2D_50"

# ---- Phase B: lineval Stage-2-distill vs teacher ----
lineval() { CUDA_VISIBLE_DEVICES=$1 $PY cosmos_generate.py --mode lineval --model "$SNAP" \
    --teacher_dir "$GEN/teacher" --resume "$3" --ratio $2 --tag $4 --out "$GEN" --n $N_EVAL \
    --num_frames $NF --steps $GEN_STEPS --size $SZ > "$EXT/on2_eval_$4.log" 2>&1; }
echo "[RS2 $(ts)] PhaseB: lineval Stage-2-distill"
lineval 0 0.75 "$S2D_75" r75_s2d & A=$!
lineval 1 0.50 "$S2D_50" r50_s2d & B=$!
wait $A; wait $B

# ---- Phase C: efficiency ----
echo "[RS2 $(ts)] PhaseC: efficiency"
CUDA_VISIBLE_DEVICES=0 $PY cosmos_eval.py --mode efficiency --transformer "$TF" --ratio 0.75 \
    --resume "$S2D_75" --frames 4,8,16,24,32,48,64 > "$EXT/on2_efficiency.log" 2>&1

# ---- report ----
echo "=== [RS2 $(ts)] COMPLETE ==="
echo "--- teacher vs GT ---"; grep "=== teacher" "$EXT/on2_teacher.log" | tail -1
for t in r75_s1 r50_s1 r75_s2d r50_s2d; do echo "--- $t ---"; grep "逼近教师" "$EXT/on2_eval_$t.log" 2>/dev/null | tail -1; done
echo "--- efficiency ---"; tail -3 "$EXT/on2_efficiency.log"
