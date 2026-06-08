#!/bin/bash
# DAVIS @ NATIVE 480px three-way prep: re-encode 480p data, re-distill student (r0.75) at 480p, generate
# teacher + student @480p, then FVD + VBench + lin-vs-teacher. Native DAVIS res (no upscale), ~hours not days.
# Both GPUs: distill on GPU0 ∥ teacher FVD-gen on GPU1. Usage: nohup bash cosmos_davis480.sh > cosmos_davis480.log 2>&1 &
set -u
EXT=/home/wm-vla-rl/_ext; PY=/home/venv/cosmos_predict2/bin/python
SNAP=$(ls -d /home/cosmos_predict2_model/models--nvidia--Cosmos-Predict2-2B-Video2World/snapshots/*/ | head -1)
TF="${SNAP}transformer"; DATA=/home/cosmos_davis480_data; OUT=$EXT/cosmos_lin_davis480
FVDO=$EXT/cosmos_fvd480_out; GEN=$EXT/cosmos_gen_davis480
SZ=480; NF=29; N_TRAIN=${N_TRAIN:-512}; N_FVD=${N_FVD:-96}; N_EVAL=16
# GBS=8 aligned w/ committed 256px, via DDP-2: GBS = micro(2) * accum(2) * world_size(2) = 8. Both GPUs train.
S1_STEPS=${S1_STEPS:-1500}; S2_STEPS=${S2_STEPS:-2000}; S1_BATCH=${S1_BATCH:-2}; S2_BATCH=${S2_BATCH:-2}
S1_ACCUM=${S1_ACCUM:-2}; S2_ACCUM=${S2_ACCUM:-2}; GEN_STEPS=30
mkdir -p "$OUT" "$FVDO" "$GEN"; cd "$EXT"; ts() { date +%H:%M:%S; }
echo "[D480 $(ts)] DAVIS @${SZ}px  N_TRAIN=$N_TRAIN N_FVD=$N_FVD S1=$S1_STEPS/b$S1_BATCH S2=$S2_STEPS/b$S2_BATCH"

# ---- Phase 0: encode 480p data (GPU0) ----
if [ ! -f "$DATA/clip_$(printf %05d $((N_TRAIN-1))).pt" ]; then
  echo "[D480 $(ts)] encode 480p data (GPU0)"
  CUDA_VISIBLE_DEVICES=0 $PY cosmos_data.py --model "$SNAP" --dataset davis --out "$DATA" --split train \
      --n_clips $N_TRAIN --frames $NF --size $SZ > "$EXT/d480_data.log" 2>&1
  echo "[D480 $(ts)] data rc=$?"
fi

# ---- Phase 1: distill r0.75 @480p (GPU0)  ∥  teacher FVD-gen 96 @480p (GPU1) ----
GBS=$((S1_BATCH*S1_ACCUM*2)); echo "[D480 $(ts)] Phase1: distill r0.75 DDP-2 GBS=$GBS (both GPUs)"
# teacher videos already generated (preserved); distill now uses BOTH GPUs via torchrun
TR="$PY -m torch.distributed.run --nproc_per_node=2 --master_port=29551"
$TR cosmos_distill.py --stage 1 --transformer "$TF" --latents "$DATA" --ratio 0.75 \
    --steps $S1_STEPS --batch $S1_BATCH --accum $S1_ACCUM --lr 1e-3 --save "$OUT" > "$EXT/d480_s1.log" 2>&1
S1=$(ls -t "$OUT"/linstate_s1_step*.safetensors 2>/dev/null | head -1)
$TR cosmos_distill.py --stage 2 --method distill --transformer "$TF" --latents "$DATA" \
    --ratio 0.75 --resume "$S1" --steps $S2_STEPS --batch $S2_BATCH --accum $S2_ACCUM --lr 2e-4 --save "$OUT" > "$EXT/d480_s2d.log" 2>&1
echo "[D480 $(ts)] distill done (DDP-2)"
S2D=$(ls -t "$OUT"/linstate_s2d_step*.safetensors 2>/dev/null | head -1)
echo "[D480 $(ts)] student ckpt: $S2D"

# ---- Phase 2: student FVD-gen 96 @480p (GPU0) ----
echo "[D480 $(ts)] Phase2: student gen $N_FVD (GPU0)"
CUDA_VISIBLE_DEVICES=0 $PY cosmos_fvd.py --mode lin --model "$SNAP" --dataset davis --resume "$S2D" --ratio 0.75 \
    --out "$FVDO" --tag r75_lin --n $N_FVD --size $SZ --steps $GEN_STEPS > "$EXT/d480_student_gen.log" 2>&1

# ---- Phase 3: FVD score + lin-vs-teacher lineval + VBench ----
echo "[D480 $(ts)] Phase3: FVD + lineval + VBench"
CUDA_VISIBLE_DEVICES=0 $PY cosmos_fvd.py --mode score --dataset davis --out "$FVDO" --tag r75_lin --size $SZ > "$EXT/d480_fvd_score.log" 2>&1
CUDA_VISIBLE_DEVICES=1 $PY cosmos_generate.py --mode teacher --model "$SNAP" --dataset davis --out "$GEN" \
    --n $N_EVAL --num_frames $NF --steps $GEN_STEPS --size $SZ > "$EXT/d480_lineval_teacher.log" 2>&1
CUDA_VISIBLE_DEVICES=1 $PY cosmos_generate.py --mode lineval --model "$SNAP" --dataset davis --teacher_dir "$GEN/teacher" \
    --resume "$S2D" --ratio 0.75 --tag r75_s2d --out "$GEN" --n $N_EVAL --num_frames $NF --steps $GEN_STEPS --size $SZ > "$EXT/d480_lineval_lin.log" 2>&1
# NOTE: VBench (needs transformers==4.33.2, conflicts with diffusers' 5.10) done separately via cosmos_vbench_env.sh

echo "=== [D480 $(ts)] COMPLETE ==="
echo "--- FVD ---"; grep -A4 "=== FVD" "$EXT/d480_fvd_score.log" 2>/dev/null
echo "--- teacher vs GT ---"; grep "=== teacher" "$EXT/d480_lineval_teacher.log" 2>/dev/null | tail -1
echo "--- lin-vs-teacher ---"; grep "逼近教师" "$EXT/d480_lineval_lin.log" 2>/dev/null | tail -1
