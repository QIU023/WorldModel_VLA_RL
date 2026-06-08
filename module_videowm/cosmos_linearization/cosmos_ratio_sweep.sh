#!/bin/bash
# Extend the quality-vs-linearization curve: full S1 attn-transfer -> S2d output-distill -> lineval-vs-teacher
# for two extra ratios in parallel (one per GPU). Reuses the existing teacher baseline (cosmos_gen_out/teacher).
# Usage: RA=0.875 RB=0.25 nohup bash cosmos_ratio_sweep.sh > cosmos_ratio_sweep.log 2>&1 &
set -u
EXT=/home/wm-vla-rl/_ext; PY=/home/venv/cosmos_predict2/bin/python
SNAP=$(ls -d /home/cosmos_predict2_model/models--nvidia--Cosmos-Predict2-2B-Video2World/snapshots/*/ | head -1)
TF="${SNAP}transformer"; DATA=/home/cosmos_lin_data; OUT=$EXT/cosmos_lin_out; GEN=$EXT/cosmos_gen_out
RA=${RA:-0.875}; RB=${RB:-0.25}
S1_STEPS=${S1_STEPS:-1500}; S2_STEPS=${S2_STEPS:-2000}; S1_BATCH=${S1_BATCH:-8}; S2_BATCH=${S2_BATCH:-8}
GEN_STEPS=30; NF=29; SZ=256; N_EVAL=24
cd "$EXT"; ts() { date +%H:%M:%S; }

run_ratio() { # $1=gpu $2=ratio
  local g=$1 r=$2
  local d="$OUT/r${r}"
  mkdir -p "$d"
  echo "[RSW $(ts)] r$r S1 (GPU$g)"
  CUDA_VISIBLE_DEVICES=$g $PY cosmos_distill.py --stage 1 --transformer "$TF" --latents "$DATA" --ratio $r \
    --steps $S1_STEPS --batch $S1_BATCH --lr 1e-3 --save "$d" > "$EXT/rsw_s1_r$r.log" 2>&1
  local S1=$(ls -t "$d"/linstate_s1_step*.safetensors 2>/dev/null | head -1)
  echo "[RSW $(ts)] r$r S2d (GPU$g) resume=$S1"
  CUDA_VISIBLE_DEVICES=$g $PY cosmos_distill.py --stage 2 --method distill --transformer "$TF" --latents "$DATA" \
    --ratio $r --resume "$S1" --steps $S2_STEPS --batch $S2_BATCH --lr 2e-4 --save "$d" > "$EXT/rsw_s2d_r$r.log" 2>&1
  local S2=$(ls -t "$d"/linstate_s2d_step*.safetensors 2>/dev/null | head -1)
  echo "[RSW $(ts)] r$r lineval (GPU$g) resume=$S2"
  CUDA_VISIBLE_DEVICES=$g $PY cosmos_generate.py --mode lineval --model "$SNAP" --teacher_dir "$GEN/teacher" \
    --resume "$S2" --ratio $r --tag "r${r}_s2d" --out "$GEN" --n $N_EVAL --num_frames $NF --steps $GEN_STEPS \
    --size $SZ > "$EXT/rsw_eval_r$r.log" 2>&1
  echo "[RSW $(ts)] r$r DONE"
}

echo "[RSW $(ts)] ratio sweep RA=$RA (GPU0) ∥ RB=$RB (GPU1)"
run_ratio 0 $RA & A=$!
run_ratio 1 $RB & B=$!
wait $A; wait $B
echo "=== [RSW $(ts)] COMPLETE ==="
for r in $RA $RB; do echo "--- r$r ---"; grep "逼近教师" "$EXT/rsw_eval_r$r.log" 2>/dev/null | tail -1; done
