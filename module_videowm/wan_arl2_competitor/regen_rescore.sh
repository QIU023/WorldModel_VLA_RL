#!/bin/bash
# Regenerate the 944 STUDENT videos with the KDA streaming-state fix (teacher unaffected -> kept),
# then re-score Quality 7. Student batch=2 (KDA chunk_kda Triton grid cap), 2x single-GPU prompt-sharded.
set -u
EXT=/home/wm-vla-rl/_ext; CF=$EXT/Causal-Forcing; PY=/home/venv/cosmos_predict2/bin/python; WE=$EXT/wan_eval
CONFIG=$CF/configs/causal_forcing_dmd_framewise_2step.yaml; BASE=/home/causal_forcing_ckpt/framewise/causal_forcing.pt
STUDENT=$EXT/wan_lin_out/wanlin_s2_step3000.safetensors; GEN_S=$WE/gen_student
ts() { date +%H:%M:%S; }
cd "$CF"
awk 'NR%2==1' "$WE/vbench_prompts.txt" > "$WE/p_g0.txt"; awk 'NR%2==0' "$WE/vbench_prompts.txt" > "$WE/p_g1.txt"
rm -rf "$GEN_S" "$GEN_S".g0.log "$GEN_S".g1.log; mkdir -p "$GEN_S"
CM="--config_path $CONFIG --checkpoint_path $BASE --use_ema --num_output_frames 21 --vbench_naming --batch_size 2 --surgery_ckpt $STUDENT --surgery_ratio 0.75 --surgery_groups 4 --surgery_lora_r 8"
echo "[regen $(ts)] regenerating 944 STUDENT with KDA streaming-state fix (2x single-GPU, batch=2)"
CUDA_VISIBLE_DEVICES=0 $PY $CF/inference.py $CM --data_path "$WE/p_g0.txt" --output_folder "$GEN_S" > "$GEN_S".g0.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 $PY $CF/inference.py $CM --data_path "$WE/p_g1.txt" --output_folder "$GEN_S" > "$GEN_S".g1.log 2>&1 &
wait
NS=$(ls "$GEN_S"/*.mp4 2>/dev/null | wc -l); echo "[regen $(ts)] student videos: $NS/944"
if [ "$NS" -lt 100 ]; then echo "[regen $(ts)] ABORT: student gen $NS (broken)"; exit 1; fi
echo "[regen $(ts)] === re-scoring Quality 7 (fixed student vs kept teacher) ==="
bash "$WE/rescore.sh"
echo "=== [regen $(ts)] DONE ==="
