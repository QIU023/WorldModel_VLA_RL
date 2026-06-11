#!/bin/bash
# WorldVLA KDA(75%)+AttnRes + CONTINUOUS-action-head, closed-loop LIBERO-10 eval (FRESH, render-fixed).
# Sharded across 2 GPUs: GPU0=HALF1 (tasks 0-4), GPU1=HALF2 (tasks 5-9) = all 10 tasks. CONT_NCHUNKS=1 (regen
# action every step -> true closed-loop, no open-loop chunk drift). N_TRIALS=50/task. Run from exps_256_all/.
set -u
cd /home/wm-vla-rl/_ext/WorldVLA/worldvla/exps_256_all
CK=output_libero_256/kda_75_s2conthead_MERGED
AA="--attn_type kda --kda_layers 3,5,6,8,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30 --use_attnres --attnres_groups 8 --cont_head"
ts() { date +%H:%M:%S; }
echo "[conthead-eval $(ts)] launching 2 shards (GPU0 tasks0-4, GPU1 tasks5-9), CONT_NCHUNKS=1, N_TRIALS=50"
DEVICE=0 HALF=1 PORT=29520 MODELTAG=conthead9k N_TRIALS=50 CONT_NCHUNKS=1 RESUME=$CK ATTN_ARGS="$AA" \
  bash eval_libero_kda_10.sh > shard_half1.log 2>&1 &
P0=$!
DEVICE=1 HALF=2 PORT=29521 MODELTAG=conthead9k N_TRIALS=50 CONT_NCHUNKS=1 RESUME=$CK ATTN_ARGS="$AA" \
  bash eval_libero_kda_10.sh > shard_half2.log 2>&1 &
P1=$!
echo "[conthead-eval $(ts)] shard pids: $P0 $P1 — waiting..."
wait $P0; wait $P1
echo "[conthead-eval $(ts)] === both shards done; aggregating success ==="
# each shard prints per-task success; final aggregate from the mp4 filenames (success=True/False) is robust
D0=output_libero_256/eval_conthead9k_libero10_his2imgonly_ck10_1a2i_half1
D1=output_libero_256/eval_conthead9k_libero10_his2imgonly_ck10_1a2i_half2
for D in "$D0" "$D1"; do
  tot=$(ls "$D"/*.mp4 2>/dev/null | wc -l)
  suc=$(ls "$D"/*success=True*.mp4 2>/dev/null | wc -l)
  echo "  $D : $suc / $tot success"
done
TOT=$(ls "$D0"/*.mp4 "$D1"/*.mp4 2>/dev/null | wc -l)
SUC=$(ls "$D0"/*success=True*.mp4 "$D1"/*success=True*.mp4 2>/dev/null | wc -l)
echo "[conthead-eval $(ts)] === TOTAL: $SUC / $TOT = $(awk "BEGIN{if($TOT>0)printf \"%.1f\", 100*$SUC/$TOT; else print 0}")% (teacher 59%, discrete-KDA 0%) ==="
