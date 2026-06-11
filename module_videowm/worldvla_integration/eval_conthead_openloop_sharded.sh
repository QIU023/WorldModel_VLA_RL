#!/bin/bash
# WorldVLA KDA(75%)+AttnRes + CONTINUOUS-action-head, LIBERO-10 eval, OPEN-LOOP chunked (CONT_NCHUNKS=25):
# predict a 25-action chunk from ONE observation and execute it open-loop = the FAIR protocol that matches
# the discrete head's generate_dis_ma action-chunking (and the teacher's 59%). This cuts decode CALLS 25x
# vs the old closed-loop NCHUNKS=1 (~0.33s/step every step) — the real speedup. The eval is RENDERING-bound
# (osmesa CPU, minutes/episode), so the extra KV-cache path (CONT_CACHED=1, only ~1.6x decode AND ~18%
# open-loop trajectory drift) buys ~nothing here and is left OFF for the deterministic reference number.
# Sharded across 2 GPUs: GPU0 = tasks 0-4 (HALF1), GPU1 = tasks 5-9 (HALF2). N_TRIALS default 50/task.
# Run from exps_256_all/:   bash eval_conthead_openloop_sharded.sh
set -u
cd /home/wm-vla-rl/_ext/WorldVLA/worldvla/exps_256_all
CK=output_libero_256/kda_75_s2conthead_MERGED
AA="--attn_type kda --kda_layers 3,5,6,8,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30 --use_attnres --attnres_groups 8 --cont_head"
NT=${N_TRIALS:-50}
TAG=${MODELTAG:-contOL25}
CONT_CACHED=${CONT_CACHED:-0}     # 0 = full-seq open-loop (reference, default); 1 = KV-cached (fast but ~18% drift)
ts() { date +%H:%M:%S; }
echo "[contOL-eval $(ts)] launching 2 shards (GPU0 tasks0-4, GPU1 tasks5-9), CONT_NCHUNKS=25 (open-loop), CONT_CACHED=$CONT_CACHED, N_TRIALS=$NT"
DEVICE=0 HALF=1 PORT=29530 MODELTAG=$TAG N_TRIALS=$NT CONT_NCHUNKS=25 CONT_CACHED=$CONT_CACHED RESUME=$CK ATTN_ARGS="$AA" \
  bash eval_libero_kda_10.sh > shard_OL_half1.log 2>&1 &
P0=$!
DEVICE=1 HALF=2 PORT=29531 MODELTAG=$TAG N_TRIALS=$NT CONT_NCHUNKS=25 CONT_CACHED=$CONT_CACHED RESUME=$CK ATTN_ARGS="$AA" \
  bash eval_libero_kda_10.sh > shard_OL_half2.log 2>&1 &
P1=$!
echo "[contOL-eval $(ts)] shard pids: $P0 $P1 — waiting..."
wait $P0; wait $P1
echo "[contOL-eval $(ts)] === both shards done; aggregating success from mp4 filenames ==="
D0=output_libero_256/eval_${TAG}_libero10_his2imgonly_ck10_1a2i_half1
D1=output_libero_256/eval_${TAG}_libero10_his2imgonly_ck10_1a2i_half2
for D in "$D0" "$D1"; do
  tot=$(ls "$D"/*.mp4 2>/dev/null | wc -l)
  suc=$(ls "$D"/*success=True*.mp4 2>/dev/null | wc -l)
  echo "  $D : $suc / $tot success"
done
TOT=$(ls "$D0"/*.mp4 "$D1"/*.mp4 2>/dev/null | wc -l)
SUC=$(ls "$D0"/*success=True*.mp4 "$D1"/*success=True*.mp4 2>/dev/null | wc -l)
echo "[contOL-eval $(ts)] === TOTAL: $SUC / $TOT = $(awk "BEGIN{if($TOT>0)printf \"%.1f\", 100*$SUC/$TOT; else print 0}")% (teacher 59%, discrete-KDA 0%) ==="
