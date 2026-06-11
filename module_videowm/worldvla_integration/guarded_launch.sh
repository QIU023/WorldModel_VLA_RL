#!/bin/bash
# GPU GUARD wrapper: never run two experiments in parallel (hard user constraint). Poll every 3 min
# until BOTH GPUs are free (no jepa_preproc_decode / train / torchrun job AND <2GB used on each), then
# launch the full-seq OPEN-LOOP cont-head eval. Self-contained so it survives across turns.
set -u
cd /home/wm-vla-rl/_ext/WorldVLA/worldvla/exps_256_all
ts() { date +%Y-%m-%d_%H:%M:%S; }

free_check() {
  # busy if any contending job is running
  if pgrep -af "jepa_preproc_decode|wan_distill|train_wvla|pretrain_solver|torchrun.*train" | grep -v grep >/dev/null 2>&1; then
    return 1
  fi
  # busy if either GPU has >2GB used
  local maxmem
  maxmem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
  if [ "${maxmem:-0}" -gt 2000 ]; then
    return 1
  fi
  return 0
}

echo "[guard $(ts)] waiting for both GPUs to be free (poll 180s)..."
while ! free_check; do
  echo "[guard $(ts)] GPUs busy: $(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | tr '\n' ' | ')  -- sleeping 180s"
  sleep 180
done
echo "[guard $(ts)] GPUs FREE -> launching full-seq OPEN-LOOP eval (CONT_NCHUNKS=25, CONT_CACHED=0, N_TRIALS=${N_TRIALS:-50})"
N_TRIALS=${N_TRIALS:-50} MODELTAG=${MODELTAG:-contOL25} CONT_CACHED=0 bash eval_conthead_openloop_sharded.sh
echo "[guard $(ts)] eval launcher returned."
