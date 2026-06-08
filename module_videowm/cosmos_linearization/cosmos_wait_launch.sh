#!/bin/bash
# Wait for the LIBERO closed-loop eval to finish (free both GPUs), then launch the cosmos overnight chain.
cd /home/wm-vla-rl/_ext
echo "[wait] $(date) waiting for eval_solver_libero shards to finish..."
while pgrep -f eval_solver_libero >/dev/null 2>&1; do sleep 60; done
echo "[wait] $(date) eval done; settling 30s then launching cosmos chain on GPU0"
sleep 30
CUDA_VISIBLE_DEVICES=0 bash cosmos_overnight.sh
echo "[wait] $(date) cosmos chain returned"
