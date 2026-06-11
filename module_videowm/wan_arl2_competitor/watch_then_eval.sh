#!/bin/bash
# Wait for 补训 to FULLY finish (process gone + step6000 ckpt on disk), then auto-run the cont VBench eval.
# Guarantees no 2 parallel experiments (waits for training process to exit first).
set -u
CK=/home/wm-vla-rl/_ext/wan_lin_out/cont5e5/wanlin_s2_step6000.safetensors
WE=/home/wm-vla-rl/_ext/wan_eval
ts() { date +%H:%M:%S; }
echo "[watch $(ts)] waiting for 补训 to finish (ckpt + process exit)..."
# poll up to ~40 min
for i in $(seq 1 480); do
  alive=$(ps aux | grep -c "[w]an_distill")
  if [ -f "$CK" ] && [ "$alive" -eq 0 ]; then
    echo "[watch $(ts)] 补训 DONE — ckpt present, training process exited. launching cont eval."
    sleep 5
    bash "$WE/rescore_cont.sh" > "$WE/rescore_cont.log" 2>&1
    echo "[watch $(ts)] cont eval finished. tail:"; tail -n 25 "$WE/rescore_cont.log"
    exit 0
  fi
  sleep 5
done
echo "[watch $(ts)] TIMEOUT waiting for 补训 (ckpt=$( [ -f "$CK" ] && echo yes || echo no ), alive procs=$alive)"
exit 1
