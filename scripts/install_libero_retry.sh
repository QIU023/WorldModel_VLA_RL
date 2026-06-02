#!/usr/bin/env bash
set -uo pipefail
VENV=/home/venv/libero
export MUJOCO_GL=egl DEBIAN_FRONTEND=noninteractive
echo "[$(date -u +%H:%M:%SZ)] install cmake + EGL headers"
(apt-get install -y -qq cmake libegl1-mesa-dev libgl1-mesa-dev 2>&1 | tail -3) || echo "apt failed, trying pip cmake"
"$VENV/bin/pip" install -q cmake 2>&1 | tail -1
which cmake; cmake --version 2>&1 | head -1
echo "[$(date -u +%H:%M:%SZ)] reinstall LIBERO requirements"
cd /home/wm-vla-rl/_ext/LIBERO
"$VENV/bin/pip" install -r requirements.txt 2>&1 | tail -12
"$VENV/bin/pip" install -e . 2>&1 | tail -3
echo "RC=$?"
echo "[$(date -u +%H:%M:%SZ)] smoke"
"$VENV/bin/python" -c "
import os; os.environ['MUJOCO_GL']='egl'
from libero.libero import benchmark
print('benchmarks:', list(benchmark.get_benchmark_dict().keys())[:6]); print('LIBERO_IMPORT_OK')
" 2>&1 | tail -6
