#!/usr/bin/env bash
set -uo pipefail
VENV=/home/venv/libero
export MUJOCO_GL=egl DEBIAN_FRONTEND=noninteractive
echo "[$(date -u +%H:%M:%SZ)] apt-get update + system cmake + EGL headers"
apt-get update -qq 2>&1 | tail -2
apt-get install -y -qq cmake libegl1-mesa-dev libgl1-mesa-dev libosmesa6-dev 2>&1 | tail -3
echo "cmake: $(which cmake) $(cmake --version 2>&1 | head -1)"
echo "[$(date -u +%H:%M:%SZ)] reinstall LIBERO"
cd /home/wm-vla-rl/_ext/LIBERO
"$VENV/bin/pip" install -r requirements.txt 2>&1 | tail -10
"$VENV/bin/pip" install -e . 2>&1 | tail -3
echo "[$(date -u +%H:%M:%SZ)] smoke"
"$VENV/bin/python" -c "
import os; os.environ['MUJOCO_GL']='egl'
from libero.libero import benchmark
print('benchmarks:', list(benchmark.get_benchmark_dict().keys())[:6]); print('LIBERO_IMPORT_OK')
" 2>&1 | tail -8
