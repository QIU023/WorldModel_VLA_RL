#!/usr/bin/env bash
# P2-prep：LIBERO sim 装到独立 venv /home/venv/libero（robosuite/mujoco，老 pin，隔离）。headless EGL。
# best-effort：失败无害（P1 已 banked）。
set -uo pipefail
VENV=/home/venv/libero
export MUJOCO_GL=egl HF_HOME=/home/.hf_home
echo "[$(date -u +%H:%M:%SZ)] create venv"
python3 -m venv "$VENV"
"$VENV/bin/pip" install -q -U pip wheel
echo "[$(date -u +%H:%M:%SZ)] install LIBERO requirements (old pins)"
cd /home/wm-vla-rl/_ext/LIBERO
"$VENV/bin/pip" install -r requirements.txt 2>&1 | tail -20
echo "[$(date -u +%H:%M:%SZ)] pip install -e LIBERO"
"$VENV/bin/pip" install -e . 2>&1 | tail -5
echo "LIBERO_INSTALL_RC=$? [$(date -u +%H:%M:%SZ)]"
echo "[$(date -u +%H:%M:%SZ)] import + benchmark smoke"
"$VENV/bin/python" -c "
import os; os.environ['MUJOCO_GL']='egl'
from libero.libero import benchmark
print('benchmarks:', list(benchmark.get_benchmark_dict().keys())[:6])
print('LIBERO_IMPORT_OK')
" 2>&1 | tail -8
