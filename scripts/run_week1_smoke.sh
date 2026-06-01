#!/usr/bin/env bash
# Week-1 V-JEPA2-AC smoke。今晚硬约束: 只碰 GPU1。
set -uo pipefail
export CUDA_VISIBLE_DEVICES=1          # ← 焊死, 不碰 GPU0
export TORCH_HOME=/home/.cache/torch
export HF_HOME=/home/.hf_home
VENV=/home/venv/vjepa2
RES=/home/wm-vla-rl/module1_vjepa2ac/results
LOG=$RES/smoke_vjepa2ac.log
SENT=$RES/SMOKE_VJEPA2AC_DONE
mkdir -p "$RES"; rm -f "$SENT"
echo "[$(date -u +%H:%M:%SZ)] week1 smoke start (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)" | tee "$LOG"
"$VENV/bin/python" /home/wm-vla-rl/scripts/smoke_vjepa2ac.py 2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}
grep -q SMOKE_OK "$LOG" && echo "SMOKE_VJEPA2AC_DONE rc=$rc $(date -u +%H:%M:%SZ)" > "$SENT"
echo "[$(date -u +%H:%M:%SZ)] smoke exit rc=$rc" | tee -a "$LOG"
