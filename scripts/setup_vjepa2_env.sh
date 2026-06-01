#!/usr/bin/env bash
# 给 V-JEPA2 建独立 venv（不污染 /home/venv/vllm = Module-2 rollout 引擎, 也不碰 conda py3.10 = AttnRes 训练）
# ckpt/hub 权重落 /home（持久）。今晚约束: 任何 GPU 触碰只用 GPU1。
set -euo pipefail
VENV=/home/venv/vjepa2
export HF_HOME=/home/.hf_home
export TORCH_HOME=/home/.cache/torch          # hub 下的 .pt 落 /home
mkdir -p "$TORCH_HOME"

echo "[$(date -u +%H:%M:%SZ)] creating venv $VENV"
python3 -m venv "$VENV"
"$VENV/bin/pip" install -q -U pip

echo "[$(date -u +%H:%M:%SZ)] installing torch 2.11.0 (cu130, 匹配现有栈)"
"$VENV/bin/pip" install -q torch==2.11.0 --index-url https://download.pytorch.org/whl/cu130

echo "[$(date -u +%H:%M:%SZ)] installing vjepa2 hub deps + 轻量依赖"
"$VENV/bin/pip" install -q timm einops numpy beartype iopath pyyaml

echo "SETUP_DONE [$(date -u +%H:%M:%SZ)]"
