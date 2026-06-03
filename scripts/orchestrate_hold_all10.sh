#!/bin/bash
# 更严格 held-out：train=libero_90(仅90任务)，held=全 10 个 libero_10 长任务（完全 unseen）
# 三件套 matched 3000-iter → exp_e16 rollout + exp_e15 1-step（EVAL_HOLD_ALL10=1, EVAL_SUF=H10）
cd /home/wm-vla-rl
PY=/home/venv/libero/bin/python
R=module1_vjepa2ac/results
C="SFT_ACT_CKPT=0 SFT_BATCH=8 FPC=8 SFT_ITERS=3000 TORCH_HOME=/home/.cache/torch SFT_HOLD_ALL10=1"
log(){ echo "[orchH10 $(date -u +%H:%M:%S)] $*"; }

log "waiting kda+sdpa (hold-all10)..."
until grep -q SFT_FULL_DONE $R/h10_kda.log 2>/dev/null && grep -q SFT_FULL_DONE $R/h10_sdpa.log 2>/dev/null; do sleep 60; done
log "kda+sdpa done. launching arlayer on GPU0"
env $(echo $C) FT_MODE=kda USE_ATTNRES=1 AR_NUM_BLOCKS=8 FT_TAG=_arlayerH10 CUDA_VISIBLE_DEVICES=0 \
  $PY scripts/exp_e14_sft_full.py </dev/null > $R/h10_arlayer.log 2>&1

log "arlayer done. running rollout-fidelity(GPU0)+1-step(GPU1), all-10 held"
EVAL_HOLD_ALL10=1 EVAL_SUF=H10 CUDA_VISIBLE_DEVICES=0 $PY scripts/exp_e16_rollout_fidelity.py </dev/null > $R/h10_e16.log 2>&1 &
EVAL_HOLD_ALL10=1 EVAL_SUF=H10 CUDA_VISIBLE_DEVICES=1 $PY scripts/exp_e15_eval_held_clean.py </dev/null > $R/h10_e15.log 2>&1 &
wait
log "ALL_DONE hold-all10"
echo "ORCH_H10_DONE" >> $R/orch_h10.log
