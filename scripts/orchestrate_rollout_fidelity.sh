#!/bin/bash
# overnight: 等 kda+sdpa 重训完 → 训 arlayer(matched) → 并行跑 exp_e16(rollout)+exp_e15(1-step复现)
cd /home/wm-vla-rl
PY=/home/venv/libero/bin/python
R=module1_vjepa2ac/results
log(){ echo "[orch $(date -u +%H:%M:%S)] $*"; }

log "waiting kda+sdpa re-train..."
until grep -q SFT_FULL_DONE $R/sft2_kda.log 2>/dev/null && grep -q SFT_FULL_DONE $R/sft2_sdpa.log 2>/dev/null; do sleep 60; done
log "kda+sdpa done. launching arlayer(matched 3000-iter) on GPU0"
SFT_ACT_CKPT=0 SFT_BATCH=8 FPC=8 SFT_ITERS=3000 TORCH_HOME=/home/.cache/torch \
  FT_MODE=kda USE_ATTNRES=1 AR_NUM_BLOCKS=8 FT_TAG=_arlayer2 CUDA_VISIBLE_DEVICES=0 \
  $PY scripts/exp_e14_sft_full.py </dev/null > $R/sft2_arlayer.log 2>&1

log "arlayer done. running rollout-fidelity(GPU0)+1-step reconfirm(GPU1)"
CUDA_VISIBLE_DEVICES=0 $PY scripts/exp_e16_rollout_fidelity.py </dev/null > $R/e16_rollout.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 $PY scripts/exp_e15_eval_held_clean.py </dev/null > $R/e15_reconfirm.log 2>&1 &
wait
log "ALL_DONE rollout-fidelity + reconfirm"
echo "ORCH_ALL_DONE" >> $R/orch.log
