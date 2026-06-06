#!/bin/bash
# CLEAN from-scratch linearization-distillation (replaces all prior contaminated/fake-distill runs).
#   - surgery-init (SOTA weights + KDA layers inserted), NO warm-start/fresh-optim hacks.
#   - REAL distillation (bug fixed: distill block now in the actively-called train_one_epoch_awm_w):
#       loss = LM CE + ARL2 self-attn-output MSE (linearization transfer) + VITA-VLA action-hidden MSE.
#   - FQN full optimizer from step 0; on any crash, auto_resume keeps Adam momentum + lr (NEVER fresh-optim).
#   - bs6 accum2 (teacher resident ~110GB, leaves headroom for FQN save spike). cosine lr. Fixed EPOCHS.
#   - DO NOT kill mid-run; let it run the full step budget.
# Parametrized: ARR_KDA (layer set), OUT (dir). 50%: layers 14-29. 75%: the ARR top-8-softmax set.
WV=/home/wm-vla-rl/_ext/WorldVLA/worldvla
cd "$WV/exps_256_all" || exit 1
export PYTHONPATH="/home/wm-vla-rl/_ext/LIBERO:/home/wm-vla-rl/_ext/WorldVLA:$WV:."
export HF_HOME=/home/.hf_home TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SOTA=/home/wm-vla-rl/_ext/worldvla_ckpts/model_256/libero_10
ARR_KDA=${ARR_KDA:-"14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29"}   # default 50%
OUT=${OUT:-output_libero_256/kda_50clean}
PORT=${PORT:-30093}
lr=${LR:-5e-6}; min_lr=${MIN_LR:-5e-7}; wd=0.1; dropout=0.05; zlw=1e-5; EPOCHS=${EPOCHS:-4}
ATTN_MSE=${ATTN_MSE:-1.0}; ACTION_HIDDEN=${ACTION_HIDDEN:-1.0}; KD_WEIGHT=${KD_WEIGHT:-0.0}
mkdir -p "$OUT"

for attempt in $(seq 1 80); do
  LATEST=$(ls -d "$OUT"/epoch* 2>/dev/null | sort -V | tail -1)
  if [ -n "$LATEST" ] && ls "$LATEST"/optimizer.full.pth >/dev/null 2>&1; then
    RESUME_ARGS="--kda_resume_ckpt $LATEST"                   # FQN auto_resume: Adam momentum + lr state survive
    echo "===== attempt $attempt: FQN auto_resume from $LATEST @ $(date) ====="
  else
    RESUME_ARGS="--no_auto_resume"                            # surgery-init from SOTA (truly from scratch)
    echo "===== attempt $attempt: surgery-init (from scratch) @ $(date) ====="
  fi
  /home/venv/libero/bin/torchrun --nnodes=1 --nproc_per_node=2 --master_port=$PORT ../pretrain_solver_awm_w_ck.py \
    --init_from "$SOTA" $RESUME_ARGS \
    --disable_length_clustering --ablation 0 --model_size 7B \
    --batch_size 6 --accum_iter 2 \
    --epochs $EPOCHS --warmup_epochs 0.05 \
    --lr $lr --min_lr $min_lr --wd $wd --clip_grad 4 \
    --dropout $dropout --z_loss_weight $zlw \
    --data_config_train ../configs/libero_256_all/his_2_all_10_img_only_ck_10_1a2i.yaml \
    --data_config_val_ind ../configs/libero_256_all/his_2_val_ind_10_img_only_ck_10_1a2i.yaml \
    --data_config_val_ood ../configs/libero_256_all/his_2_val_ood_10_img_only_ck_10_1a2i.yaml \
    --cache_ann_on_disk --num_workers 8 \
    --output_dir "$OUT" \
    --ckpt_max_keep 1 --only_save_trainable --save_iteration_interval 2000 \
    --max_seq_len 4096 --unmask_image_logits \
    --attn_type kda --kda_layers "$ARR_KDA" --use_attnres --attnres_groups 8 \
    --distill --teacher_ckpt "$SOTA" --kd_weight $KD_WEIGHT --kd_temp 2.0 \
    --attn_mse_weight $ATTN_MSE --action_hidden_weight $ACTION_HIDDEN
  code=$?
  echo "===== attempt $attempt exited code $code @ $(date) ====="
  [ $code -eq 0 ] && echo "CLEAN_DONE clean" && break
  sleep 15
done
