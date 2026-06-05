#!/bin/bash
# Distillation v2: frozen SDPA SOTA teacher -> KDA(75% Kimi)+AttnRes student, with
#   (a) action-token logit-KD  +  (b) ARL2/MOHAWK-style per-layer hidden-state MSE on KDA layers.
# Student warm-starts from epoch0 (or the latest model-only ckpt on restart) with a fresh optimizer.
# bs4 x accum4 (=eff 32). Teacher replicated+frozen (~14G/GPU). model-only ckpts (14G) -> disk-friendly.
WV=/home/wm-vla-rl/_ext/WorldVLA/worldvla
cd "$WV/exps_256_all" || exit 1
export PYTHONPATH="/home/wm-vla-rl/_ext/LIBERO:/home/wm-vla-rl/_ext/WorldVLA:$WV:."
export HF_HOME=/home/.hf_home TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SOTA=/home/wm-vla-rl/_ext/worldvla_ckpts/model_256/libero_10
EPOCH0=output_libero_256/kda_2gpu_bs16/epoch0
OUT=output_libero_256/kda_distill_amse
lr=5e-6; wd=0.1; dropout=0.05; zlw=1e-5
KD_WEIGHT=${KD_WEIGHT:-5.0}; KD_TEMP=${KD_TEMP:-1.0}; ATTN_MSE=${ATTN_MSE:-5.0}
mkdir -p "$OUT"

for attempt in $(seq 1 60); do
  # model-only ckpts can't auto_resume (no optimizer files) -> always warm-start from the latest model
  # ckpt with a fresh optimizer (constant lr; momentum loss on rare restarts is negligible).
  LATEST=$(ls -d "$OUT"/epoch* 2>/dev/null | sort -V | tail -1)
  if [ -n "$LATEST" ] && ls "$LATEST"/model-*.safetensors >/dev/null 2>&1; then
    START="$LATEST"
  else
    START="$EPOCH0"
  fi
  echo "===== attempt $attempt: warm-start $START @ $(date) ====="
  /home/venv/libero/bin/torchrun --nnodes=1 --nproc_per_node=2 --master_port=30061 ../pretrain_solver_awm_w_ck.py \
    --init_from "$SOTA" --no_auto_resume --kda_resume_ckpt "$START" \
    --disable_length_clustering --ablation 0 --model_size 7B \
    --batch_size 4 --accum_iter 4 \
    --epochs 40 --warmup_epochs 0.01 \
    --lr $lr --min_lr $lr --wd $wd --clip_grad 4 \
    --dropout $dropout --z_loss_weight $zlw \
    --data_config_train ../configs/libero_256_all/his_2_all_10_img_only_ck_10_1a2i.yaml \
    --data_config_val_ind ../configs/libero_256_all/his_2_val_ind_10_img_only_ck_10_1a2i.yaml \
    --data_config_val_ood ../configs/libero_256_all/his_2_val_ood_10_img_only_ck_10_1a2i.yaml \
    --cache_ann_on_disk --num_workers 8 \
    --output_dir "$OUT" \
    --ckpt_max_keep 1 --only_save_trainable --save_iteration_interval 2000 \
    --max_seq_len 4096 --unmask_image_logits \
    --attn_type kda --kda_layers kimi --use_attnres --attnres_groups 8 \
    --distill --teacher_ckpt "$SOTA" --kd_weight $KD_WEIGHT --kd_temp $KD_TEMP --kd_action_only \
    --attn_mse_weight $ATTN_MSE --no_save_optimizer
  code=$?
  echo "===== attempt $attempt exited code $code @ $(date) ====="
  [ $code -eq 0 ] && echo "TRAIN_KDA_DISTILL_DONE clean" && break
  sleep 15
done
