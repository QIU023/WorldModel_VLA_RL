#!/bin/bash
# Stage-1: plain-CE adapt (NO distillation) of the ARR-selected KDA model from surgery-init -> functional.
# ARR layer selection: KDA only on low-action middle layers 14-29; keep action-critical 0-13,30,31 softmax.
# This is the fast functional-izer (distillation-from-broken was glacial). Then check action_0:
# if ARR (action-critical layers softmax) fixes action_0 -> may not even need Stage-2 distill.
# No teacher resident -> bs8 (faster). model-only ckpts. cosine lr.
WV=/home/wm-vla-rl/_ext/WorldVLA/worldvla
cd "$WV/exps_256_all" || exit 1
export PYTHONPATH="/home/wm-vla-rl/_ext/LIBERO:/home/wm-vla-rl/_ext/WorldVLA:$WV:."
export HF_HOME=/home/.hf_home TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SOTA=/home/wm-vla-rl/_ext/worldvla_ckpts/model_256/libero_10
OUT=output_libero_256/kda_75arr
ARR_KDA="3,5,6,8,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30"
lr=${LR:-5e-6}; min_lr=${MIN_LR:-5e-7}; wd=0.1; dropout=0.05; zlw=1e-5; EPOCHS=${EPOCHS:-4}
mkdir -p "$OUT"

for attempt in $(seq 1 40); do
  LATEST=$(ls -d "$OUT"/epoch* 2>/dev/null | sort -V | tail -1)
  if [ -n "$LATEST" ] && ls "$LATEST"/model-*.safetensors >/dev/null 2>&1; then
    RESUME_ARGS="--no_auto_resume --kda_resume_ckpt $LATEST"   # restart: warm-start latest model, fresh optim
    echo "===== attempt $attempt: warm-start $LATEST @ $(date) ====="
  else
    RESUME_ARGS="--no_auto_resume"                              # surgery-init + ARR
    echo "===== attempt $attempt: surgery-init + ARR (plain CE) @ $(date) ====="
  fi
  /home/venv/libero/bin/torchrun --nnodes=1 --nproc_per_node=2 --master_port=30081 ../pretrain_solver_awm_w_ck.py \
    --init_from "$SOTA" $RESUME_ARGS \
    --disable_length_clustering --ablation 0 --model_size 7B \
    --batch_size 8 --accum_iter 2 \
    --epochs $EPOCHS --warmup_epochs 0.05 \
    --lr $lr --min_lr $min_lr --wd $wd --clip_grad 4 \
    --dropout $dropout --z_loss_weight $zlw \
    --data_config_train ../configs/libero_256_all/his_2_all_10_img_only_ck_10_1a2i.yaml \
    --data_config_val_ind ../configs/libero_256_all/his_2_val_ind_10_img_only_ck_10_1a2i.yaml \
    --data_config_val_ood ../configs/libero_256_all/his_2_val_ood_10_img_only_ck_10_1a2i.yaml \
    --cache_ann_on_disk --num_workers 8 \
    --output_dir "$OUT" \
    --ckpt_max_keep 1 --only_save_trainable --save_iteration_interval 2000 --no_save_optimizer \
    --max_seq_len 4096 --unmask_image_logits \
    --attn_type kda --kda_layers $ARR_KDA --use_attnres --attnres_groups 8
  code=$?
  echo "===== attempt $attempt exited code $code @ $(date) ====="
  [ $code -eq 0 ] && echo "STAGE1_DONE clean" && break
  sleep 15
done
