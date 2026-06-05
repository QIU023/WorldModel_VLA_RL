#!/bin/bash
# Distillation (proper run): frozen SDPA teacher -> KDA(75% Kimi)+AttnRes student.
#   loss = LM CE + action-token logit-KD + ARL2-faithful self-attn-output MSE.
#   COSINE lr (anneals 5e-6 -> min_lr; constant lr let the action-KD oscillate/drift).
#   FULL FQN-optimizer ckpts -> real resume (Adam momentum + lr schedule survive restarts).
# Warm-starts ONCE from the best distilled model ckpt (l1~0.11); thereafter auto_resume (FQN).
WV=/home/wm-vla-rl/_ext/WorldVLA/worldvla
cd "$WV/exps_256_all" || exit 1
export PYTHONPATH="/home/wm-vla-rl/_ext/LIBERO:/home/wm-vla-rl/_ext/WorldVLA:$WV:."
export HF_HOME=/home/.hf_home TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SOTA=/home/wm-vla-rl/_ext/worldvla_ckpts/model_256/libero_10
BEST=output_libero_256/kda_2gpu_bs16/epoch0   # (unused now) clean 1-epoch KDA adapt
OUT=output_libero_256/kda_distill_arr
lr=${LR:-5e-6}; min_lr=${MIN_LR:-5e-7}; wd=0.1; dropout=0.05; zlw=1e-5
# VERIFIED-standard recipe (no self-invented loss): LM CE + ARL2 self-attn-output MSE (linearization
# transfer) + VITA-VLA action-token last-layer hidden-state MSE (action distillation). No logit-KD.
EPOCHS=${EPOCHS:-8}; KD_WEIGHT=${KD_WEIGHT:-0.0}; KD_TEMP=${KD_TEMP:-2.0}
ATTN_MSE=${ATTN_MSE:-1.0}; ACTION_HIDDEN=${ACTION_HIDDEN:-1.0}
mkdir -p "$OUT"

for attempt in $(seq 1 60); do
  LATEST=$(ls -d "$OUT"/epoch* 2>/dev/null | sort -V | tail -1)
  if [ -n "$LATEST" ] && ls "$LATEST"/optimizer.full.pth >/dev/null 2>&1; then
    # real resume: auto_resume finds LATEST (FQN optimizer + epoch/lr state); kda_resume_ckpt loads KDA model
    RESUME_ARGS="--kda_resume_ckpt $LATEST"
    echo "===== attempt $attempt: FQN auto_resume from $LATEST @ $(date) ====="
  else
    RESUME_ARGS="--no_auto_resume"   # SURGERY-INIT (ARR selection keeps action-critical layers softmax -> functional)
    echo "===== attempt $attempt: surgery-init + ARR layer selection @ $(date) ====="
  fi
  /home/venv/libero/bin/torchrun --nnodes=1 --nproc_per_node=2 --master_port=30071 ../pretrain_solver_awm_w_ck.py \
    --init_from "$SOTA" $RESUME_ARGS \
    --disable_length_clustering --ablation 0 --model_size 7B \
    --batch_size 4 --accum_iter 4 \
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
    --attn_type kda --kda_layers 14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29 --use_attnres --attnres_groups 8 \
    --distill --teacher_ckpt "$SOTA" --kd_weight $KD_WEIGHT --kd_temp $KD_TEMP \
    --attn_mse_weight $ATTN_MSE --action_hidden_weight $ACTION_HIDDEN
  code=$?
  echo "===== attempt $attempt exited code $code @ $(date) ====="
  [ $code -eq 0 ] && echo "TRAIN_KDA_DISTILL_DONE clean" && break
  sleep 15
done
