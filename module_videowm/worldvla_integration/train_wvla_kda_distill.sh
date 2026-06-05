#!/bin/bash
# Distillation: frozen SDPA SOTA teacher -> KDA(75% Kimi)+AttnRes student.
# Student warm-starts from epoch0 (1-epoch KDA weights) with a FRESH optimizer; teacher is the
# 59% SDPA SOTA, loaded replicated+frozen (~14G/GPU). Loss = LM CE (image+action, world-model) +
# kd_weight * action-token logit-KD (targets the diagnosed action gap: l1 0.14 -> ~0.018).
# Memory: bs4 x accum4 (=eff 32, same as paper; halved micro-batch makes room for the teacher).
# Optimizer now saved FQN-based -> resume actually works; auto-restart loop survives crashes.
WV=/home/wm-vla-rl/_ext/WorldVLA/worldvla
cd "$WV/exps_256_all" || exit 1
export PYTHONPATH="/home/wm-vla-rl/_ext/LIBERO:/home/wm-vla-rl/_ext/WorldVLA:$WV:."
export HF_HOME=/home/.hf_home TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SOTA=/home/wm-vla-rl/_ext/worldvla_ckpts/model_256/libero_10   # teacher + arch source
EPOCH0=output_libero_256/kda_2gpu_bs16/epoch0                   # student warm-start (KDA model weights)
OUT=output_libero_256/kda_distill
lr=5e-6; wd=0.1; dropout=0.05; zlw=1e-5
KD_WEIGHT=${KD_WEIGHT:-5.0}; KD_TEMP=${KD_TEMP:-1.0}
mkdir -p "$OUT"

for attempt in $(seq 1 40); do
  if ls "$OUT"/epoch* >/dev/null 2>&1; then
    RESUME_ARGS=""                                              # FQN optim resume from $OUT
    echo "===== attempt $attempt: resume from $OUT @ $(date) ====="
  else
    RESUME_ARGS="--no_auto_resume --kda_resume_ckpt $EPOCH0"    # warm-start epoch0 weights, fresh optim
    echo "===== attempt $attempt: warm-start $EPOCH0 @ $(date) ====="
  fi
  /home/venv/libero/bin/torchrun --nnodes=1 --nproc_per_node=2 --master_port=30051 ../pretrain_solver_awm_w_ck.py \
    --init_from "$SOTA" $RESUME_ARGS \
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
    --distill --teacher_ckpt "$SOTA" --kd_weight $KD_WEIGHT --kd_temp $KD_TEMP --kd_action_only
  code=$?
  echo "===== attempt $attempt exited code $code @ $(date) ====="
  [ $code -eq 0 ] && echo "TRAIN_KDA_DISTILL_DONE clean" && break
  sleep 15
done
