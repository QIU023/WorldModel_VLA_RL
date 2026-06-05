#!/bin/bash
# Resume KDA(75% Kimi)+AttnRes adapt from epoch0 and keep training many more epochs.
# 1 epoch (closs 1.16) is 1/40 of the paper's schedule → expectedly undertrained. Train as long as
# feasible, tracking closs vs the SDPA-SOTA target. bs8+accum2 (=eff 32, fills GPU per tier0, proven
# in epoch0). Auto-restart loop: the epoch0->1 boundary OOM'd once (transient save/max-len spike) —
# on any crash, auto_resume picks up from the latest epoch/interval ckpt so the overnight run survives.
WV=/home/wm-vla-rl/_ext/WorldVLA/worldvla
cd "$WV/exps_256_all" || exit 1
export PYTHONPATH="/home/wm-vla-rl/_ext/LIBERO:/home/wm-vla-rl/_ext/WorldVLA:$WV:."
export HF_HOME=/home/.hf_home TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SP=/home/wm-vla-rl/_ext/worldvla_ckpts/chameleon/starting_point
lr=5e-6; wd=0.1; dropout=0.05; zlw=1e-5

for attempt in $(seq 1 40); do
  echo "===== launch attempt $attempt @ $(date) ====="
  /home/venv/libero/bin/torchrun --nnodes=1 --nproc_per_node=2 --master_port=30031 ../pretrain_solver_awm_w_ck.py \
    --init_from "$SP" \
    --disable_length_clustering --ablation 0 --model_size 7B \
    --batch_size 8 --accum_iter 2 \
    --epochs 40 --warmup_epochs 0.01 \
    --lr $lr --min_lr $lr --wd $wd --clip_grad 4 \
    --dropout $dropout --z_loss_weight $zlw \
    --data_config_train ../configs/libero_256_all/his_2_all_10_img_only_ck_10_1a2i.yaml \
    --data_config_val_ind ../configs/libero_256_all/his_2_val_ind_10_img_only_ck_10_1a2i.yaml \
    --data_config_val_ood ../configs/libero_256_all/his_2_val_ood_10_img_only_ck_10_1a2i.yaml \
    --cache_ann_on_disk --num_workers 8 \
    --output_dir output_libero_256/kda_2gpu_bs16 \
    --ckpt_max_keep 1 --only_save_trainable --save_iteration_interval 2000 \
    --max_seq_len 4096 --unmask_image_logits \
    --attn_type kda --kda_layers kimi --use_attnres --attnres_groups 8
  code=$?
  echo "===== attempt $attempt exited code $code @ $(date) ====="
  [ $code -eq 0 ] && echo "TRAIN_KDA_RESUME_DONE clean" && break
  sleep 15   # crashed (likely boundary OOM) → loop re-launches, auto_resume continues from latest ckpt
done
