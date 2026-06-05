#!/bin/bash
# KDA(75% Kimi)+AttnRes adapt eval on LIBERO-Long (libero_10), 256px, his=2 img_only, ck_10.
# Mirrors the original eval_libero_7B_..._10.sh but: local paths, single GPU, KDA surgery flags.
set -e
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LIBGL_DRIVERS_PATH=/usr/lib/x86_64-linux-gnu/dri/
# CPU software rendering (osmesa): EGL GPU contention SIGABRTs with 2 concurrent procs; osmesa is stable.
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export PYTHONPATH=/home/wm-vla-rl/_ext/LIBERO:/home/wm-vla-rl/_ext/WorldVLA/worldvla:/home/wm-vla-rl/_ext/WorldVLA/worldvla/libero_util:${PYTHONPATH}

CKPT_ROOT=output_libero_256/kda_2gpu_bs16
# pick the newest full-epoch ckpt (epochN), fall back to newest interval ckpt (epochN-iterM)
RESUME=${RESUME:-$(ls -d ${CKPT_ROOT}/epoch* 2>/dev/null | grep -vE 'iter' | sort | tail -1)}
RESUME=${RESUME:-$(ls -d ${CKPT_ROOT}/epoch*-iter* 2>/dev/null | sort -t r -k3 -n | tail -1)}
echo "[eval] RESUME=$RESUME"

lr=5e-6; wd=0.1; dropout=0.05; z_loss_weight=1e-5
DEVICE=${DEVICE:-0}

data_config_train=../configs/libero_256_all/his_2_all_10_img_only_ck_10_1a2i.yaml
data_config_val_ind=../configs/libero_256_all/his_2_val_ind_10_img_only_ck_10_1a2i.yaml
data_config_val_ood=../configs/libero_256_all/his_2_val_ood_10_img_only_ck_10_1a2i.yaml

HALF=${HALF:-0}
N_TRIALS=${N_TRIALS:-50}
PORT=${PORT:-29507}
base_exp_name=eval_${MODELTAG:-kda}_libero10_his2imgonly_ck10_1a2i_half${HALF}
base_output_dir=output_libero_256/"$base_exp_name"
mkdir -p "$base_output_dir"

/home/venv/libero/bin/torchrun --nnodes=1 --nproc_per_node=1 --master_port=${PORT} ../eval_solver_libero_256.py \
    --device ${DEVICE} \
    --task_suite_name libero_10 \
    --his 2h_1a_img_only \
    --no_auto_resume \
    --resume_path "$RESUME" \
    --eval_only True \
    --model_size 7B \
    --half ${HALF} \
    --n_trials ${N_TRIALS} \
    --batch_size 4 \
    --accum_iter 1 \
    --epochs 10 \
    --warmup_epochs 0.01 \
    --lr ${lr} --min_lr ${lr} --wd ${wd} --clip_grad 4 \
    --data_config_train $data_config_train \
    --data_config_val_ind $data_config_val_ind \
    --data_config_val_ood $data_config_val_ood \
    --cache_ann_on_disk \
    --num_workers 8 \
    --output_dir "$base_output_dir" \
    --checkpointing \
    --max_seq_len 4096 \
    --unmask_image_logits \
    --dropout ${dropout} \
    --z_loss_weight ${z_loss_weight} \
    --ckpt_max_keep 0 \
    ${ATTN_ARGS:---attn_type kda --kda_layers kimi --use_attnres --attnres_groups 8} \
    2>&1 | tee "$base_output_dir"/output.log
