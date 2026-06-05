#!/bin/bash
# WorldVLA-7B KDA(24层+8full Kimi)+AttnRes,严格复现 recipe,2-GPU。
# effective batch = bs16 × accum1 × 2GPU = 32(= 论文 bs8×4GPU)。无 AC(143GB 充裕,tier0)。
WV=/home/wm-vla-rl/_ext/WorldVLA/worldvla
cd "$WV/exps_256_all" || exit 1
export PYTHONPATH="/home/wm-vla-rl/_ext/LIBERO:/home/wm-vla-rl/_ext/WorldVLA:$WV:."
export HF_HOME=/home/.hf_home TOKENIZERS_PARALLELISM=false
SP=/home/wm-vla-rl/_ext/worldvla_ckpts/model_256/libero_10
lr=5e-6; wd=0.1; dropout=0.05; zlw=1e-5
mkdir -p output_libero_256/kda_2gpu_bs16
/home/venv/libero/bin/torchrun --nnodes=1 --nproc_per_node=2 --master_port=30012 ../pretrain_solver_awm_w_ck.py \
  --init_from "$SP" --no_auto_resume \
  --disable_length_clustering --ablation 0 --model_size 7B \
  --batch_size 8 --accum_iter 2 \
  --epochs 2 --warmup_epochs 0.01 \
  --lr $lr --min_lr $lr --wd $wd --clip_grad 4 \
  --data_config_train ../configs/libero_256_all/his_2_all_10_img_only_ck_10_1a2i.yaml \
  --data_config_val_ind ../configs/libero_256_all/his_2_val_ind_10_img_only_ck_10_1a2i.yaml \
  --data_config_val_ood ../configs/libero_256_all/his_2_val_ood_10_img_only_ck_10_1a2i.yaml \
  --cache_ann_on_disk --num_workers 8 \
  --output_dir output_libero_256/kda_2gpu_bs16 \
  --max_seq_len 4096 --unmask_image_logits --dropout $dropout --z_loss_weight $zlw \
  --ckpt_max_keep 1 --only_save_trainable --save_iteration_interval 2000 --attn_type kda --kda_layers kimi --use_attnres --attnres_groups 8
echo "TRAIN_KDA_DONE exit $?"
