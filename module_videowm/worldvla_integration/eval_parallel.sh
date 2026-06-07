#!/bin/bash
# Parallel task-sharded closed-loop LIBERO-Long eval. Closed-loop eval is CPU-render(osmesa)-bound +
# single-env/proc -> each proc only uses ~17G/143G GPU, so we fan out N shards across 2 GPUs.
# libero_10 = 10 tasks; N_SHARDS=10 -> 1 task/proc, 5 procs/GPU (~85G/card). ~5x faster than 2-way half.
# Usage: RESUME=<merged_ckpt_dir> ARR=<kda_layer_csv> N_SHARDS=10 N_TRIALS=10 TAG=foo bash eval_parallel.sh
set -e
WV=/home/wm-vla-rl/_ext/WorldVLA/worldvla
cd "$WV/exps_256_all"
export PYTHONPATH=/home/wm-vla-rl/_ext/LIBERO:$WV:$WV/libero_util:${PYTHONPATH}
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
export LIBGL_DRIVERS_PATH=/usr/lib/x86_64-linux-gnu/dri/
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

RESUME=${RESUME:?set RESUME=merged ckpt dir}
ARR=${ARR:-"3,5,6,8,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30"}
N_SHARDS=${N_SHARDS:-10}
N_TRIALS=${N_TRIALS:-10}
TAG=${TAG:-par}
NGPU=${NGPU:-2}
LOGDIR=${LOGDIR:-/home/wm-vla-rl/_ext/eval_${TAG}_logs}
mkdir -p "$LOGDIR"
CONT_HEAD=${CONT_HEAD:-0}; CONT_ARG=""; [ "$CONT_HEAD" = "1" ] && CONT_ARG="--cont_head"   # continuous action head decode
echo "[eval_parallel] RESUME=$RESUME N_SHARDS=$N_SHARDS N_TRIALS=$N_TRIALS NGPU=$NGPU CONT_HEAD=$CONT_HEAD -> $LOGDIR"

for s in $(seq 0 $((N_SHARDS-1))); do
  dev=$((s % NGPU))
  port=$((29600 + s))
  out=output_libero_256/eval_${TAG}_shard${s}
  mkdir -p "$out"
  CUDA_VISIBLE_DEVICES=$dev /home/venv/libero/bin/torchrun --nnodes=1 --nproc_per_node=1 --master_port=$port \
    ../eval_solver_libero_256.py \
    --device 0 --task_suite_name libero_10 --his 2h_1a_img_only \
    --no_auto_resume --resume_path "$RESUME" --eval_only True --model_size 7B \
    --n_shards $N_SHARDS --shard_idx $s --n_trials $N_TRIALS \
    --batch_size 4 --accum_iter 1 --epochs 10 --warmup_epochs 0.01 \
    --lr 5e-6 --min_lr 5e-6 --wd 0.1 --clip_grad 4 \
    --data_config_train ../configs/libero_256_all/his_2_all_10_img_only_ck_10_1a2i.yaml \
    --data_config_val_ind ../configs/libero_256_all/his_2_val_ind_10_img_only_ck_10_1a2i.yaml \
    --data_config_val_ood ../configs/libero_256_all/his_2_val_ood_10_img_only_ck_10_1a2i.yaml \
    --cache_ann_on_disk --num_workers 2 --output_dir "$out" --checkpointing \
    --max_seq_len 4096 --unmask_image_logits --dropout 0.05 --z_loss_weight 1e-5 --ckpt_max_keep 0 \
    --attn_type kda --kda_layers "$ARR" --use_attnres --attnres_groups 8 \
    > "$LOGDIR/shard${s}.log" 2>&1 &
  echo "  shard $s -> GPU$dev pid $! (port $port)"
  sleep 3
done
echo "[eval_parallel] launched $N_SHARDS shards. tail $LOGDIR/shard*.log"
wait
echo "[eval_parallel] ALL SHARDS DONE"
# aggregate
python - <<PY
import glob,re
tot=succ=0
for f in sorted(glob.glob("$LOGDIR/shard*.log")):
    txt=open(f,errors='ignore').read().replace('\r','\n')
    e=re.findall(r'# episodes completed so far: (\d+)', txt)
    s=re.findall(r'# successes: (\d+)', txt)
    if e and s: tot+=int(e[-1]); succ+=int(s[-1]); print(f, "ep",e[-1],"succ",s[-1])
print(f"=== TOTAL: {succ}/{tot} = {100*succ/max(tot,1):.1f}% ===")
PY
