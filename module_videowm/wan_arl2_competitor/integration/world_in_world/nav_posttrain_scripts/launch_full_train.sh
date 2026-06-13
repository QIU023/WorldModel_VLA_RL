#!/bin/bash
# Full nav-LoRA training, 2x H200 DDP, background. Logs to /root/nav_posttrain/train.log.
set -e
cd /home/wm-vla-rl/_ext/Causal-Forcing
export OMP_NUM_THREADS=8
nohup /home/venv/cosmos_predict2/bin/torchrun --nproc_per_node=2 --master_port=29577 \
    /root/nav_posttrain/scripts/nav_posttrain.py --mode train \
    --latents /root/nav_posttrain/latents \
    --batch 1 --accum 4 --steps 2000 --lr 3e-4 --warmup 100 \
    --log_every 25 --save_every 500 \
    --ckpt_dir /root/nav_posttrain/ckpts \
    --home_copy /home/wm-vla-rl/_ext/wan_lin_out/nav_lora \
    > /root/nav_posttrain/train.log 2>&1 &
echo "launched full DDP train PID $!  -> /root/nav_posttrain/train.log"
