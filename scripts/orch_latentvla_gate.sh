#!/bin/bash
cd /home/wm-vla-rl; R=module1_vjepa2ac/results
PY=/home/venv/latentvla/bin/python
ENV="MUJOCO_GL=osmesa HF_HOME=/home/.hf_home TORCH_HOME=/home/.cache/torch PYTHONPATH=/home/wm-vla-rl/_ext/vjepa2:/home/wm-vla-rl/_ext/LIBERO"
until grep -q L1_DONE $R/latentvla_align.log 2>/dev/null; do sleep 60; done
echo "[orch] BC done; running closed-loop gate" >> $R/latentvla_gate.log
env $(echo $ENV) L2_NTASK=5 L2_EP=3 CUDA_VISIBLE_DEVICES=0 $PY scripts/exp_l2_latentvla_closedloop.py > $R/l2_full.log 2>&1
env $(echo $ENV) L2_NTASK=5 L2_EP=3 L2_BLIND=1 CUDA_VISIBLE_DEVICES=0 $PY scripts/exp_l2_latentvla_closedloop.py > $R/l2_blind.log 2>&1
echo "LATENTVLA_GATE_DONE" >> $R/latentvla_gate.log
