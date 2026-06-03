#!/bin/bash
cd /home/wm-vla-rl; PY=/home/venv/libero/bin/python; R=module1_vjepa2ac/results
HH=${HH:-128}; ST=${ST:-5}
run(){ M2J_WM=$1 M2J_KDA=$2 M2J_AR=$3 M2J_NAME=$4 M2J_H=$HH M2J_STEPS=$ST CUDA_VISIBLE_DEVICES=0 $PY scripts/exp_m2j_ultralong_imagine.py >$R/m2j_$4.log 2>&1; }
run sdpa_sdpaH10_sftfull.pt 0 0 sdpa
run kda_kdaonlyH10_sftfull.pt 1 0 kda
run kda_arlayerH10_sftfull.pt 1 1 kda_attnres
echo "M2J_COMPARE_ALL_DONE" >> $R/m2j_orch.log
