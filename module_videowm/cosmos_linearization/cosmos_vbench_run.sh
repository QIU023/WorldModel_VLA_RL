#!/bin/bash
# Full VBench (SANA headline metric) on the 96 teacher + 96 lin-r0.75 videos. Sequential (teacher first
# downloads all dimension models, lin reuses cache -> no download race). GPU0.
# Usage: nohup bash cosmos_vbench_run.sh > cosmos_vbench_run.log 2>&1 &
set -u
EXT=/home/wm-vla-rl/_ext; PY=/home/venv/cosmos_predict2/bin/python
DIMS=${DIMS:-subject_consistency,background_consistency,motion_smoothness,dynamic_degree,aesthetic_quality,imaging_quality}
cd "$EXT"; ts() { date +%H:%M:%S; }
echo "[VB $(ts)] VBench dims=$DIMS"
echo "[VB $(ts)] teacher (downloads dimension models)..."
CUDA_VISIBLE_DEVICES=0 $PY cosmos_vbench.py --vids cosmos_fvd_out/vids_teacher.pt --name teacher --dims "$DIMS" > "$EXT/vb_teacher.log" 2>&1
echo "[VB $(ts)] lin-r0.75 (cached models)..."
CUDA_VISIBLE_DEVICES=0 $PY cosmos_vbench.py --vids cosmos_fvd_out/vids_r75_lin.pt --name r75_lin --dims "$DIMS" > "$EXT/vb_lin.log" 2>&1
echo "=== [VB $(ts)] DONE — teacher vs lin VBench ==="
$PY - <<'PY'
import json
t=json.load(open("vbench_out/scores_teacher.json")); l=json.load(open("vbench_out/scores_r75_lin.json"))
print(f"{'dimension':28} {'teacher':>9} {'lin-r0.75':>10} {'Δ':>8}")
for k in t:
    d=l.get(k,float('nan'))-t[k]
    print(f"{k:28} {t[k]:9.4f} {l.get(k,float('nan')):10.4f} {d:+8.4f}")
json.dump({"teacher":t,"lin_r75":l},open("vbench_out/vbench_compare.json","w"),indent=2)
PY
