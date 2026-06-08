#!/bin/bash
# VBench@480p in an isolated transformers version (vbench needs 4.33.2; diffusers/Cosmos needs 5.10).
# cosmos_vbench.py only reads the generated mp4s (no diffusers), so we swap transformers, score, restore.
# Usage: nohup bash cosmos_vbench_env.sh > cosmos_vbench_env.log 2>&1 &
set -u
EXT=/home/wm-vla-rl/_ext; PY=/home/venv/cosmos_predict2/bin/python; PIP=/home/venv/cosmos_predict2/bin/pip
FVDO=$EXT/cosmos_fvd480_out
DIMS=subject_consistency,background_consistency,motion_smoothness,dynamic_degree,aesthetic_quality,imaging_quality
cd "$EXT"; ts() { date +%H:%M:%S; }
restore() { echo "[VB480 $(ts)] restore transformers 5.10"; $PIP install -q transformers==5.10.0 2>&1 | tail -1; }
trap restore EXIT   # always restore Cosmos's transformers, even on failure

echo "[VB480 $(ts)] swap transformers -> 4.33.2 (vbench)"; $PIP install -q transformers==4.33.2 2>&1 | tail -1
echo "[VB480 $(ts)] VBench teacher"; CUDA_VISIBLE_DEVICES=0 $PY cosmos_vbench.py --vids "$FVDO/vids_teacher.pt" --name d480_teacher --dims "$DIMS" > "$EXT/vb480_teacher.log" 2>&1
echo "[VB480 $(ts)] VBench lin";     CUDA_VISIBLE_DEVICES=0 $PY cosmos_vbench.py --vids "$FVDO/vids_r75_lin.pt" --name d480_lin --dims "$DIMS" > "$EXT/vb480_lin.log" 2>&1

echo "=== [VB480 $(ts)] VBench@480p teacher vs lin ==="
$PY - <<'PY'
import json
t=json.load(open("vbench_out/scores_d480_teacher.json")); l=json.load(open("vbench_out/scores_d480_lin.json"))
print(f"{'dimension':28}{'teacher':>9}{'lin-75%':>9}{'delta':>8}")
for k in t: print(f"{k:28}{t[k]:9.4f}{l.get(k,0):9.4f}{l.get(k,0)-t[k]:+8.4f}")
json.dump({"teacher":t,"lin":l},open("vbench_out/vbench480_compare.json","w"),indent=2)
PY
