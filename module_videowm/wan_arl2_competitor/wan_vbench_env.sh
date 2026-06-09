#!/bin/bash
# VBench scoring for the ARL2-competitor, in an isolated transformers version (vbench needs 4.33.2; diffusers/Wan
# need 5.10). wan_generate.py already produced the mp4-able {videos} tensors; cosmos_vbench.py only reads them, so
# we swap transformers, score teacher vs student, then restore. Usage: bash wan_vbench_env.sh
set -u
EXT=/home/wm-vla-rl/_ext; PY=/home/venv/cosmos_predict2/bin/python; PIP=/home/venv/cosmos_predict2/bin/pip
GEN=/home/wan_eval_gen
DIMS=subject_consistency,background_consistency,motion_smoothness,dynamic_degree,aesthetic_quality,imaging_quality
cd "$EXT"; ts() { date +%H:%M:%S; }
restore() { echo "[VB $(ts)] restore transformers 5.10"; $PIP install -q transformers==5.10.0 2>&1 | tail -1; }
trap restore EXIT   # always restore Wan's transformers, even on failure

for s in teacher student; do
  [ -f "$GEN/vids_$s.pt" ] || { echo "[VB] missing $GEN/vids_$s.pt — run wan_generate first"; exit 1; }
done
echo "[VB $(ts)] swap transformers -> 4.33.2 (vbench)"; $PIP install -q transformers==4.33.2 2>&1 | tail -1
echo "[VB $(ts)] VBench teacher"; CUDA_VISIBLE_DEVICES=0 $PY cosmos_vbench.py --vids "$GEN/vids_teacher.pt" --name wan_teacher --dims "$DIMS" > "$EXT/vb_teacher.log" 2>&1
echo "[VB $(ts)] VBench student"; CUDA_VISIBLE_DEVICES=0 $PY cosmos_vbench.py --vids "$GEN/vids_student.pt" --name wan_student --dims "$DIMS" > "$EXT/vb_student.log" 2>&1

echo "=== [VB $(ts)] VBench teacher vs student (KDA+AttnRes 75%) ==="
$PY - <<'PY'
import json
t=json.load(open("vbench_out/wan_teacher_eval_results.json")); l=json.load(open("vbench_out/wan_student_eval_results.json"))
def g(d,k): v=d.get(k); return v[0] if isinstance(v,list) else v
print(f"{'dimension':26}{'teacher':>9}{'student':>9}{'delta':>8}")
keys=[k for k in t]
for k in keys: print(f"{k:26}{g(t,k):9.4f}{g(l,k):9.4f}{g(l,k)-g(t,k):+8.4f}")
json.dump({"teacher":{k:g(t,k) for k in keys},"student":{k:g(l,k) for k in keys}}, open("vbench_out/wan_vbench_compare.json","w"), indent=2)
PY
