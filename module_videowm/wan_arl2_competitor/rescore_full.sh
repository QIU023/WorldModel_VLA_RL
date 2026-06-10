#!/bin/bash
# Full 16-dim VBench scoring once GRiT (HF mirror) + the 16 missing teacher videos are ready.
# Fixes vs the broken first attempt: (1) vbench_score.py now patches torch.load weights_only=False
# (torch 2.12 broke VBench's old AMT/RAFT ckpts); (2) GRiT from HF mirror (azure URL was 409);
# (3) teacher regenerated to 944 (was 928). Teacher GPU0 + student GPU1 parallel.
set -u
EXT=/home/wm-vla-rl/_ext; PY=/home/venv/cosmos_predict2/bin/python; WE=$EXT/wan_eval; VBOUT=$WE/vbench_out
GRIT=$HOME/.cache/vbench/grit_model/grit_b_densecap_objectdet.pth
ts() { date +%H:%M:%S; }
echo "[full $(ts)] waiting for GRiT download + teacher=944 ..."
while [ ! -s "$GRIT" ] || [ "$(ls "$WE"/gen_teacher/*.mp4 2>/dev/null | wc -l)" -lt 944 ]; do sleep 20; done
echo "[full $(ts)] ready: GRiT $(du -h "$GRIT"|cut -f1), teacher=$(ls "$WE"/gen_teacher/*.mp4|wc -l), student=$(ls "$WE"/gen_student/*.mp4|wc -l)"

echo "[full $(ts)] scoring ALL 16 dims, teacher(GPU0)+student(GPU1) parallel"
CUDA_VISIBLE_DEVICES=0 $PY $WE/vbench_score.py "$WE/gen_teacher" wan_teacher "$VBOUT" all > "$WE/vb_teacher.log" 2>&1 &
CUDA_VISIBLE_DEVICES=1 $PY $WE/vbench_score.py "$WE/gen_student" wan_student "$VBOUT" all > "$WE/vb_student.log" 2>&1 &
wait

echo "[full $(ts)] === teacher vs student vs ARL2 (75%: Q86.71 / S70.24) ==="
$PY - "$VBOUT" <<'PY'
import json, sys, os
vb=sys.argv[1]
QUAL=["subject_consistency","background_consistency","temporal_flickering","motion_smoothness","aesthetic_quality","imaging_quality","dynamic_degree"]
SEM =["object_class","multiple_objects","human_action","color","spatial_relationship","scene","appearance_style","temporal_style","overall_consistency"]
try:
    t=json.load(open(os.path.join(vb,"wan_teacher_eval_results.json")))
    s=json.load(open(os.path.join(vb,"wan_student_eval_results.json")))
except Exception as e:
    print("compare unavailable:", e); sys.exit(0)
g=lambda d,k:(d[k][0] if isinstance(d.get(k),list) else d.get(k))
def avg(d,ks): vs=[g(d,k)*100 for k in ks if g(d,k) is not None]; return sum(vs)/len(vs) if vs else float('nan')
print(f"{'dimension':26}{'teacher':>9}{'student':>9}{'delta':>8}")
for k in QUAL+SEM:
    if g(t,k) is None: continue
    print(f"{k:26}{g(t,k)*100:9.2f}{g(s,k)*100:9.2f}{(g(s,k)-g(t,k))*100:+8.2f}")
print(f"\n{'Quality avg (7, simple)':26}{avg(t,QUAL):9.2f}{avg(s,QUAL):9.2f}{avg(s,QUAL)-avg(t,QUAL):+8.2f}   [ARL2 75%: 86.71]")
print(f"{'Semantic avg (9, simple)':26}{avg(t,SEM):9.2f}{avg(s,SEM):9.2f}{avg(s,SEM)-avg(t,SEM):+8.2f}   [ARL2 75%: 70.24]")
print("\nNOTE: simple mean of dim groups (not VBench's exact weighted Quality/Semantic Score) — for ballpark vs ARL2.")
json.dump({"teacher":{k:g(t,k) for k in QUAL+SEM if g(t,k) is not None},
           "student":{k:g(s,k) for k in QUAL+SEM if g(s,k) is not None},
           "quality_avg":{"teacher":avg(t,QUAL),"student":avg(s,QUAL)},
           "semantic_avg":{"teacher":avg(t,SEM),"student":avg(s,SEM)}},
          open(os.path.join(vb,"wan_vbench_compare.json"),"w"), indent=2)
print("saved", os.path.join(vb,"wan_vbench_compare.json"))
PY
echo "=== [full $(ts)] DONE ==="
