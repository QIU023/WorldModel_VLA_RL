#!/bin/bash
# Re-score with the 12 dims that DON'T need GRiT (its weight URL returns HTTP 409, dead).
# Skipped (GRiT object-detector): object_class, multiple_objects, color, spatial_relationship.
# Kept: ALL 7 Quality dims (the ARL2 headline) + 5 Semantic. Teacher on GPU0, student on GPU1, parallel.
set -u
EXT=/home/wm-vla-rl/_ext; PY=/home/venv/cosmos_predict2/bin/python; WE=$EXT/wan_eval; VBOUT=$WE/vbench_out
# Quality 7 only — none use HF transformers (DINO/CLIP/AMT/MUSIQ/RAFT/pixel) so they work on transformers 5.10.
# (Semantic dims need tag2text/ViCLIP/UMT which import apply_chunking_to_forward, removed in transformers 5.10;
#  + GRiT dims need detectron2 + a dead URL. Quality is the ARL2 headline + the gate metric.)
DIMS="subject_consistency,background_consistency,temporal_flickering,motion_smoothness,aesthetic_quality,imaging_quality,dynamic_degree"
ts() { date +%H:%M:%S; }
echo "[rescore $(ts)] Quality 7 dims, teacher(GPU0)+student(GPU1) parallel"
CUDA_VISIBLE_DEVICES=0 $PY $WE/vbench_score.py "$WE/gen_teacher" wan_teacher "$VBOUT" "$DIMS" > "$WE/vb_teacher.log" 2>&1 &
CUDA_VISIBLE_DEVICES=1 $PY $WE/vbench_score.py "$WE/gen_student" wan_student "$VBOUT" "$DIMS" > "$WE/vb_student.log" 2>&1 &
wait
echo "[rescore $(ts)] === teacher vs student vs ARL2(75%: Q86.71/S70.24) ==="
$PY - "$VBOUT" <<'PY'
import json, sys, os
vb=sys.argv[1]
try:
    t=json.load(open(os.path.join(vb,"wan_teacher_eval_results.json")))
    s=json.load(open(os.path.join(vb,"wan_student_eval_results.json")))
except Exception as e:
    print("compare unavailable:", e); sys.exit(0)
g=lambda d,k:(d[k][0] if isinstance(d[k],list) else d[k])
keys=list(t.keys())
print(f"{'dimension':26}{'teacher':>9}{'student':>9}{'delta':>8}   (x100)")
for k in keys: print(f"{k:26}{g(t,k)*100:9.2f}{g(s,k)*100:9.2f}{(g(s,k)-g(t,k))*100:+8.2f}")
qt=sum(g(t,k) for k in keys)/len(keys)*100; qs=sum(g(s,k) for k in keys)/len(keys)*100
print(f"\n{'Quality avg (7, simple)':26}{qt:9.2f}{qs:9.2f}{qs-qt:+8.2f}   [ARL2 75%: 86.71 vs its teacher 87.09]")
print("NOTE: simple mean (not VBench's exact weighted Quality Score); GRiT(4)+transformers-Semantic(5) dims unavailable in this env.")
json.dump({"teacher":{k:g(t,k) for k in keys},"student":{k:g(s,k) for k in keys},
           "quality_avg":{"teacher":qt,"student":qs}},
          open(os.path.join(vb,"wan_vbench_compare.json"),"w"), indent=2)
print("\nsaved", os.path.join(vb,"wan_vbench_compare.json"))
PY
echo "=== [rescore $(ts)] DONE ==="
