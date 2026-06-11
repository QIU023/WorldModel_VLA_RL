#!/bin/bash
# FINAL 75% number on the 补训 9K-total ckpt (cont5e5/wanlin_s2_step6000).
# Teacher is UNCHANGED -> reuse cached gen_teacher (944) + wan_teacher_eval_results.json (no teacher regen, halves cost).
# 1) regen 944 STUDENT from the cont ckpt (2x single-GPU prompt-sharded, batch=2 KDA Triton cap)
# 2) score student -> wan_student_cont   3) compare teacher(cached) vs student_cont vs old-student(83.48 cached) vs ARL2
set -u
EXT=/home/wm-vla-rl/_ext; CF=$EXT/Causal-Forcing; PY=/home/venv/cosmos_predict2/bin/python; WE=$EXT/wan_eval; VBOUT=$WE/vbench_out
CONFIG=$CF/configs/causal_forcing_dmd_framewise_2step.yaml; BASE=/home/causal_forcing_ckpt/framewise/causal_forcing.pt
STUDENT=$EXT/wan_lin_out/cont5e5/wanlin_s2_step6000.safetensors; GEN_S=$WE/gen_student_cont
DIMS="subject_consistency,background_consistency,temporal_flickering,motion_smoothness,aesthetic_quality,imaging_quality,dynamic_degree"
ts() { date +%H:%M:%S; }
cd "$CF"
[ -f "$STUDENT" ] || { echo "[cont $(ts)] ABORT: cont ckpt missing $STUDENT"; exit 1; }
awk 'NR%2==1' "$WE/vbench_prompts.txt" > "$WE/p_g0.txt"; awk 'NR%2==0' "$WE/vbench_prompts.txt" > "$WE/p_g1.txt"
rm -rf "$GEN_S" "$GEN_S".g0.log "$GEN_S".g1.log; mkdir -p "$GEN_S"
CM="--config_path $CONFIG --checkpoint_path $BASE --use_ema --num_output_frames 21 --vbench_naming --batch_size 2 --surgery_ckpt $STUDENT --surgery_ratio 0.75 --surgery_groups 4 --surgery_lora_r 8"
echo "[cont $(ts)] regen 944 STUDENT_cont (9K-total ckpt) — 2x single-GPU batch=2"
CUDA_VISIBLE_DEVICES=0 $PY $CF/inference.py $CM --data_path "$WE/p_g0.txt" --output_folder "$GEN_S" > "$GEN_S".g0.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 $PY $CF/inference.py $CM --data_path "$WE/p_g1.txt" --output_folder "$GEN_S" > "$GEN_S".g1.log 2>&1 &
wait
NS=$(ls "$GEN_S"/*.mp4 2>/dev/null | wc -l); echo "[cont $(ts)] student_cont videos: $NS/944"
[ "$NS" -lt 100 ] && { echo "[cont $(ts)] ABORT: student_cont gen $NS (broken)"; exit 1; }
echo "[cont $(ts)] === scoring Quality-7 (student_cont; teacher reused from cache) ==="
CUDA_VISIBLE_DEVICES=0 $PY $WE/vbench_score.py "$GEN_S" wan_student_cont "$VBOUT" "$DIMS" > "$WE/vb_student_cont.log" 2>&1
echo "[cont $(ts)] === teacher(cached) vs student_cont(9K) vs student_orig(83.48) vs ARL2(75%:Q86.71) ==="
$PY - "$VBOUT" <<'PY'
import json, sys, os
vb=sys.argv[1]; g=lambda d,k:(d[k][0] if isinstance(d[k],list) else d[k])
t=json.load(open(os.path.join(vb,"wan_teacher_eval_results.json")))
c=json.load(open(os.path.join(vb,"wan_student_cont_eval_results.json")))
try: o=json.load(open(os.path.join(vb,"wan_student_eval_results.json")))
except Exception: o=None
keys=list(t.keys())
hdr=f"{'dimension':26}{'teacher':>9}{'orig3k':>9}{'cont9k':>9}{'Δcont-T':>9}"
print(hdr)
for k in keys:
    ov=f"{g(o,k)*100:9.2f}" if o else f"{'-':>9}"
    print(f"{k:26}{g(t,k)*100:9.2f}{ov}{g(c,k)*100:9.2f}{(g(c,k)-g(t,k))*100:+9.2f}")
qt=sum(g(t,k) for k in keys)/len(keys)*100; qc=sum(g(c,k) for k in keys)/len(keys)*100
qo=sum(g(o,k) for k in keys)/len(keys)*100 if o else None
print(f"\n{'Quality avg (7)':26}{qt:9.2f}{(f'{qo:9.2f}' if o else '        -')}{qc:9.2f}{qc-qt:+9.2f}   [ARL2 75%: 86.71 vs its teacher 87.09 = -0.38]")
print("NOTE: simple 7-dim mean; reproducible OUR-student-vs-OUR-teacher is the verdict, ARL2 is a SOFT reference (unreleased code/data/lr).")
json.dump({"teacher":{k:g(t,k) for k in keys},"student_cont9k":{k:g(c,k) for k in keys},
           "quality_avg":{"teacher":qt,"student_orig3k":qo,"student_cont9k":qc}},
          open(os.path.join(vb,"wan_vbench_compare_cont.json"),"w"), indent=2)
print("saved", os.path.join(vb,"wan_vbench_compare_cont.json"))
PY
echo "=== [cont $(ts)] DONE ==="
