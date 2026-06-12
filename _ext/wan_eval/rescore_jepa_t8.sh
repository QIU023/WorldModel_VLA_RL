#!/bin/bash
# §3a RUN-2 FINAL: VBench Quality-7 on the JEPA-T8 Stage-2 ckpt (jepa_l025_T8/step6000). Reuse cached teacher/baselines.
# regen 944 student from the T8 ckpt -> score -> compare teacher(88.09) / MSE-only cont9k(83.84) / JEPA-T2(83.90) / JEPA-T8.
# KEY QUESTION: did dynamic_degree RECOVER (not drop like T2's -5.55) while keeping the spatial gains?
set -u
EXT=/home/wm-vla-rl/_ext; CF=$EXT/Causal-Forcing; PY=/home/venv/cosmos_predict2/bin/python; WE=$EXT/wan_eval; VBOUT=$WE/vbench_out
CONFIG=$CF/configs/causal_forcing_dmd_framewise_2step.yaml; BASE=/home/causal_forcing_ckpt/framewise/causal_forcing.pt
STUDENT=$EXT/wan_lin_out/jepa_l025_T8/wanlin_s2_step6000.safetensors; GEN_S=$WE/gen_student_jepa_t8
DIMS="subject_consistency,background_consistency,temporal_flickering,motion_smoothness,aesthetic_quality,imaging_quality,dynamic_degree"
ts() { date +%H:%M:%S; }
cd "$CF"
[ -f "$STUDENT" ] || { echo "[jepa-t8-eval $(ts)] ABORT: ckpt missing $STUDENT"; exit 1; }
awk 'NR%2==1' "$WE/vbench_prompts.txt" > "$WE/p_g0.txt"; awk 'NR%2==0' "$WE/vbench_prompts.txt" > "$WE/p_g1.txt"
rm -rf "$GEN_S" "$GEN_S".g0.log "$GEN_S".g1.log; mkdir -p "$GEN_S"
CM="--config_path $CONFIG --checkpoint_path $BASE --use_ema --num_output_frames 21 --vbench_naming --batch_size 2 --surgery_ckpt $STUDENT --surgery_ratio 0.75 --surgery_groups 4 --surgery_lora_r 8"
echo "[jepa-t8-eval $(ts)] regen 944 STUDENT_jepa_t8 (MSE+0.25 JEPA-T8 ckpt) — 2x single-GPU batch=2"
CUDA_VISIBLE_DEVICES=0 $PY $CF/inference.py $CM --data_path "$WE/p_g0.txt" --output_folder "$GEN_S" > "$GEN_S".g0.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 $PY $CF/inference.py $CM --data_path "$WE/p_g1.txt" --output_folder "$GEN_S" > "$GEN_S".g1.log 2>&1 &
wait
NS=$(ls "$GEN_S"/*.mp4 2>/dev/null | wc -l); echo "[jepa-t8-eval $(ts)] student_jepa_t8 videos: $NS/944"
[ "$NS" -lt 100 ] && { echo "[jepa-t8-eval $(ts)] ABORT: gen $NS broken"; exit 1; }
echo "[jepa-t8-eval $(ts)] === scoring Quality-7 (student_jepa_t8; teacher/baselines reused from cache) ==="
CUDA_VISIBLE_DEVICES=0 $PY $WE/vbench_score.py "$GEN_S" wan_student_jepa_t8 "$VBOUT" "$DIMS" > "$WE/vb_student_jepa_t8.log" 2>&1
echo "[jepa-t8-eval $(ts)] === teacher(88.09) vs MSE-only cont9k(83.84) vs JEPA-T2(83.90) vs JEPA-T8 ==="
$PY - "$VBOUT" <<'PY'
import json, sys, os
vb=sys.argv[1]; g=lambda d,k:(d[k][0] if isinstance(d[k],list) else d[k])
def load(n):
    try: return json.load(open(os.path.join(vb,n+"_eval_results.json")))
    except Exception: return None
t=load("wan_teacher"); c=load("wan_student_cont"); j2=load("wan_student_jepa"); j8=load("wan_student_jepa_t8")
keys=list(t.keys())
print(f"{'dimension':26}{'teacher':>9}{'MSEonly':>9}{'JEPA-T2':>9}{'JEPA-T8':>9}{'dT8-MSE':>9}{'dT8-T2':>9}")
for k in keys:
    cv=g(c,k)*100 if c else float('nan'); v2=g(j2,k)*100 if j2 else float('nan'); v8=g(j8,k)*100
    print(f"{k:26}{g(t,k)*100:9.2f}{cv:9.2f}{v2:9.2f}{v8:9.2f}{v8-cv:+9.2f}{v8-v2:+9.2f}")
avg=lambda d: sum(g(d,k) for k in keys)/len(keys)*100
qt=avg(t); qc=avg(c) if c else float('nan'); q2=avg(j2) if j2 else float('nan'); q8=avg(j8)
print(f"\n{'Quality avg (7)':26}{qt:9.2f}{qc:9.2f}{q2:9.2f}{q8:9.2f}{q8-qc:+9.2f}{q8-q2:+9.2f}")
dd='dynamic_degree'
ddc=g(c,dd)*100 if c else float('nan'); dd2=g(j2,dd)*100 if j2 else float('nan'); dd8=g(j8,dd)*100
print(f"  dynamic_degree: MSE-only {ddc:.2f} -> JEPA-T2 {dd2:.2f} ({dd2-ddc:+.2f}, the run-1 regression) -> JEPA-T8 {dd8:.2f} ({dd8-ddc:+.2f} vs MSE, {dd8-dd2:+.2f} vs T2)")
recovered = dd8 >= ddc - 1.0
net_t2 = q8 > q2; net_mse = q8 > qc
print(f"  §3a RUN-2 VERDICT: dynamic_degree {'RECOVERED' if recovered else 'STILL SUPPRESSED'}; "
      f"net vs JEPA-T2 {q8-q2:+.2f} ({'win' if net_t2 else 'no win'}); net vs MSE-only {q8-qc:+.2f} ({'win' if net_mse else 'no win'})")
json.dump({"teacher":{k:g(t,k) for k in keys},"mse_only":{k:g(c,k) for k in keys} if c else None,
           "jepa_t2":{k:g(j2,k) for k in keys} if j2 else None,"jepa_t8":{k:g(j8,k) for k in keys},
           "quality_avg":{"teacher":qt,"mse_only":qc,"jepa_t2":q2,"jepa_t8":q8}},
          open(os.path.join(vb,"wan_vbench_compare_jepa_t8.json"),"w"), indent=2)
print("saved", os.path.join(vb,"wan_vbench_compare_jepa_t8.json"))
PY
echo "=== [jepa-t8-eval $(ts)] DONE ==="
