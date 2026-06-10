#!/bin/bash
# OVERNIGHT autonomous chain for the ARL2-competitor:
#   0) wait for S2c to finish (final wanlin_s2_step3000.safetensors)
#   1) SANITY: 4-prompt student gen -> verify surgery key-match (unexpected=0) + videos non-empty (abort if broken)
#   2) FULL VBench eval (944 prompts, teacher+student, DDP-2, 81 px frames) via run_vbench_eval.sh (standard impls)
#   3) EFFICIENCY F-sweep (same student vs teacher @ 81/165/501/921 frames; NO retrain — linear-attn length-generalizes)
#   4) write REPORT
# Usage: nohup bash wan_overnight.sh > wan_overnight.log 2>&1 &
set -u
EXT=/home/wm-vla-rl/_ext; CF=$EXT/Causal-Forcing; PY=/home/venv/cosmos_predict2/bin/python
OUT=$EXT/wan_lin_out; STUDENT=$OUT/wanlin_s2_step3000.safetensors
CONFIG=$CF/configs/causal_forcing_dmd_framewise_2step.yaml
BASE=/home/causal_forcing_ckpt/framewise/causal_forcing.pt
PROMPTS=$EXT/wan_eval/vbench_prompts.txt
REPORT=$EXT/wan_eval/OVERNIGHT_REPORT.md
ts() { date '+%m-%d %H:%M:%S'; }
log() { echo "[overnight $(ts)] $*"; }

log "=== START. waiting for S2c final ckpt: $STUDENT ==="
# S2c writes wanlin_s2_step3000.safetensors once, after its loop ends (then exits). That file = done signal.
while [ ! -f "$STUDENT" ]; do sleep 60; done
# ckpt is written before S2c's barrier+destroy+exit; wait for the process to fully release both GPUs
while [ "$(pgrep -f 'wan_distill.py --stage 2' | wc -l)" -gt 0 ]; do sleep 10; done
sleep 15
log "S2c done + GPUs released. student ckpt present ($(du -h "$STUDENT"|cut -f1)). final S2c log:"
tr '\r' '\n' < $EXT/wan_s2c.log | grep -aE '\[s2\] step' | tail -2

# ---- Phase 1: SANITY (test BOTH teacher AND student gen paths, 2 prompts each, single GPU) ----
log "=== Phase 1: SANITY teacher+student gen ==="
head -2 "$PROMPTS" > $EXT/wan_eval/sanity_prompts.txt
cd "$CF"
# teacher path (no surgery) — catches flash-attn / base-load issues the student path would mask
ST=$EXT/wan_eval/gen_sanity_t; rm -rf "$ST"
CUDA_VISIBLE_DEVICES=0 $PY $CF/inference.py --config_path $CONFIG --checkpoint_path $BASE --use_ema \
  --data_path $EXT/wan_eval/sanity_prompts.txt --num_output_frames 21 --vbench_naming --batch_size 2 \
  --output_folder "$ST" > $EXT/wan_eval/sanity_teacher.log 2>&1
NVT=$(ls "$ST"/*.mp4 2>/dev/null | wc -l)
# student path (surgery)
SAN=$EXT/wan_eval/gen_sanity; rm -rf "$SAN"
CUDA_VISIBLE_DEVICES=0 $PY $CF/inference.py --config_path $CONFIG --checkpoint_path $BASE --use_ema \
  --surgery_ckpt $STUDENT --surgery_ratio 0.75 --surgery_groups 4 --surgery_lora_r 8 \
  --data_path $EXT/wan_eval/sanity_prompts.txt --num_output_frames 21 --vbench_naming --batch_size 2 \
  --output_folder "$SAN" > $EXT/wan_eval/sanity.log 2>&1
UNEXP=$(grep -aoE "unexpected=[0-9]+" $EXT/wan_eval/sanity.log | head -1 | grep -oE "[0-9]+")
NVID=$(ls "$SAN"/*.mp4 2>/dev/null | wc -l)
log "sanity: teacher_videos=${NVT}  student_videos=${NVID}  surgery_unexpected=${UNEXP:-?}"
if [ "$NVT" -lt 1 ] || [ "${UNEXP:-99}" != "0" ] || [ "$NVID" -lt 1 ]; then
  log "!!! SANITY FAILED (teacher=${NVT}, student=${NVID}, unexpected=${UNEXP:-?}). Aborting. See sanity_teacher.log / sanity.log"
  echo "# OVERNIGHT ABORTED at sanity ($(ts))" > "$REPORT"
  echo "teacher_videos=${NVT} student_videos=${NVID} unexpected=${UNEXP:-?}. See wan_eval/sanity*.log" >> "$REPORT"
  exit 1
fi
log "SANITY PASS (teacher+student both gen) -> full eval"

# ---- Phase 2: FULL VBench (standard impls) ----
log "=== Phase 2: FULL VBench eval (944 prompts, teacher+student) ==="
STUDENT_CKPT=$STUDENT NFRAMES=21 bash $EXT/wan_eval/run_vbench_eval.sh >> $EXT/wan_eval/full_eval.log 2>&1
log "VBench eval done. compare json:"
cat $EXT/wan_eval/vbench_out/wan_vbench_compare.json 2>/dev/null | head -50

# ---- Phase 3: EFFICIENCY F-sweep (same student, no retrain) ----
log "=== Phase 3: efficiency F-sweep (best-effort) ==="
if [ -f $EXT/wan_eval.py ]; then
  CUDA_VISIBLE_DEVICES=0 $PY $EXT/wan_eval.py --student $STUDENT --frames 81,165,501,921 \
    > $EXT/wan_eval/efficiency.log 2>&1 && log "efficiency done" || log "efficiency sweep failed (non-fatal); see efficiency.log"
else
  log "wan_eval.py not present; skipping efficiency (do later)."
fi

# ---- Phase 4: REPORT ----
log "=== writing report ==="
{
  echo "# ARL2-competitor overnight eval — $(ts)"
  echo
  echo "Student: KDA(75%)+AttnRes+LoRA, S2c shortened lr-5e-5 cosine, $STUDENT"
  echo "Target (ARL2 75%): VBench Quality 86.71 / Semantic 70.24 (vs Causal-Forcing teacher Q 87.09)"
  echo
  echo "## VBench (official vbench_standard, 944 prompts, 81 frames, teacher vs OUR student)"
  echo '```'
  $PY - <<'PY'
import json, os
vb="/home/wm-vla-rl/_ext/wan_eval/vbench_out"
try:
    c=json.load(open(os.path.join(vb,"wan_vbench_compare.json")))
    t,s=c["teacher"],c["student"]
    print(f"{'dimension':28}{'teacher':>9}{'student':>9}{'delta':>8}")
    for k in t: print(f"{k:28}{t[k]:9.4f}{s[k]:9.4f}{s[k]-t[k]:+8.4f}")
except Exception as e:
    print("VBench compare not available:", e)
PY
  echo '```'
  echo
  echo "Logs: full_eval.log, vb_teacher.log, vb_student.log, efficiency.log"
} > "$REPORT"
log "=== OVERNIGHT COMPLETE -> $REPORT ==="
