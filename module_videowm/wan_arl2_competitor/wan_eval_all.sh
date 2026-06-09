#!/bin/bash
# One-shot full eval after S2 finishes: 逼近教师 velocity-match + efficiency F-sweep + teacher/student video gen
# (dual-GPU parallel). VBench scoring is a separate step (needs transformers 4.33.2 swap; see wan_vbench step at end).
# Usage: bash wan_eval_all.sh
set -u
EXT=/home/wm-vla-rl/_ext; CF=$EXT/Causal-Forcing; PY=/home/venv/cosmos_predict2/bin/python
OUT=$EXT/wan_lin_out; GEN=/home/wan_eval_gen; NVID=${NVID:-32}
S2=$(ls -t "$OUT"/wanlin_s2_step*.safetensors 2>/dev/null | head -1)
if [ -z "$S2" ]; then echo "[eval] no S2 ckpt in $OUT — run distillation first"; exit 1; fi
mkdir -p "$GEN"; cd "$CF"; ts() { date +%H:%M:%S; }
echo "[eval $(ts)] student ckpt = $S2"

# 1) 逼近教师 velocity-match (held-out latents: rel-MSE + cosine, student vs teacher)
echo "[eval $(ts)] (1/4) velocity-match"
CUDA_VISIBLE_DEVICES=0 $PY "$EXT/wan_eval.py" --mode velocity --resume "$S2" --n 64 > "$EXT/eval_velocity.log" 2>&1
grep -aE "velocity-match|rel_MSE|cosine" "$EXT/eval_velocity.log" | tail -3

# 2) efficiency micro-benchmark (KDA vs softmax, F 21->231 = px 81->921)
echo "[eval $(ts)] (2/4) efficiency F-sweep"
CUDA_VISIBLE_DEVICES=0 $PY "$EXT/wan_eval.py" --mode efficiency > "$EXT/eval_efficiency.log" 2>&1
grep -aE "\[eff\] F=|speedup" "$EXT/eval_efficiency.log" | tail -8

# 3) VBench video generation — teacher (GPU0) ∥ student (GPU1), same pipeline/mode (fair)
echo "[eval $(ts)] (3/4) generate $NVID teacher ∥ student videos @832x480 81f"
CUDA_VISIBLE_DEVICES=0 $PY "$EXT/wan_generate.py" --mode teacher --n $NVID --out "$GEN" > "$EXT/gen_teacher.log" 2>&1 &
CUDA_VISIBLE_DEVICES=1 $PY "$EXT/wan_generate.py" --mode student --resume "$S2" --n $NVID --out "$GEN" > "$EXT/gen_student.log" 2>&1 &
wait
echo "[eval $(ts)] generated: $(ls "$GEN"/vids_*.pt 2>/dev/null | wc -l) sets"

# 4) VBench scoring — separate (transformers 4.33.2 vs 5.10 conflict). Reuse cosmos_vbench.py (model-agnostic).
echo "[eval $(ts)] (4/4) VBench: run after this with the transformers swap, e.g."
echo "    bash $EXT/wan_vbench_env.sh   # scores vids_teacher.pt vs vids_student.pt on 6 reference-free dims"
echo "=== [eval $(ts)] velocity+efficiency+gen DONE; VBench is the final step ==="
