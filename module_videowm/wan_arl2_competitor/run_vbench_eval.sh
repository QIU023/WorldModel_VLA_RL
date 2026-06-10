#!/bin/bash
# ARL2-comparable VBench eval using STANDARD implementations only:
#   - generation: Causal-Forcing's own inference.py / CausalInferencePipeline (few-step causal sampler)
#   - scoring:    official vbench 0.1.5  VBench.evaluate(mode='vbench_standard')  on the 944-prompt suite
# Our only addition: inference.py --surgery_ckpt injects the KDA+AttnRes+LoRA student into the standard generator.
# NOTE: CF inference.py's torchrun DDP path has a device bug (.to(gpu=cuda:0) ignores local_rank -> cuda:1 inputs
# mismatch). So we parallelize via 2 single-GPU processes (prompt-sharded) instead — single-GPU is the path the
# sanity validated. Usage: STUDENT_CKPT=... NFRAMES=21 bash run_vbench_eval.sh
set -u
EXT=/home/wm-vla-rl/_ext; CF=$EXT/Causal-Forcing; PY=/home/venv/cosmos_predict2/bin/python; WE=$EXT/wan_eval
PROMPTS=$WE/vbench_prompts.txt
CONFIG=$CF/configs/causal_forcing_dmd_framewise_2step.yaml
BASE_CKPT=/home/causal_forcing_ckpt/framewise/causal_forcing.pt   # teacher base (generator_ema)
STUDENT_CKPT=${STUDENT_CKPT:-$EXT/wan_lin_out/wanlin_s2_step3000.safetensors}
NFRAMES=${NFRAMES:-21}                                            # 21 latent -> 81 pixel frames (ARL2's 81-frame point)
BS=${BS:-8}                                                       # batched T2V per GPU (fills 140GB; far faster than batch=1)
GEN_T=$WE/gen_teacher; GEN_S=$WE/gen_student; VBOUT=$WE/vbench_out
ts() { date +%H:%M:%S; }
cd "$CF"
COMMON="--config_path $CONFIG --checkpoint_path $BASE_CKPT --use_ema --num_output_frames $NFRAMES --vbench_naming --batch_size $BS"

# prompt shards: GPU0 gets odd lines, GPU1 gets even lines (disjoint, together = all 944)
awk 'NR%2==1' "$PROMPTS" > "$WE/p_g0.txt"; awk 'NR%2==0' "$PROMPTS" > "$WE/p_g1.txt"

gen() {  # $1=output_folder ; $2.. = extra args (surgery). 2 single-GPU procs in parallel, disjoint prompt shards.
  local out="$1"; shift
  mkdir -p "$out"
  CUDA_VISIBLE_DEVICES=0 $PY $CF/inference.py $COMMON --data_path "$WE/p_g0.txt" --output_folder "$out" "$@" > "$out.g0.log" 2>&1 &
  CUDA_VISIBLE_DEVICES=1 $PY $CF/inference.py $COMMON --data_path "$WE/p_g1.txt" --output_folder "$out" "$@" > "$out.g1.log" 2>&1 &
  wait
}

echo "[eval $(ts)] === Phase 1: generate TEACHER (944 prompts, 2x single-GPU) ==="
gen "$GEN_T"
NT=$(ls "$GEN_T"/*.mp4 2>/dev/null | wc -l); echo "[eval $(ts)] teacher videos: $NT/944"
if [ "$NT" -lt 100 ]; then echo "[eval $(ts)] ABORT: teacher gen produced $NT videos (broken). see $GEN_T.g0.log/.g1.log"; exit 1; fi

echo "[eval $(ts)] === Phase 2: generate STUDENT (KDA+AttnRes+LoRA 75%, 2x single-GPU) ==="
gen "$GEN_S" --surgery_ckpt "$STUDENT_CKPT" --surgery_ratio 0.75 --surgery_groups 4 --surgery_lora_r 8
NS=$(ls "$GEN_S"/*.mp4 2>/dev/null | wc -l); echo "[eval $(ts)] student videos: $NS/944"
if [ "$NS" -lt 100 ]; then echo "[eval $(ts)] ABORT: student gen produced $NS videos (broken). see $GEN_S.g0.log/.g1.log"; exit 1; fi

echo "[eval $(ts)] === Phase 3: official VBench scoring (vbench_standard, all 16 dims) ==="
PIP=/home/venv/cosmos_predict2/bin/pip
score() {  # try on current transformers (5.10 for Wan); if it errors, fall back to 4.33.2 then restore
  local dir="$1" nm="$2" log="$3"
  CUDA_VISIBLE_DEVICES=0 $PY $WE/vbench_score.py "$dir" "$nm" "$VBOUT" all > "$log" 2>&1
  if [ $? -ne 0 ] && grep -qiE "transformers|ImportError|AttributeError|cannot import" "$log"; then
    echo "[eval $(ts)] $nm scoring failed on current transformers -> swap to 4.33.2 and retry"
    $PIP install -q transformers==4.33.2 2>&1 | tail -1
    CUDA_VISIBLE_DEVICES=0 $PY $WE/vbench_score.py "$dir" "$nm" "$VBOUT" all >> "$log" 2>&1
    $PIP install -q transformers==5.10.0 2>&1 | tail -1   # restore for Wan
  fi
}
score "$GEN_T" wan_teacher "$WE/vb_teacher.log"
score "$GEN_S" wan_student "$WE/vb_student.log"

echo "[eval $(ts)] === Phase 4: teacher vs student vs ARL2(75%: Q86.71/S70.24) ==="
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
print(f"{'dimension':28}{'teacher':>9}{'student':>9}{'delta':>8}")
for k in keys: print(f"{k:28}{g(t,k):9.4f}{g(s,k):9.4f}{g(s,k)-g(t,k):+8.4f}")
json.dump({"teacher":{k:g(t,k) for k in keys},"student":{k:g(s,k) for k in keys}},
          open(os.path.join(vb,"wan_vbench_compare.json"),"w"), indent=2)
print("saved", os.path.join(vb,"wan_vbench_compare.json"))
PY
echo "=== [eval $(ts)] VBench eval COMPLETE ==="
