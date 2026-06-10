#!/bin/bash
# ARL2-comparable VBench eval using STANDARD implementations only:
#   - generation: Causal-Forcing's own inference.py / CausalInferencePipeline (few-step causal sampler)
#   - scoring:    official vbench 0.1.5  VBench.evaluate(mode='vbench_standard')  on the 944-prompt suite
# Our only addition: inference.py --surgery_ckpt injects the KDA+AttnRes+LoRA student into the standard generator.
# Run AFTER S2c finishes (needs both GPUs free). Usage: nohup bash run_vbench_eval.sh > run_vbench_eval.log 2>&1 &
set -u
EXT=/home/wm-vla-rl/_ext; CF=$EXT/Causal-Forcing; PY=/home/venv/cosmos_predict2/bin/python
PROMPTS=$EXT/wan_eval/vbench_prompts.txt
CONFIG=$CF/configs/causal_forcing_dmd_framewise_2step.yaml
BASE_CKPT=/home/causal_forcing_ckpt/framewise/causal_forcing.pt   # teacher base (generator_ema)
STUDENT_CKPT=${STUDENT_CKPT:-$EXT/wan_lin_out/wanlin_s2_step3000.safetensors}  # final S2c (override-able)
NFRAMES=${NFRAMES:-21}                                            # 21 latent -> 81 pixel frames (ARL2's 81-frame point)
GEN_T=$EXT/wan_eval/gen_teacher; GEN_S=$EXT/wan_eval/gen_student
VBOUT=$EXT/wan_eval/vbench_out
ts() { date +%H:%M:%S; }
cd "$CF"
TR="$PY -m torch.distributed.run --nproc_per_node=2 --master_port=29577 $CF/inference.py"
COMMON="--config_path $CONFIG --checkpoint_path $BASE_CKPT --use_ema --data_path $PROMPTS --num_output_frames $NFRAMES --vbench_naming"

echo "[eval $(ts)] === Phase 1: generate TEACHER (944 prompts, DDP-2) ==="
$TR $COMMON --output_folder "$GEN_T"
echo "[eval $(ts)] teacher videos: $(ls "$GEN_T"/*.mp4 2>/dev/null | wc -l)"

echo "[eval $(ts)] === Phase 2: generate STUDENT (KDA+AttnRes+LoRA 75%, DDP-2) ==="
$TR $COMMON --surgery_ckpt "$STUDENT_CKPT" --surgery_ratio 0.75 --surgery_groups 4 --surgery_lora_r 8 --output_folder "$GEN_S"
echo "[eval $(ts)] student videos: $(ls "$GEN_S"/*.mp4 2>/dev/null | wc -l)"

echo "[eval $(ts)] === Phase 3: official VBench scoring (vbench_standard, all 16 dims) ==="
CUDA_VISIBLE_DEVICES=0 $PY $EXT/wan_eval/vbench_score.py "$GEN_T" wan_teacher "$VBOUT" all > "$EXT/wan_eval/vb_teacher.log" 2>&1
CUDA_VISIBLE_DEVICES=0 $PY $EXT/wan_eval/vbench_score.py "$GEN_S" wan_student "$VBOUT" all > "$EXT/wan_eval/vb_student.log" 2>&1

echo "[eval $(ts)] === Phase 4: teacher vs student vs ARL2(75%: Q86.71/S70.24) ==="
$PY - "$VBOUT" <<'PY'
import json, sys, os
vb=sys.argv[1]
t=json.load(open(os.path.join(vb,"wan_teacher_eval_results.json")))
s=json.load(open(os.path.join(vb,"wan_student_eval_results.json")))
g=lambda d,k:(d[k][0] if isinstance(d[k],list) else d[k])
keys=list(t.keys())
print(f"{'dimension':28}{'teacher':>9}{'student':>9}{'delta':>8}")
for k in keys:
    print(f"{k:28}{g(t,k):9.4f}{g(s,k):9.4f}{g(s,k)-g(t,k):+8.4f}")
json.dump({"teacher":{k:g(t,k) for k in keys},"student":{k:g(s,k) for k in keys}},
          open(os.path.join(vb,"wan_vbench_compare.json"),"w"), indent=2)
print("saved", os.path.join(vb,"wan_vbench_compare.json"))
PY
echo "=== [eval $(ts)] VBench eval COMPLETE ==="
