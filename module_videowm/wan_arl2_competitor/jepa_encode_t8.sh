#!/bin/bash
# §3a RUN-2 re-encode: V-JEPA2 target with T_VJEPA=8 temporal slices + spatial 2x2 pool -> [8,15,26,1408] fp16.
# Run-1 used T=2 @ 30x52 which made the temporally-interpolated target SMOOTH -> suppressed student motion
# (VBench dynamic_degree -5.55). T=8 keeps frame-to-frame variation; 2x2 spatial pool keeps disk equal (~52.7GB).
# FREES the old T=2 feats first (run-1 ckpts jepa_l025/ + result JSON are already saved, so they're disposable).
# 2-GPU sharded, ~2.25h. Usage: nohup bash jepa_encode_t8.sh & ; tail -f jepa_encode_t8.log
set -u
HERE=/home/wm-vla-rl/module_videowm/wan_arl2_competitor; PY=/home/venv/cosmos_predict2/bin/python
OUT=/home/wan_jepa_feats
ts() { date +%H:%M:%S; }
echo "[encode $(ts)] freeing old T=2 feats in $OUT (run-1 ckpts/JSON saved separately)"
rm -rf "$OUT"/*
df -h /home | tail -1
echo "[encode $(ts)] launching 2-GPU sharded T=8 encode -> $OUT (target ~52.7GB, [8,15,26,1408])"
CUDA_VISIBLE_DEVICES=0 JEPA_T=8 $PY "$HERE/jepa_preproc_encode.py" 0 2 > "$HERE/jepa_encode_t8.g0.log" 2>&1 &
P0=$!
CUDA_VISIBLE_DEVICES=1 JEPA_T=8 $PY "$HERE/jepa_preproc_encode.py" 1 2 > "$HERE/jepa_encode_t8.g1.log" 2>&1 &
P1=$!
wait $P0 $P1
N=$(ls "$OUT"/clip_*.pt 2>/dev/null | wc -l)
echo "[encode $(ts)] DONE: $N/6000 feats in $OUT"
du -sh "$OUT"; df -h /home | tail -1
