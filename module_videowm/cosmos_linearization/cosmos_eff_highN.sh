#!/bin/bash
# Push the efficiency sweep to minute-scale N (up to the model's max 128 latent frames × high res) to show the
# O(N) vs O(N^2) gap at SANA-WM's regime. Auto-fires when the DAVIS overnight completes. GPU0.
# Usage: nohup bash cosmos_eff_highN.sh > cosmos_eff_highN.log 2>&1 &
set -u
EXT=/home/wm-vla-rl/_ext; PY=/home/venv/cosmos_predict2/bin/python
SNAP=$(ls -d /home/cosmos_predict2_model/models--nvidia--Cosmos-Predict2-2B-Video2World/snapshots/*/ | head -1)
TF="${SNAP}transformer"; S2D="$EXT/cosmos_lin_out/r75/linstate_s2d_step2000.safetensors"
cd "$EXT"
echo "[effHN $(date +%H:%M:%S)] waiting for DAVIS overnight to COMPLETE..."
while ! grep -q "COMPLETE" cosmos_davis_overnight.log 2>/dev/null; do sleep 30; done
sleep 20   # let GPU settle
echo "[effHN $(date +%H:%M:%S)] DAVIS done — running high-N efficiency (N up to ~1.2M)"
# N = T_latentframes * (size/8)^2 ; model max temporal = 128 latent frames
$PY cosmos_eval.py --mode efficiency --transformer "$TF" --ratio 0.75 --resume "$S2D" \
    --size 512  --frames 32,64,96,128 > "$EXT/eff_highN_512.log" 2>&1
$PY cosmos_eval.py --mode efficiency --transformer "$TF" --ratio 0.75 --resume "$S2D" \
    --size 768  --frames 32,64,96,128 > "$EXT/eff_highN_768.log" 2>&1
$PY cosmos_eval.py --mode efficiency --transformer "$TF" --ratio 0.75 --resume "$S2D" \
    --size 1024 --frames 16,32,48,64  > "$EXT/eff_highN_1024.log" 2>&1
echo "=== [effHN $(date +%H:%M:%S)] DONE ==="
for s in 512 768 1024; do echo "--- ${s}px ---"; grep "\[eff\]" "$EXT/eff_highN_${s}.log" 2>/dev/null; done
