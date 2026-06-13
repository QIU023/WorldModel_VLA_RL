#!/usr/bin/env bash
# Apply the KDA-world-model integration to a clean VLA-RFT checkout.
#
# Usage:  bash apply_patches.sh [/path/to/VLA-RFT]
# Default repo: /home/wm-vla-rl/_ext/VLA-RFT
#
# This (1) git-applies the 3 source patches and (2) copies the 3 new modules into the
# verl rollout package. Idempotent-ish: re-running will fail the git apply if already
# applied (use `git apply -R patches/*.patch` to revert first).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${1:-/home/wm-vla-rl/_ext/VLA-RFT}"
ROLLOUT="$REPO/train/verl/verl/workers/rollout"

echo "[apply] repo = $REPO"
cd "$REPO"

for p in 01_fsdp_workers 02_rollout_init 03_trainer_yaml; do
  echo "[apply] git apply patches/$p.patch"
  git apply --check "$HERE/patches/$p.patch"
  git apply        "$HERE/patches/$p.patch"
done

echo "[apply] copy new modules -> $ROLLOUT"
cp "$HERE/hf_world_model_rollout.py" "$ROLLOUT/hf_world_model_rollout.py"
cp "$HERE/wm_kda_surgery.py"         "$ROLLOUT/wm_kda_surgery.py"
cp "$HERE/kda_llama.py"              "$ROLLOUT/kda_llama.py"

echo "[apply] done. To enable KDA at launch, add to run_vla_rft.sh:"
echo "    world_model_rollout.rollout.name=hf \\"
echo "    world_model_rollout.world_model.attn_type=kda \\"
echo "    world_model_rollout.world_model.kda_layers=3 \\"
echo "    world_model_rollout.world_model.attnres_groups=4 \\"
echo "    world_model_rollout.rollout.hf_recurrent=True"
