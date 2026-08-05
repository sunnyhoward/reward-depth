#!/bin/bash
# Rebuild the EAGLE replay corpus + replay-distilled tf heads (both died with the last box).
# Prerequisite for BOTH queue items: item 1's L24 stage-1 cells and item 2's encoding-depth
# remeasurement read /workspace/eagle_head_tf_L{L}.pt.
set -euo pipefail
cd /workspace/reward-depth
source /venv/main/bin/activate
export HF_TOKEN=$(grep HF_TOKEN /workspace/.env | cut -d'"' -f2)

if [ ! -f /workspace/eagle_replay_2048x128.pt ]; then
  echo "=== replay corpus ==="
  python eagle/eagle_replay.py
fi

echo "=== tf heads, replay-distilled, L=4,12,24,32 ==="
HEAD_ARCH=tf HEAD_DATA=replay LAYERS=4,12,24,32 python eagle/eagle_head.py

echo "=== promote _replay.pt into the canonical tf slots (§17 convention) ==="
for L in 4 12 24 32; do
  src=/workspace/eagle_head_tf_L${L}_replay.pt
  dst=/workspace/eagle_head_tf_L${L}.pt
  if [ -f "$src" ]; then cp -f "$src" "$dst"; echo "  $dst <- $src"; fi
done
ls -la /workspace/eagle_head_tf_L*.pt
