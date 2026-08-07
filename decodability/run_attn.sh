#!/bin/bash
# The attention rung (AttnScalar), sequential -- it holds the model AND sequence features on the
# card, so it is the one family-A path that competes with family B for GPU memory.
set -u
cd "$(dirname "$0")/.."
source /venv/main/bin/activate
[ -f /workspace/.env ] && source /workspace/.env
export DEC_RUNGS=attn
export DEC_SEEDS=${DEC_SEEDS:-0,1}
DATASETS=${DATASETS:-"brit_language brit_culture brit_truth uf styc"}
for m in "$@"; do
  for d in $DATASETS; do
    echo "=== attn $m x $d ==="
    python decodability/dec_scalar.py "$m" "$d" chat \
      > "/workspace/dec_cache/log_attn_${m}_${d}.log" 2>&1 || echo "FAILED $m/$d"
  done
done
echo "=== attn sweep done ==="
