#!/bin/bash
# Family B across the full grid, strictly sequential (see run_familyB.sh for why).
# Smallest model first, so a failure surfaces in minutes rather than after the 8B replay corpus.
set -u
cd "$(dirname "$0")/.."
for m in qwen3-0.6b qwen3-1.7b qwen3-4b qwen3-8b; do
  bash decodability/run_familyB.sh "$m" "eagle-mlp,eagle-attn,eagle-tf,eagle-2l" \
    || echo "FAILED: $m"
done
echo "=== family B grid complete ==="
