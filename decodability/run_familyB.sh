#!/bin/bash
# Family B: distil heads, then score pairs through them. STRICTLY SEQUENTIAL.
#
# Two reasons this must not be parallelised:
#   - fp32 log_softmax over a 151,936-token vocab is the peak allocation here, and NEXT.md:60-61
#     records three concurrent such jobs exhausting the 95 GB card.
#   - the replay corpus is per-model and built on first use; two processes racing to build it
#     would write the same file twice.
#
# Usage: run_familyB.sh <model_key> <arch,arch,...>
set -u
cd "$(dirname "$0")/.."
source /venv/main/bin/activate
[ -f /workspace/.env ] && source /workspace/.env

MODEL=${1:?usage: run_familyB.sh <model_key> <archs>}
ARCHS=${2:-eagle-mlp,eagle-attn,eagle-tf,eagle-2l}

echo "=== distil $MODEL [$ARCHS] ==="
python decodability/dec_distill.py "$MODEL" "$ARCHS" || { echo "DISTILL FAILED: $MODEL"; exit 1; }

echo "=== score $MODEL [$ARCHS] ==="
python decodability/dec_through.py "$MODEL" "$ARCHS" all || { echo "THROUGH FAILED: $MODEL"; exit 1; }

echo "=== family B done: $MODEL ==="
