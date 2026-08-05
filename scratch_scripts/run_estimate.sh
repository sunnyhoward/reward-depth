#!/bin/bash
# K-FAC factor estimation for the EAGLE leash test (NEXT.md queue item 1).
# Full 7-projection coverage of layers 0..24 — matches exactly the modules the
# L24 stage-1 LoRA adapts, so the leash is not partial. 37.4 GiB dense fp32
# accumulators, all GPU-resident (--placement model): the old box's 3h+ CPU pass
# was the auto placement pushing the 11008-dim mlp factors to CPU.
set -euo pipefail
cd /workspace/kfac
source /venv/main/bin/activate
export HF_TOKEN=$(grep HF_TOKEN /workspace/.env | cut -d'"' -f2)
ARGS=$(sed 's/^/--target /' targets_L24.txt | tr '\n' ' ')
exec replay-kfac-ewc estimate \
  --model Qwen/Qwen2.5-3B \
  --corpus replay/library.jsonl \
  --output factors_qwen3b_L24 \
  $ARGS \
  --batch-size 8 \
  --device cuda:0 \
  --placement model \
  --eigh-device cuda:0 \
  --max-dense-gib 80 \
  --checkpoint kfac-accum-L24.pt \
  --checkpoint-every 200
