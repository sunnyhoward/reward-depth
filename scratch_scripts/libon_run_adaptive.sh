#!/bin/bash
# Libon port, remaining arms. frozen was TRUNCATED at step 40 (ckpt25 kept): its probe loss
# saturated at ~0.01 by step 20 and sat there, so further steps could not change what that arm
# demonstrates. It is still evaluated — the claim "probe score falls while the model stays
# harmful" needs the compliance number, not just the loss curve.
set -u
cd /workspace/reward-depth/libon
source /venv/main/bin/activate
export HF_HOME=/workspace/.hf_home
export HF_TOKEN=$(grep HF_TOKEN /workspace/.env | cut -d'"' -f2)
export HF_DATASETS_DISABLE_PROGRESS_BARS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
L=/workspace/kfac

echo "=== frozen ckpt25 eval (truncated arm) ==="
CKPT=/workspace/libon_frozen/ckpt25 TAG=frozen_ckpt25 python libon_eval.py \
  > $L/libon_eval_frozen_25.log 2>&1
grep -E "degeneracy|utility|judge:" $L/libon_eval_frozen_25.log

for R in continuous retrained; do
  echo "=== $R : train ==="
  REGIME=$R STEPS=150 python libon_train.py > $L/libon_train_$R.log 2>&1
  echo "  trained"
  for C in 50 100 150; do
    CKPT=/workspace/libon_$R/ckpt$C TAG=${R}_ckpt$C python libon_eval.py \
      > $L/libon_eval_${R}_$C.log 2>&1
    echo "  --- $R ckpt$C"; grep -E "degeneracy|utility|judge:" $L/libon_eval_${R}_$C.log
  done
done
echo "LIBON DONE"
