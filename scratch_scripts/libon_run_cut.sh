#!/bin/bash
# Libon port, trimmed schedule. Both adaptive arms stop at 75 steps and are evaluated at
# ckpt25/50/75.
#
# Why: the continuous arm's benign-prompt degeneracy went 0.00 at step 25 -> 0.31 at step 50
# (word loops + token garbage, mean length 73 -> 53). The paper reports results only up to the
# last checkpoint holding >=80% of base utility, so checkpoints at 100/150 would sit far past
# their own stopping rule. 25/50/75 brackets the boundary instead of overshooting it.
set -u
cd /workspace/reward-depth/libon
source /venv/main/bin/activate
export HF_HOME=/workspace/.hf_home
export HF_TOKEN=$(grep HF_TOKEN /workspace/.env | cut -d'"' -f2)
export HF_DATASETS_DISABLE_PROGRESS_BARS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
L=/workspace/kfac

echo "=== continuous: evals (trained to 75) ==="
for C in 25 50 75; do
  CKPT=/workspace/libon_continuous/ckpt$C TAG=continuous_ckpt$C python libon_eval.py \
    > $L/libon_eval_continuous_$C.log 2>&1
  echo "  --- continuous ckpt$C"; grep -E "degeneracy|utility|judge:" $L/libon_eval_continuous_$C.log
done

echo "=== retrained: train 75 ==="
REGIME=retrained STEPS=75 python libon_train.py > $L/libon_train_retrained.log 2>&1
echo "  trained"
for C in 25 50 75; do
  CKPT=/workspace/libon_retrained/ckpt$C TAG=retrained_ckpt$C python libon_eval.py \
    > $L/libon_eval_retrained_$C.log 2>&1
  echo "  --- retrained ckpt$C"; grep -E "degeneracy|utility|judge:" $L/libon_eval_retrained_$C.log
done
echo "LIBON DONE"
