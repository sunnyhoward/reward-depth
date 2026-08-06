#!/bin/bash
# Libon port — regimes + evals, resuming from stage-0 artifacts already in /workspace/libon
# (probe_corpus.jsonl, probes_init.pt, eval_base.json).
#
# Restarted after fixing an OOM: holding the probe backward graph and a 16-sequence full-vocab
# KL graph simultaneously allocated 84 GiB at step 0. The two loss terms are now backward-ed
# separately (identical gradients, since the loss is a sum) and the KL is micro-batched.
# frozen is the control arm, so it gets one eval; the adaptive arms get a utility trajectory
# because the 80%-utility line is what decides which of their checkpoints are readable.
set -u
cd /workspace/reward-depth/libon
source /venv/main/bin/activate
export HF_HOME=/workspace/.hf_home
export HF_TOKEN=$(grep HF_TOKEN /workspace/.env | cut -d'"' -f2)
export HF_DATASETS_DISABLE_PROGRESS_BARS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
L=/workspace/kfac

for R in frozen continuous retrained; do
  echo "=== $R : train ==="
  REGIME=$R STEPS=150 python libon_train.py > $L/libon_train_$R.log 2>&1
  echo "  trained"
  CKPTS="150"; [ "$R" != "frozen" ] && CKPTS="50 100 150"
  for C in $CKPTS; do
    CKPT=/workspace/libon_$R/ckpt$C TAG=${R}_ckpt$C python libon_eval.py \
      > $L/libon_eval_${R}_$C.log 2>&1
    echo "  --- $R ckpt$C"; grep -E "degeneracy|utility|judge:" $L/libon_eval_${R}_$C.log
  done
done
echo "LIBON DONE"
