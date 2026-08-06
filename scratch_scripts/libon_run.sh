#!/bin/bash
# Libon et al. port, end to end. See libon/NOTE.md.
#
# The independent variable is the probe-update regime; everything else is held at the paper's
# values. Utility is measured at three checkpoints per adaptive regime because their 80%-utility
# line is what decides which checkpoints are readable at all.
set -u
cd /workspace/reward-depth/libon
source /venv/main/bin/activate
export HF_HOME=/workspace/.hf_home
export HF_TOKEN=$(grep HF_TOKEN /workspace/.env | cut -d'"' -f2)
export HF_DATASETS_DISABLE_PROGRESS_BARS=1
L=/workspace/kfac    # log dir (reused; nothing k-fac about it)

echo "=== 0. probe corpus + initial probes ==="
N_PROBE=768 PROBE_STEPS=1500 MAX_NEW=96 python libon_prepare.py > $L/libon_prepare.log 2>&1
grep -E "AUROC|corpus\]|extract\]" $L/libon_prepare.log | tail -4

echo "=== 1. base eval ==="
CKPT=base TAG=base python libon_eval.py > $L/libon_eval_base.log 2>&1
grep -E "degeneracy|utility|judge:" $L/libon_eval_base.log

for R in frozen continuous retrained; do
  echo "=== 2.$R : train ==="
  REGIME=$R STEPS=150 python libon_train.py > $L/libon_train_$R.log 2>&1
  echo "  trained"
  for C in 50 100 150; do
    CKPT=/workspace/libon_$R/ckpt$C TAG=${R}_ckpt$C python libon_eval.py \
      > $L/libon_eval_${R}_$C.log 2>&1
    echo "  --- $R ckpt$C"; grep -E "degeneracy|utility|judge:" $L/libon_eval_${R}_$C.log
  done
done
echo "LIBON DONE"
