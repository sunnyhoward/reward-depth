#!/bin/bash
# Depth sweep in the LOGISTIC continuous regime — the best arm in the study
# (continuous_ckpt25: compliance 0.408 from base 0.583, zero broken, utility 0.480). The Bayesian
# arms were run first but lose to it on both axes once compliance is adjusted for brokenness, so
# the depth question is asked in the setting that actually works.
#
# Everything identical to the all-layer continuous arm except WHICH layers carry the probe loss:
#   band_early  {0, 6}      evidence weights .015 / .101   (ELBO says: almost nothing at L0)
#   band_mid    {12, 18}    evidence weights .284 / .269   (the evidence peak)
#   band_late   {24, 30}    evidence weights .209 / .121
#
# The question: does suppression follow the evidence? Every arm so far supervised all six at
# once, so "L12/L18 carry the harmfulness evidence" is a statement about readability, not about
# which probe does the work. Baseline for comparison is the existing all-six continuous arm.
set -u
cd /workspace/reward-depth/libon
export HF_HOME=/workspace/.hf_home
export HF_TOKEN=$(grep HF_TOKEN /workspace/.env | cut -d'"' -f2)
export HF_DATASETS_DISABLE_PROGRESS_BARS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export KL_MICRO=2
L=/workspace/kfac

# don't start until the first eval queue drains (2 slots) and atten0.31 finishes
until grep -q "EVALQUEUE DONE" $L/libon_evalqueue.log 2>/dev/null; do sleep 20; done
# NOT `pgrep -f` — NEXT.md standing trap: pgrep waiters deadlock here because the tool-wrapper
# shells carry the queued command text in their own cmdline, so a waiter matches its own parent.
# Wait on the log line the job itself writes instead.
until grep -q "^DONE" $L/libon_bayes_atten0.31.log 2>/dev/null; do sleep 20; done
echo "slots free; starting depth sweep"

train_one () {
  set -u
  cd /workspace/reward-depth/libon
  source /venv/main/bin/activate
  NAME=$(echo "$1" | awk '{print $1}'); LYRS=$(echo "$1" | awk '{print $2}')
  LIBON_LAYERS=$LYRS REGIME=continuous STEPS=75 KL_MICRO=2 \
    RUN_DIR=/workspace/libon_depth_$NAME python libon_train.py \
    > /workspace/kfac/libon_depth_$NAME.log 2>&1
  echo "  [$NAME] trained"
}
export -f train_one
printf 'early 0,6\nmid 12,18\nlate 24,30\n' > /tmp/libon_depthjobs.txt
xargs -a /tmp/libon_depthjobs.txt -d '\n' -I{} -P 2 bash -c 'train_one "{}"'

eval_one () {
  set -u
  cd /workspace/reward-depth/libon
  source /venv/main/bin/activate
  TAG=$(echo "$1" | awk '{print $1}'); CK=$(echo "$1" | awk '{print $2}')
  [ -d "$CK" ] || { echo "  [$TAG] MISSING $CK"; return 0; }
  CKPT=$CK TAG=$TAG python libon_eval.py > /workspace/kfac/libon_eval_$TAG.log 2>&1
  echo "  [$TAG] $(grep -E 'judge:' /workspace/kfac/libon_eval_$TAG.log | tail -1)"
}
export -f eval_one
: > /tmp/libon_deptheval.txt
for B in early mid late; do
  for C in 25 75; do
    echo "depth_${B}_ckpt$C /workspace/libon_depth_$B/ckpt$C" >> /tmp/libon_deptheval.txt
  done
done
xargs -a /tmp/libon_deptheval.txt -d '\n' -I{} -P 2 bash -c 'eval_one "{}"'

# atten0.31 evals too (section-3 control), now that a slot is free
for C in 25 75; do
  ( source /venv/main/bin/activate
    CKPT=/workspace/libon_bayes_atten0.31/ckpt$C TAG=bayes_atten0.31_ckpt$C \
      python libon_eval.py > $L/libon_eval_atten_$C.log 2>&1
    echo "  [atten0.31 ckpt$C] $(grep -E 'judge:' $L/libon_eval_atten_$C.log | tail -1)" )
done

source /venv/main/bin/activate
python libon_report.py
echo "DEPTH DONE"
