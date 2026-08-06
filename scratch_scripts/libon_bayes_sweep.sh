#!/bin/bash
# Bayesian-probe sweep on the Libon pipeline. Concurrency 2 (the box has 98 GiB; each run peaks
# well under 30 GiB, and the baseline driver may still hold ~20 GiB).
#
# Arms, all REGIME=sequential (prior = previous posterior), 75 steps:
#   lam 0    uniform    core section-2 arm, and the lam=0 reference for the sweep
#   lam 0.5  uniform    section 3
#   lam 1    uniform    section 3
#   lam 2    uniform    section 3
#   lam 0    evidence   section 4 ablation vs the lam=0 uniform arm
#   lam 1    uniform, PESS_SIGN=+1
#     ^ the spec says score with mu - lam*sigma. For a SUPPRESSION objective that makes an
#       uncertain completion look LESS harmful, so uncertainty relaxes the pressure. +1 is the
#       safety-conservative reading. One run settles which direction the spec wants.
set -u
cd /workspace/reward-depth/libon
source /venv/main/bin/activate
export HF_HOME=/workspace/.hf_home
export HF_TOKEN=$(grep HF_TOKEN /workspace/.env | cut -d'"' -f2)
export HF_DATASETS_DISABLE_PROGRESS_BARS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
L=/workspace/kfac

run () {  # tag lam layerw sign
  TAG=$1 LAM=$2 LAYER_W=$3 PESS_SIGN=$4 REGIME=sequential STEPS=75 \
    RUN_DIR=/workspace/libon_bayes_$1 python libon_train_bayes.py > $L/libon_bayes_$1.log 2>&1
  echo "  [$1] trained"
}

run lam0_uniform    0    uniform  -1 &
run lam1_uniform    1    uniform  -1 &
wait
run lam0.5_uniform  0.5  uniform  -1 &
run lam2_uniform    2    uniform  -1 &
wait
run lam0_evidence   0    evidence -1 &
run lam1_plussign   1    uniform  +1 &
wait
echo "SWEEP DONE"
