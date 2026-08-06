#!/bin/bash
# Extra arms, CONCURRENCY 1 — these run alongside the eval queue (2 slots), and three heavy jobs
# OOM this card. 75 steps each, single seed, everything else at the reproduction's values.
#
#   atten0.31   lambda=0 with the probe loss scaled by a constant 0.31.
#               THE CONTROL FOR SECTION 3. lambda with -sigma turned out to be a dial on effective
#               loss magnitude (+sigma, which amplifies, collapsed fastest of all arms). 0.31 is
#               the measured ratio of lambda=1's mean probe loss to lambda=0's over the first 25
#               steps, so this attenuates by the same average amount with NO per-example
#               structure. If it reproduces lambda=1's survival curve, "pessimism" is a
#               reparameterised step size; if it does not, the per-example sigma is doing work.
#
#   band_early/mid/late   lambda=1, supervising only {0,6} / {12,18} / {24,30}.
#               The causal counterpart to the per-layer evidence numbers: ELBO says mid-stack
#               carries the harmfulness evidence (L12/L18 ~.29 weight, L0 ~.02), but every arm so
#               far supervises all six at once, so that is a statement about readability, not
#               about which probe does the suppression. Compare against lam1_uniform (all six).
set -u
cd /workspace/reward-depth/libon
source /venv/main/bin/activate
export HF_HOME=/workspace/.hf_home
export HF_TOKEN=$(grep HF_TOKEN /workspace/.env | cut -d'"' -f2)
export HF_DATASETS_DISABLE_PROGRESS_BARS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export KL_MICRO=2
L=/workspace/kfac

# wait for the last training arm so we never exceed two heavy jobs
until grep -q "QUEUE DONE" $L/libon_queue.log 2>/dev/null; do sleep 20; done

echo "=== atten0.31 (uniform-attenuation control for section 3) ==="
TAG=atten0.31 LAM=0 PROBE_SCALE=0.31 REGIME=sequential STEPS=75 \
  RUN_DIR=/workspace/libon_bayes_atten0.31 python libon_train_bayes.py \
  > $L/libon_bayes_atten0.31.log 2>&1
echo "  atten0.31 done"

for B in "early 0,6" "mid 12,18" "late 24,30"; do
  set -- $B
  echo "=== band_$1 (layers $2) ==="
  TAG=band_$1 LAM=1 LIBON_LAYERS=$2 REGIME=sequential STEPS=75 \
    RUN_DIR=/workspace/libon_bayes_band_$1 python libon_train_bayes.py \
    > $L/libon_bayes_band_$1.log 2>&1
  echo "  band_$1 done"
done
echo "EXTRA DONE"
