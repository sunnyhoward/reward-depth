#!/bin/bash
# Serialized job queue, concurrency 2. Each training run peaks at 30-37 GiB on this 98 GiB card,
# so two fit and three do not — an earlier attempt ran three and lost two of them to OOM.
# xargs -P 2 enforces the cap for the whole queue instead of per-pair, so a slow arm cannot
# overlap with the next pair.
set -u
cd /workspace/reward-depth/libon
export HF_HOME=/workspace/.hf_home
export HF_TOKEN=$(grep HF_TOKEN /workspace/.env | cut -d'"' -f2)
export HF_DATASETS_DISABLE_PROGRESS_BARS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export KL_MICRO=2
L=/workspace/kfac

cat > /tmp/libon_jobs.txt <<'EOF'
REGIME=retrained STEPS=75 RUN_DIR=/workspace/libon_retrained SCRIPT=libon_train.py LOG=libon_train_retrained
TAG=lam0_uniform LAM=0 LAYER_W=uniform PESS_SIGN=-1 LOG=libon_bayes_lam0_uniform
TAG=lam1_uniform LAM=1 LAYER_W=uniform PESS_SIGN=-1 LOG=libon_bayes_lam1_uniform
TAG=lam0.5_uniform LAM=0.5 LAYER_W=uniform PESS_SIGN=-1 LOG=libon_bayes_lam0.5_uniform
TAG=lam2_uniform LAM=2 LAYER_W=uniform PESS_SIGN=-1 LOG=libon_bayes_lam2_uniform
TAG=lam0_evidence LAM=0 LAYER_W=evidence PESS_SIGN=-1 LOG=libon_bayes_lam0_evidence
TAG=lam1_plussign LAM=1 LAYER_W=uniform PESS_SIGN=+1 LOG=libon_bayes_lam1_plussign
EOF

run_one () {
  set -u
  cd /workspace/reward-depth/libon
  source /venv/main/bin/activate
  eval "export $1"
  SCRIPT=${SCRIPT:-libon_train_bayes.py}
  : ${RUN_DIR:=/workspace/libon_bayes_$TAG}
  export RUN_DIR
  [ "$SCRIPT" = "libon_train_bayes.py" ] && export REGIME=${REGIME:-sequential} STEPS=${STEPS:-75}
  python $SCRIPT > /workspace/kfac/$LOG.log 2>&1
  echo "  [$LOG] exit $?"
}
export -f run_one

xargs -a /tmp/libon_jobs.txt -d '\n' -I{} -P 2 bash -c 'run_one "{}"'
echo "QUEUE DONE"
