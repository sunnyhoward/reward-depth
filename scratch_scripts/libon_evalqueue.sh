#!/bin/bash
# Judged evals for every trained arm, concurrency 2. Waits for the training queue to finish so
# the two queues never overlap (three concurrent jobs OOM this card).
#
# ckpt25 and ckpt75 per arm: 25 is at or near the budget boundary for the aggressive regimes,
# 75 is the endpoint. ckpt50 is skipped to keep the eval phase under an hour; the in-loop
# coherence/utility trace already covers it.
set -u
cd /workspace/reward-depth/libon
export HF_HOME=/workspace/.hf_home
export HF_TOKEN=$(grep HF_TOKEN /workspace/.env | cut -d'"' -f2)
export HF_DATASETS_DISABLE_PROGRESS_BARS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

until grep -q "QUEUE DONE" /workspace/kfac/libon_queue.log 2>/dev/null; do sleep 30; done
echo "training queue finished; starting evals"

: > /tmp/libon_evaljobs.txt
for C in 25 75; do
  echo "retrained_ckpt$C /workspace/libon_retrained/ckpt$C" >> /tmp/libon_evaljobs.txt
done
for A in lam0_uniform lam1_uniform lam0.5_uniform lam2_uniform lam0_evidence lam1_plussign; do
  for C in 25 75; do
    echo "bayes_${A}_ckpt$C /workspace/libon_bayes_$A/ckpt$C" >> /tmp/libon_evaljobs.txt
  done
done

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
xargs -a /tmp/libon_evaljobs.txt -d '\n' -I{} -P 2 bash -c 'eval_one "{}"'

# IFEval only where it is diagnostic (the distribution-mismatch point is already made)
for T in retrained_ckpt75:/workspace/libon_retrained/ckpt75 \
         bayes_lam0_uniform_ckpt75:/workspace/libon_bayes_lam0_uniform/ckpt75; do
  ( source /venv/main/bin/activate
    CKPT=${T##*:} TAG=${T%%:*} N_IF=64 python libon_ifeval.py \
      > /workspace/kfac/libon_ifeval_${T%%:*}.log 2>&1 )
done

source /venv/main/bin/activate
python libon_report.py
echo "EVALQUEUE DONE"
