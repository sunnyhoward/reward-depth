#!/bin/bash
# Second eval pass: the four extra arms (section-3 attenuation control + three depth bands), plus
# IFEval on the Bayesian arms that the first pass skipped. Waits for both the first eval queue and
# the extra-arm queue so the card never holds more than two heavy jobs.
set -u
cd /workspace/reward-depth/libon
export HF_HOME=/workspace/.hf_home
export HF_TOKEN=$(grep HF_TOKEN /workspace/.env | cut -d'"' -f2)
export HF_DATASETS_DISABLE_PROGRESS_BARS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

until grep -q "EVALQUEUE DONE" /workspace/kfac/libon_evalqueue.log 2>/dev/null; do sleep 30; done
until grep -q "EXTRA DONE" /workspace/kfac/libon_extra.log 2>/dev/null; do sleep 30; done
echo "both queues finished; starting second eval pass"

: > /tmp/libon_evaljobs2.txt
for A in atten0.31 band_early band_mid band_late; do
  for C in 25 75; do
    echo "bayes_${A}_ckpt$C /workspace/libon_bayes_$A/ckpt$C" >> /tmp/libon_evaljobs2.txt
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
xargs -a /tmp/libon_evaljobs2.txt -d '\n' -I{} -P 2 bash -c 'eval_one "{}"'

# IFEval on the Bayesian arms the first pass skipped — needed to state whether the
# distribution-mismatch result (their budget misses the damage) holds for these arms too.
for A in lam0_uniform lam1_uniform lam2_uniform atten0.31; do
  for C in 75; do
    ( source /venv/main/bin/activate
      CKPT=/workspace/libon_bayes_$A/ckpt$C TAG=bayes_${A}_ckpt$C N_IF=64 python libon_ifeval.py \
        > /workspace/kfac/libon_ifeval_${A}_$C.log 2>&1 )
  done
done

source /venv/main/bin/activate
python libon_report.py
echo "EVALQUEUE2 DONE"
