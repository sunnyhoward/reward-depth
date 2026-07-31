#!/bin/bash
# styc_prl queue (2026-07-31): waits for the scaling chain (14B stage A) and the pooled 3B
# cache, then runs the three probe-RL arms serialized. No set -e (tokenizer exit-134 trap);
# success = output file exists. Histories banked+committed after EACH arm.
cd /workspace/reward-depth || exit 1
source /venv/main/bin/activate
# gate only on the pooled cache the arms actually need; 14B caching co-runs fine (inference-
# only, short texts) and the 3B arms + 14B cache fit a 96GB card together
echo "[queue] waiting for pooled 3B cache..."
until [ -f /workspace/styc_stageA_3B_mean.json ]; do sleep 30; done
echo "[queue] prerequisites ready; starting arms"
for M in shaped seqrl pooled_margin; do
  echo "[queue] === arm $M ==="
  MODE=$M python styc_probe_rl.py > /workspace/styc_prl_${M}.log 2>&1
  if [ -f /workspace/styc_prl_${M}_history.json ]; then
    cp /workspace/styc_prl_${M}_history.json results/
    git add results/styc_prl_${M}_history.json
    git commit -q -m "styc_prl arm ${M}: pooled frozen-read probe RL history"
    echo "[queue] arm $M banked"
  else
    echo "[queue] arm $M FAILED (no history); see /workspace/styc_prl_${M}.log"
  fi
done
echo ALLDONE
