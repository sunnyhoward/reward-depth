#!/bin/bash
# Read-depth differential: same soft-DPO recipe, labels from probes at different layers, plus the
# ground-truth-label DPO reference. One arm at a time (single GPU). After each arm the run JSONs
# are copied into the repo and committed -- phase 5's outputs were lost because this was left to
# the end of the session, and /workspace is not a volume.
#
# Usage: bash uf/run_depth_arms.sh [arm ...]     (default: all)
set -u
cd /workspace/reward-depth
source /venv/main/bin/activate
mkdir -p /workspace/logs results/runs

preserve () {   # $1 = arm name
    cp -f /workspace/uf_softdpo_*_history.json /workspace/uf_dpo_*_history.json \
          /workspace/uf_probe_curve_lenmatch.json results/runs/ 2>/dev/null
    git add -A results/runs >/dev/null 2>&1
    git commit -q -m "results: $1 run history" >/dev/null 2>&1 \
        && echo "[preserve] committed $1" || echo "[preserve] nothing new for $1"
}

run_arm () {    # $1 = name, rest = env assignments
    local name=$1; shift
    local log=/workspace/logs/${name}.log
    if [ -f /workspace/.done_${name} ]; then echo "[skip] $name already done"; return; fi
    echo "[run] $name -> $log"
    env "$@" python "$SCRIPT" > "$log" 2>&1
    local rc=$?
    if [ $rc -ne 0 ]; then echo "[FAIL] $name rc=$rc — see $log"; tail -20 "$log"; return $rc; fi
    touch /workspace/.done_${name}
    preserve "$name"
}

ARMS=${*:-"L12 L31 GT"}
for arm in $ARMS; do
  case $arm in
    # --- read-depth arms: identical recipe, only the label layer differs ---
    L12) SCRIPT=uf/uf_soft_dpo.py run_arm L12 L_OVERRIDE=12 RUN_TAG=L12 ;;
    L31) SCRIPT=uf/uf_soft_dpo.py run_arm L31 L_OVERRIDE=31 RUN_TAG=L31 ;;
    # L16 is the accuracy max (0.799 == L12) but the most length-aligned layer
    # (corr(z,len_diff) +0.103 vs +0.006). L12-vs-L16 is therefore a near-pure depth contrast at
    # matched probe accuracy -- it separates "earlier" from "simply more accurate".
    L16) SCRIPT=uf/uf_soft_dpo.py run_arm L16 L_OVERRIDE=16 RUN_TAG=L16 ;;
    # --- ground-truth-label reference (hard labels, same config) ---
    GT)  SCRIPT=uf/uf_dpo_train.py run_arm GT RUN_TAG=GT SAVE_MERGED=0 ;;
    *)   echo "[skip] unknown arm $arm" ;;
  esac
done
echo "[done] arms: $ARMS"
