#!/bin/bash
# cc 300-step Goodhart race: jonly lower vs upper vs full write windows (follow-up to the
# 100-step result: lower installs 96% but transfers worse -- does the 300-step over-optimization
# trajectory separate the windows further, and in which direction?). Runs alongside the UF queue
# (Qwen-3B, small footprint).
set -u
source /venv/main/bin/activate
. /workspace/.env 2>/dev/null; export HF_TOKEN
cd /workspace/reward-depth
run () {
  local tag=$1; shift
  [ -f /workspace/cc_stage2_${tag}_history.json ] && { echo "[skip] $tag"; return 0; }
  echo "[race $(date +%H:%M)] $tag"
  env "$@" ARM=hybrid MCOEF=0 EWC_KL=1.0 STEPS=300 RUN_TAG=$tag python cc_stage2.py \
    > /workspace/logs_cc_${tag}.log 2>&1
  echo "[race $(date +%H:%M)] $tag exit $?"
  cp -f /workspace/cc_stage2_${tag}_history.json results/ 2>/dev/null
  git add results/cc_stage2_${tag}_history.json >/dev/null 2>&1
  git commit -q -m "results: cc 300-step jonly write-depth race cell ${tag}

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>" 2>/dev/null || true
}
run jonly_lower300 JONLY_LOW=1
run jonly_upper300
run jonly_full300  JONLY_FULL=1
echo "[race] ALL DONE"; touch /workspace/.cc_race_done
