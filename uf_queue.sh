#!/bin/bash
# UF experiment queue, 2026-07-30 afternoon (user-approved plan).
# Sequential 300-step arms, all uf_hybrid_md.py, shared Stage-A cache; bank + commit after each.
#   A margin300    mean-diff margin ONLY (EMIT=none, DPOP anchor on) -- "does the pure activation
#                  push move an 8B on UF"; forging detector armed
#   B sd_upper300  soft-DPO, writes > L12 only -- J-only-style control
#   C hyb2_300     margin <=L12 + soft-DPO >L12 co-trained -- the faithful stage-2.5 port
#   D twostage300  soft-DPO >L12 ON TOP of A's frozen low edit -- the legibility experiment
#                  (step-0 eval shows whether A's edit already moved emitted preferences)
#   E sd_lower300  soft-DPO, writes <=L12 only -- the write-depth Occam cell
set -u
source /venv/main/bin/activate
. /workspace/.env 2>/dev/null; export HF_TOKEN
cd /workspace/reward-depth
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export RL_STEPS=300

bank () {  # $1 = history file basename
  cp -f /workspace/$1 results/ 2>/dev/null
  git add results/$1 >/dev/null 2>&1
  git commit -q -m "results: $1 (uf queue)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>" 2>/dev/null || true
}
run () {  # $1 tag; rest env pairs
  local tag=$1; shift
  [ -f /workspace/uf_hybrid_md_${tag}_history.json ] && { echo "[skip] $tag"; return 0; }
  echo "[queue $(date +%H:%M)] $tag"
  env "$@" RUN_TAG=$tag python uf/uf_hybrid_md.py > /workspace/logs_${tag}.log 2>&1
  local rc=$?
  echo "[queue $(date +%H:%M)] $tag exit $rc"
  bank uf_hybrid_md_${tag}_history.json
  return $rc
}

run margin300   EMIT=none    MCOEF=1                              || exit 1
run sd_upper300 EMIT=softdpo MCOEF=0 JONLY_UPPER=1                || exit 1
run hyb2_300    EMIT=softdpo MCOEF=1                              || exit 1
run twostage300 EMIT=softdpo MCOEF=0 JONLY_UPPER=1 LOAD_LORA=/workspace/uf_hybrid_md_margin300_lora || exit 1
run sd_lower300 EMIT=softdpo MCOEF=0 JONLY_LOW=1                  || exit 1
echo "[queue] ALL DONE"; touch /workspace/.uf_queue_done
