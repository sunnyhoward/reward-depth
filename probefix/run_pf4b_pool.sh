#!/usr/bin/env bash
# ISOLATE THE POOLING. Same refitted-probe objective, read at the completion end vs pooled.
#
# MD_L20 changed THREE things at once against the old stage 1 -- pooled reads, a lag-1
# mean-difference direction instead of a continuously-refitted probe, and a saturating hinge -- and
# it installed where the old stage 1 installs nothing. That is one arm and three candidate causes.
#
# This splits them. Holding the objective fixed (MODE=probe, the continuously-refitted probe with
# its own saturating hinge at TARGET) and moving ONLY the read:
#
#   PR_last_L20   refitted probe, completion-end read   <- B4's exact configuration, retrained here
#   PR_mean_L20   refitted probe, POOLED read           <- the new cell
#   MD_L20_r1     lag-1 mean-diff,  pooled read         <- already run
#
#   PR_last vs PR_mean   isolates POOLING
#   PR_mean vs MD_L20    isolates the OBJECTIVE (refitted probe vs lag-1 mean difference)
#
# phase-8 §13 attributes the closure of the forging channel specifically to pooling -- "with the
# target being the mean of every emission state, there is no causally-dead single state to cheaply
# rewrite" -- so the prediction is that PR_mean recovers most of the gap. If instead PR_mean sits
# with PR_last, the pooling is not the active ingredient and the lag-1 direction is.
#
# B4 IS RETRAINED RATHER THAN REUSED. The banked B4 is from another session's environment, and
# 0814 already showed banked arms reproduce to within ~3 points -- close, but this contrast is
# the whole point of the run, so it is held within one environment.
#
# Replay ON in both, matching every other probefix arm; the 0814 replay 2x2 showed the term makes
# little difference either way and this run is not about it.
#
# 600 steps with CKPT_EVERY=100: B4's original protocol was 300 steps, so ckpt300 is the
# like-for-like B4 comparison and ckpt600 is the like-for-like MD_L20 comparison.
#
# RUNS IN PARALLEL with the md2 sweep -- that one holds ~32 GiB of 96, so two fit. NEXT_0810 §4 is
# the standing warning here: other sessions' runs have held 55-73 GiB and OOMed jobs on this box.
set -euo pipefail
cd "$(dirname "$0")"

set -a; . /workspace/.env; set +a
export HF_HOME=/workspace/.hf_home
PY=${PY:-/venv/main/bin/python}
export SUP_MODEL=Qwen/Qwen3.5-4B
export SUP_BRIT=/workspace/reward-depth/supervisor/britishness/dosed/brit_dose20.jsonl
L=${L:-20}; TOP=31; ROOT=${ROOT:-/workspace/probefix4b_pool}
STEPS=${STEPS:-600}
export PF_LAYER=$L SEED=${SEED:-0} EVAL_EVERY=50 CKPT_EVERY=100 PROBE_WARM=2048
mkdir -p "$ROOT"

nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader || true

run () { local tag="$1"; shift
  [ -f "$ROOT/$tag/DONE" ] && { echo "== $tag done, skipping"; return; }
  echo "===== $tag ====="
  env "$@" OUT="$ROOT/$tag" "$PY" pf_train.py > "$ROOT/$tag.log" 2>&1 \
    && touch "$ROOT/$tag/DONE" || { echo "!! $tag FAILED, see $ROOT/$tag.log"; return 1; }
  tail -3 "$ROOT/$tag.log"; }

# The new cell first: if only one arm finishes, the pooled one is the one worth having.
run "PR_mean_L${L}" MODE=probe PROBE_READ=mean W_REPLAY=1 \
                    LORA_MIN=0 LORA_MAX=$TOP STEPS=$STEPS
run "PR_last_L${L}" MODE=probe PROBE_READ=last W_REPLAY=1 \
                    LORA_MIN=0 LORA_MAX=$TOP STEPS=$STEPS
echo POOLDONE
