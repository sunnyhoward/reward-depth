#!/usr/bin/env bash
# Score the replay 2x2: generate false_friend + style for the four arms, then judge them.
#
# Waits for the training sweep to finish rather than being launched by hand afterwards, so the
# GPU is never idle between the last training step and the first generation.
#
# `base` is regenerated here rather than reused from results/probefix4b_famgen/. That bank came
# from another session's environment; 0813 had to discard the 0811 D-arm comparison for exactly
# that reason, and the floor is the one number every contrast is measured against.
#
# Only false_friend and style: RESULTS_0814_ELICITATION.md shows the other three families never
# elicit their own item, so generating them would cost GPU and measure nothing.
set -euo pipefail
cd "$(dirname "$0")"

set -a; . /workspace/.env; set +a
export HF_HOME=/workspace/.hf_home
PY=${PY:-/venv/main/bin/python}
export SUP_MODEL=Qwen/Qwen3.5-4B
export SUP_BRIT=/workspace/reward-depth/supervisor/britishness/dosed/brit_dose20.jsonl
ROOT=${ROOT:-/workspace/probefix4b_replay}
GEN=${GEN:-/workspace/probefix_replay_famgen}
S1=/workspace/.hf_home/hub/models--sunnyhoward--reward-depth-probefix4b/snapshots/3b55e6a54b017b522c841b8033807ae614bce53e/B4_probe_L20/ckpt300
ARMS_LIST="P1_r0_noreplay P1_r1_replay C1_r0_noreplay C1_r1_replay"

echo "waiting for the sweep..."
until [ -f "$ROOT/C1_r1_replay/DONE" ]; do
  for a in $ARMS_LIST; do
    [ -f "$ROOT/$a.log" ] && grep -q "Traceback" "$ROOT/$a.log" && { echo "!! $a crashed"; exit 1; }
  done
  sleep 30
done
echo "sweep complete at $(date '+%H:%M:%S')"

SPEC="base="
for a in $ARMS_LIST; do
  ck="$ROOT/$a/ckpt600"
  [ -d "$ck" ] || { echo "missing $ck"; exit 1; }
  case "$a" in
    C1_*) SPEC="$SPEC,$a=$S1:$ck" ;;      # two-stage: stage 1 merged first, then stage 2
    *)    SPEC="$SPEC,$a=$ck" ;;
  esac
done
echo "ARMS=$SPEC"

ARMS="$SPEC" OUT="$GEN" FAMS=false_friend,style "$PY" pf_famgen_arms.py
IN="$GEN" FAMS=false_friend,style BS=16 "$PY" pf_judge_all.py
echo REPLAYSCOREDONE
