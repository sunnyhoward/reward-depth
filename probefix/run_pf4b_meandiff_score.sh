#!/usr/bin/env bash
# Score the meandiff arms IN THE SAME JUDGE RUN as the replay 2x2 and base.
#
# The comparison that matters is meandiff vs plain DPO (P1) vs the two-stage recipe (C1), and
# `pf_judge_all.py` shuffles items blind ACROSS arms within a family. Judging the new arms on
# their own would put them in a different judge invocation from their comparators, which is a
# free way to introduce a batch effect into the one contrast the run exists to make. So the
# already-generated famgen files are copied in and everything is judged together.
set -euo pipefail
cd "$(dirname "$0")"

set -a; . /workspace/.env; set +a
export HF_HOME=/workspace/.hf_home
PY=${PY:-/venv/main/bin/python}
export SUP_MODEL=Qwen/Qwen3.5-4B
export SUP_BRIT=/workspace/reward-depth/supervisor/britishness/dosed/brit_dose20.jsonl
ROOT=${ROOT:-/workspace/probefix4b_meandiff}
PREV=${PREV:-/workspace/probefix_replay_famgen}
GEN=${GEN:-/workspace/probefix_all_famgen}
L=${L:-20}
ARMS_LIST="MD_L${L}_r1_replay MD_L${L}_r0_noreplay"

echo "waiting for the meandiff sweep..."
until [ -f "$ROOT/MD_L${L}_r0_noreplay/DONE" ]; do
  for a in $ARMS_LIST; do
    [ -f "$ROOT/$a.log" ] && grep -q "Traceback" "$ROOT/$a.log" && { echo "!! $a crashed"; exit 1; }
  done
  sleep 30
done
echo "sweep complete at $(date '+%H:%M:%S')"

mkdir -p "$GEN"
cp "$PREV"/famgen_*.json "$GEN"/ 2>/dev/null || true   # base + the replay 2x2, already generated

SPEC=""
for a in $ARMS_LIST; do
  ck="$ROOT/$a/ckpt600"
  [ -d "$ck" ] || { echo "missing $ck"; exit 1; }
  SPEC="${SPEC:+$SPEC,}$a=$ck"
done
echo "ARMS=$SPEC"

ARMS="$SPEC" OUT="$GEN" FAMS=false_friend,style "$PY" pf_famgen_arms.py
IN="$GEN" FAMS=false_friend,style BS=16 "$PY" pf_judge_all.py
echo MEANDIFFSCOREDONE
