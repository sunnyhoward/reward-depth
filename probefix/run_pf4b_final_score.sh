#!/usr/bin/env bash
# ONE judge invocation over EVERY arm of 2026-08-14.
#
# The day produced four sweeps -- the replay 2x2, meandiff, md2 (stage 2 on meandiff) and the
# pooling isolation -- and the contrasts that matter run ACROSS them: PR_mean vs MD_L20 isolates
# the objective, PR_last vs PR_mean isolates pooling, MD2 vs P1 vs meandiff-alone decides the
# two-stage question. pf_judge_all.py shuffles blind across arms WITHIN one invocation, so
# splitting these over several judge runs would put the key contrasts in different batches for no
# reason. The generations are all banked already; only the judging is redone.
#
# The partial md2-only judge run was stopped at 160/816 rather than completed, because finishing
# it and then judging the pool arms separately would have cost more GPU than one clean pass and
# left the pooling contrast spanning two invocations.
#
# Pool arms are scored at ckpt300 AND ckpt600: B4's original protocol was 300 steps, so ckpt300 is
# the like-for-like comparison against the old stage 1 and ckpt600 against MD_L20.
set -euo pipefail
cd "$(dirname "$0")"

set -a; . /workspace/.env; set +a
export HF_HOME=/workspace/.hf_home
PY=${PY:-/venv/main/bin/python}
export SUP_MODEL=Qwen/Qwen3.5-4B
export SUP_BRIT=/workspace/reward-depth/supervisor/britishness/dosed/brit_dose20.jsonl
POOL=${POOL:-/workspace/probefix4b_pool}
PREV=${PREV:-/workspace/probefix_md2_famgen}     # already holds base + replay 2x2 + meandiff + md2
GEN=${GEN:-/workspace/probefix_final_famgen}
L=${L:-20}

mkdir -p "$GEN"
cp "$PREV"/famgen_*.json "$GEN"/ 2>/dev/null || true

SPEC=""
for a in "PR_mean_L${L}" "PR_last_L${L}"; do
  for c in 300 600; do
    ck="$POOL/$a/ckpt$c"
    [ -d "$ck" ] || { echo "missing $ck"; exit 1; }
    SPEC="${SPEC:+$SPEC,}${a}_s${c}=$ck"
  done
done
echo "generating: $SPEC"
ARMS="$SPEC" OUT="$GEN" FAMS=false_friend,style "$PY" pf_famgen_arms.py

echo "arms in the judge run: $(ls "$GEN"/famgen_*.json | wc -l)"
IN="$GEN" FAMS=false_friend,style BS=16 "$PY" pf_judge_all.py
echo FINALSCOREDONE
