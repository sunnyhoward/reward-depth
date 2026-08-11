#!/usr/bin/env bash
# Guard-dose x attach-depth on britishness (2026-08-11).
#
# THE PREDICTION UNDER TEST. Plain britishness is an L0 phenomenon -- sub-token orthography,
# decodable at ceiling from the embedding layer -- and eagle/RESULTS.md §2 found stage-1 at L4
# installs it cheaply (.64 terse-rate at KL 0.13) while L12 shows the surrogate gap (.16, "head
# satisfied, top not listening") and deep attach misfires. Today's guard-dose sweep
# (decodability/brit_guard_dose.py) shows that adding guard pairs moves the composite rule from
# L0 to 0.44-0.67 fractional depth, because resolving truth-vs-dialect needs the truth axis and
# that is mid-stack. So: the attach depth that installs should MOVE UP with the dose.
#
# The informative failure is L4 installing the 20% arm just as well as the 4% arm -- that would
# say attach depth does not track where the rule lives, and the surrogate-gap story is wrong.
#
# WHAT IS HELD FIXED across all six arms: the split (brit_dose_data.py writes one split into every
# dose file), steps, lr, beta, pairs/step, replay, seed. Only SUP_LAYER and the dose file vary.
#
# K-FAC IS OFF and W_KFAC=0, following results_0809 §5a (1:3:1 and 1:0:1 indistinguishable) and
# the 0810 UF arms, which made the same call. That is a documented control, not a shortcut.
#
# THE HEAD-FIDELITY CONFOUND IS REAL AND IS NOT REMOVED BY THIS SCRIPT. Equal distillation budget
# is not equal convergence -- a deep head fits an easier problem -- so the arms carry a covariate
# that rises with depth by construction (measured here: L4/L12/L20 -> see head_L*.json).
# results_uf_0810 §5c is the reason to run it anyway: improving a shallow head's fidelity
# 0.404 -> 0.562 moved its install by 0.010. If the shallow arm loses at high dose, the GRAFT arm
# below (no head at all) is the control that separates "weak head" from "wrong depth".
#
# Usage: bash supervisor/run_brit_dose_depth.sh [layers] [doses]
set -u
cd "$(dirname "$0")"
source /venv/main/bin/activate

LAYERS="${1:-4 12 20}"
DOSES="${2:-4 20}"
STEPS="${STEPS:-300}"
SEED="${SEED:-0}"
ROOT="${ROOT:-/workspace/brit_dose}"
mkdir -p "$ROOT"

for L in $LAYERS; do
  if [ ! -f "/workspace/sup/head_tf_L${L}.pt" ]; then
    echo "!! missing head_tf_L${L}.pt -- run SUP_LAYER=$L SKIP_KFAC=1 HEAD_STEPS=5000 python sup_prepare.py"
    exit 1
  fi
done

for D in $DOSES; do
  for L in $LAYERS; do
    TAG="L${L}_d${D}"
    DIR="$ROOT/$TAG"
    # Never silently reuse: sup_prepare.py:152 already cost this experiment one unfair arm by
    # skipping on file-exists, so an existing run directory is announced, not overwritten.
    if [ -d "$DIR/ckpt${STEPS}" ]; then
      echo "== $TAG already complete, skipping (delete $DIR to redo)"
      continue
    fi
    echo "===== $TAG : attach L$L, guard ${D}% ====="
    STAGE=1 \
    SUP_LAYER="$L" \
    SUP_BRIT="$PWD/britishness/dosed/brit_dose${D}.jsonl" \
    RUN_TAG_DIR="$DIR" \
    STEPS="$STEPS" SEED="$SEED" \
    W_PREF=1 W_KFAC=0 W_REPLAY=1 \
    EVAL_EVERY=50 CKPT_EVERY=100 \
      python sup_train.py 2>&1 | tail -40
  done
done

echo "===== scoring every checkpoint, per family ====="
for D in $DOSES; do
  for L in $LAYERS; do
    DIR="$ROOT/L${L}_d${D}"
    [ -d "$DIR" ] || continue
    CK=$(ls -d "$DIR"/ckpt* 2>/dev/null | sort -V)
    [ -n "$CK" ] || continue
    SUP_BRIT="$PWD/britishness/dosed/brit_dose${D}.jsonl" \
    SUP_LAYER="$L" \
    OUT_JSON="$ROOT/fam_eval_L${L}_d${D}.json" \
      python sup_eval_brit_families.py $CK 2>&1 | grep -vE "^Loading|it/s\]$"
  done
done
echo ALLARMSDONE
