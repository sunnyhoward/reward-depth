#!/usr/bin/env bash
# BOOSTED stage 1: K probes in cascade -- probe k fit on the pairs probes <k fail to separate.
#
# WHY THIS AND NOT THE PARALLEL K=8 ARM (which was a null, 0813). Eight probes hinged to one
# shared label are eight rotations of the same easy direction: the objective was satisfied at
# init and the network was never pushed (frac_sat .98 by step 25). The preference's real
# structure is FUNCTIONAL, by family -- per-family probe directions at L20 are mutually
# near-orthogonal (|cos| < .25) and the lexicon probe scores .23 on truth_dialect, BELOW chance.
# Weighting each probe by residual difficulty should recover that structure without labels.
#
# FALSIFIABLE PREDICTION: the cascade's later probes should concentrate on truth_dialect and
# false_friend -- the two families with the lowest own-probe accuracy (.48-.58) and the ones
# other families' probes actively get wrong. If probes 1..K-1 spread evenly over families, the
# cascade found nothing the parallel version didn't.
set -eu
cd "$(dirname "$0")"
source /venv/main/bin/activate
export SUP_MODEL=Qwen/Qwen3.5-4B
export SUP_BRIT=/workspace/rd-branch/supervisor/britishness/dosed/brit_dose20.jsonl
L=20; TOP=31; ROOT=/workspace/probefix4b_boost
export PF_LAYER=$L SEED=0 EVAL_EVERY=50 CKPT_EVERY=100 PROBE_WARM=2048
mkdir -p "$ROOT"

run () { local tag="$1"; shift
  [ -f "$ROOT/$tag/DONE" ] && { echo "== $tag done, skipping"; return; }
  echo "===== $tag ====="
  env "$@" OUT="$ROOT/$tag" python pf_train.py > "$ROOT/$tag.log" 2>&1 \
    && touch "$ROOT/$tag/DONE" || { echo "!! $tag FAILED, see $ROOT/$tag.log"; return 1; }
  tail -2 "$ROOT/$tag.log"; }

run "B6_boost4_L${L}" MODE=probe K_PROBES=4 BOOST=1 LORA_MIN=0 LORA_MAX=$TOP STEPS=300
S1="$ROOT/B6_boost4_L${L}/ckpt300"
run "C4_two_plain_boost" MODE=final INIT_MERGE="$S1" LAMBDA=0 LORA_MIN=$((L+1)) LORA_MAX=$TOP STEPS=600

for cell in "C4_boost=$ROOT/C4_two_plain_boost/ckpt600"; do
  tag="${cell%%=*}"; ckpt="${cell#*=}"
  echo "===== score $tag ====="
  CKPT="$ckpt" TAG="$tag" S1_MERGE="$S1" SUP_LAYER=$L python ../supervisor/sup_eval.py 2>&1 \
      | grep -E "behaviour|ranking ALL" || true
  cp "/workspace/sup/eval_$tag.json" "$ROOT/behaviour_$tag.json" 2>/dev/null || true
done
echo BOOSTDONE
