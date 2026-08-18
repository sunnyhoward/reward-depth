#!/usr/bin/env bash
# The discriminating arm for RESULTS_0818_STAGE1_SIDES.md §4: is the CONTINUOUS REFIT what lets
# stage 1 answer the objective in a direction of its own making?
#
# §4 measured the trained probe at cos 0.43 to the base preference axis, against 0.72 for the same
# logistic recipe fitted with no training -- it separates the BASE model worse than that control
# (9.2 vs 15.3) and the TRAINED model 4x better (36.7), while separation along the base axis does
# not move (21.09 -> 20.87). Reading: the refitted probe follows the model into a constructed
# subspace, so the probe registers a gain that the readout never sees.
#
# THE TEST NEEDS NO CODE. `PROBE_STEPS=0` makes `probe.refit(PROBE_STEPS)` (pf_train.py:576) a
# no-op, so the direction is whatever the 600-step warm start fitted on the BASE model and never
# moves again. Everything else -- warm start, sigma0, hinge, saturation, LoRA range, replay, seed --
# is identical to B4_probe_L20. If the direction is what matters, this arm cannot chase a new one:
# it must either move the base axis (sep(u) rises, install follows) or fail to satisfy its objective
# at all (`sat` stays low, z stays near 0). Both outcomes are informative; a third -- objective
# satisfied, sep(u) flat, install flat -- would refute §4's reading and put the failure somewhere
# else entirely.
#
# NOTE the direction of the prior: MODE=meandiff, the ONE activation objective in this project that
# installs (+21.8 over base, RESULTS_0817_MEANDIFF_REJUDGE.md), is also the one whose direction is
# not freely refitted -- MD_LAG=1 optimises against the PREVIOUS step's mean difference. This arm
# asks whether that is the operative difference or a coincidence.
set -eu
cd "$(dirname "$0")"
source /venv/main/bin/activate
export SUP_MODEL=Qwen/Qwen3.5-4B
export SUP_BRIT=/workspace/reward-depth/supervisor/britishness/dosed/brit_dose20.jsonl
L=20; TOP=31; ROOT=/workspace/probefix4b
export PF_LAYER=$L SEED=0 EVAL_EVERY=50 CKPT_EVERY=100 PROBE_WARM=2048
RES=/workspace/reward-depth/results/probefix
mkdir -p "$ROOT" "$RES"

TAG="B4_probefroz_L${L}"
if [ -f "$ROOT/$TAG/DONE" ]; then echo "== $TAG done, skipping"; else
  echo "===== $TAG ====="
  env MODE=probe PROBE_STEPS=0 LORA_MIN=0 LORA_MAX=$TOP STEPS=300 OUT="$ROOT/$TAG" \
    python pf_train.py > "$ROOT/$TAG.log" 2>&1 \
    && touch "$ROOT/$TAG/DONE" || { echo "!! $TAG FAILED, see $ROOT/$TAG.log"; exit 1; }
  tail -2 "$ROOT/$TAG.log"
fi

ARMS="FROZ_s100=$ROOT/$TAG/ckpt100,FROZ_s300=$ROOT/$TAG/ckpt300"
for R in mean last; do
  echo "===== mdsides READ=$R ====="
  env ARMS="$ARMS" LAYERS=6,12,18,20,24,30 N=128 READ=$R \
    OUT="$RES/mdsides_probefroz_$R.json" python pf_mdsides.py 2>&1 | tee "$ROOT/mdsides_froz_$R.log"
done

echo "===== rot ====="
env CKPTS="$ROOT/$TAG/ckpt100,$ROOT/$TAG/ckpt300" LAYER=$L N=128 CTRL_N=1024 READ=last \
  python pf_stage1_rot.py 2>&1 | tee "$ROOT/rot_froz.log"
echo PROBEFROZDONE
