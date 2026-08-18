#!/usr/bin/env bash
# NEXT_0817.md §6.0a -- does the DIFFERENCE-ONLY defect explain stage 1's signature failure?
#
# THE QUESTION. `MODE=probe` scores `probe.score(read(chosen) - read(rejected))`: a difference,
# structurally the same degree of freedom `pf_mdsides.py` caught open in the mean-diff arms
# (RESULTS_0817_MDSTACK.md §2, RESULTS_0817_MDFLOOR.md §1). Its three anti-forging provisions --
# scale-free RMS read, continuous refit, saturation at TARGET -- all bound the RESIDUAL; none pins
# either side's ABSOLUTE position. So stage 1's most-repeated result (probefix/HANDOVER.md:
# decodability rises everywhere, including the final block, and nothing installs at the output) has
# an untested explanation: if the probe's gain comes from pushing the REJECTED side down, the probe
# registers it and generation -- which depends on the chosen side -- never moves.
#
# READING. Dchosen ~ 0 with Drejected strongly negative CONFIRMS it. Both sides rising (the P3
# profile, Dchosen +138.4 at L30 in RESULTS_0817_MDFLOOR.md §1) refutes it and the dissociation
# needs a different mechanism.
#
# The stage-1 adapter does not survive the box, so it is retrained -- byte-identical to the arm
# every stage-1 claim rests on (run_pf4b.sh:36 = run_pf4b_dpop_c.sh:37): MODE=probe, LoRA 0-31,
# 300 steps, seed 0, replay ON at the default W_REPLAY=1.
set -eu
cd "$(dirname "$0")"
source /venv/main/bin/activate
export SUP_MODEL=Qwen/Qwen3.5-4B
export SUP_BRIT=/workspace/reward-depth/supervisor/britishness/dosed/brit_dose20.jsonl
L=20; TOP=31; ROOT=/workspace/probefix4b
export PF_LAYER=$L SEED=0 EVAL_EVERY=50 CKPT_EVERY=100 PROBE_WARM=2048
RES=/workspace/reward-depth/results/probefix
mkdir -p "$ROOT" "$RES"

TAG="B4_probe_L${L}"
if [ -f "$ROOT/$TAG/DONE" ]; then echo "== $TAG done, skipping"; else
  echo "===== $TAG ====="
  env MODE=probe LORA_MIN=0 LORA_MAX=$TOP STEPS=300 OUT="$ROOT/$TAG" \
    python pf_train.py > "$ROOT/$TAG.log" 2>&1 \
    && touch "$ROOT/$TAG/DONE" || { echo "!! $TAG FAILED, see $ROOT/$TAG.log"; exit 1; }
  tail -2 "$ROOT/$TAG.log"
fi

# The diagnostic, at the prescribed 128 pairs in the base-fitted frame. LAYERS adds L20 -- the
# attach point, absent from pf_mdsides.py's default set and the one read the objective sees.
# BOTH pooling modes, deliberately: READ=mean is the column every banked mdsides row uses
# (MD/MDSTACK/P3 in RESULTS_0817_MDFLOOR.md §1) so the arms stay comparable, and READ=last is what
# `MODE=probe` actually optimised (PROBE_READ defaults to `last` outside meandiff/mdstack).
# ckpt100 as well as ckpt300 so the trajectory is visible, not just the endpoint.
ARMS="B4_s100=$ROOT/$TAG/ckpt100,B4_s300=$ROOT/$TAG/ckpt300"
for R in mean last; do
  echo "===== mdsides READ=$R ====="
  env ARMS="$ARMS" LAYERS=6,12,18,20,24,30 N=128 READ=$R \
    OUT="$RES/mdsides_stage1_$R.json" python pf_mdsides.py 2>&1 | tee "$ROOT/mdsides_$R.log"
done
echo STAGE1SIDESDONE
