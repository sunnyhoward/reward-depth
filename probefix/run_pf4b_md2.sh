#!/usr/bin/env bash
# THE C1 RECIPE WITH A STAGE 1 THAT ACTUALLY INSTALLS.
#
# WHY. Every two-stage arm in this project has been built on `B4_probe_L20` -- verified from the
# banked history: mode=probe, read=LAST, layer 20. A continuously-refitted probe scored at the
# completion end, which installs NOTHING at the output (the surrogate gap in probefix/HANDOVER.md:
# decodability rises everywhere including the final block while output ranking and free generation
# sit at base). So "stage 1 does not help stage 2" has never been a fair test -- there was nothing
# behavioural there to help WITH.
#
# `MD_L20_*` is now a stage 1 that does install on its own: false_friend free generation 36.0 ->
# 66.4 (+30.3, 3.7 SE), coherence 94.9-97.1, no degeneration (RESULTS_0814_MEANDIFF.md). This
# swaps it in and changes nothing else: same MODE=final stage 2, same LAMBDA=50, same write range
# 21-31, same 600 steps, same seed as C1.
#
# THREE OUTCOMES, ALL INFORMATIVE.
#   beats P1 (78.8 ff / 73.4 style)  the two-stage idea was right and was starved by its stage 1
#   matches P1                       depth-attached preparation buys nothing even when it works --
#                                    a far stronger negative than C1-vs-P1, with no excuse left
#   below meandiff alone (66.4)      stage 2 UNDOES the encoder edit, which is its own finding
#
# The s100 arm is included because meandiff's ckpt100 and ckpt600 are behaviourally equivalent
# (false_friend 65.4 vs 66.4) while ckpt600 has been driven to 7x its saturation target. If stage 2
# undoes the edit, the less over-optimised base is the one with a chance of surviving it.
set -euo pipefail
cd "$(dirname "$0")"

set -a; . /workspace/.env; set +a
export HF_HOME=/workspace/.hf_home
PY=${PY:-/venv/main/bin/python}
export SUP_MODEL=Qwen/Qwen3.5-4B
export SUP_BRIT=/workspace/reward-depth/supervisor/britishness/dosed/brit_dose20.jsonl
L=${L:-20}; TOP=31
MD=${MD:-/workspace/probefix4b_meandiff}
ROOT=${ROOT:-/workspace/probefix4b_md2}
STEPS=${STEPS:-600}
export PF_LAYER=$L SEED=${SEED:-0} EVAL_EVERY=50 CKPT_EVERY=100
mkdir -p "$ROOT"

run () { local tag="$1"; shift
  [ -f "$ROOT/$tag/DONE" ] && { echo "== $tag done, skipping"; return; }
  echo "===== $tag ====="
  env "$@" OUT="$ROOT/$tag" "$PY" pf_train.py > "$ROOT/$tag.log" 2>&1 \
    && touch "$ROOT/$tag/DONE" || { echo "!! $tag FAILED, see $ROOT/$tag.log"; return 1; }
  tail -3 "$ROOT/$tag.log"; }

for s1 in "MD_L${L}_r1_replay/ckpt600" "MD_L${L}_r0_noreplay/ckpt600" "MD_L${L}_r0_noreplay/ckpt100"; do
  [ -d "$MD/$s1" ] || { echo "missing stage-1 checkpoint $MD/$s1"; exit 1; }
done

# LAMBDA=50 here is the MODE=final DPO-Positive weight, inside the logsigmoid margin -- NOT the
# 1.0 used as an added DPOP floor in MODE=meandiff. Same scale as C1, deliberately.
run "MD2_r1_s600" MODE=final LAMBDA=50 W_REPLAY=1 \
    INIT_MERGE="$MD/MD_L${L}_r1_replay/ckpt600"   LORA_MIN=$((L + 1)) LORA_MAX=$TOP STEPS=$STEPS
run "MD2_r0_s600" MODE=final LAMBDA=50 W_REPLAY=0 \
    INIT_MERGE="$MD/MD_L${L}_r0_noreplay/ckpt600" LORA_MIN=$((L + 1)) LORA_MAX=$TOP STEPS=$STEPS
run "MD2_r0_s100" MODE=final LAMBDA=50 W_REPLAY=0 \
    INIT_MERGE="$MD/MD_L${L}_r0_noreplay/ckpt100" LORA_MIN=$((L + 1)) LORA_MAX=$TOP STEPS=$STEPS
echo MD2DONE
