#!/usr/bin/env bash
# Step-matching fix: the output-loss stage at 600 steps for the arms that only got 300.
#
# WHY. The convergence check on the 300-step arms says none of them had settled: over their last
# three evals C1 moved install +0.056 / guard -0.120, C3 +0.063 / -0.180, D truth_dialect +0.200,
# and D's truth_dialect trajectory OSCILLATES (0.33 -> 0.69 -> 0.20 -> 0.16 -> 0.18 -> 0.36)
# rather than plateauing -- so "D stalls at 0.360" was reading a snapshot of a moving, unstable
# trajectory as an asymptote. A meanwhile ran 600 output-loss steps against these arms' 300, so
# the behaviour comparison in RESULTS.md 3.4 is confounded with output-step count for the pair it
# leans on hardest (A vs C2).
#
# Same seed, so the first 300 steps reproduce the existing runs exactly and steps 300-600 are the
# new information.
set -eu
cd "$(dirname "$0")"
source /venv/main/bin/activate
export SUP_BRIT="${SUP_BRIT:-/workspace/reward-depth/supervisor/britishness/dosed/brit_dose20.jsonl}"
L=12; ROOT=/workspace/probefix
export PF_LAYER=$L SEED=0 EVAL_EVERY=50 CKPT_EVERY=100 PROBE_WARM=2048
S1="$ROOT/B_probe_L${L}/ckpt300"

run () { local tag="$1"; shift
  [ -f "$ROOT/$tag/DONE" ] && { echo "== $tag done"; return; }
  echo "===== $tag ====="
  env "$@" OUT="$ROOT/$tag" python pf_train.py > "$ROOT/$tag.log" 2>&1 \
    && touch "$ROOT/$tag/DONE" || { echo "!! $tag FAILED"; return 1; }
  tail -2 "$ROOT/$tag.log"; }

run "D_dpo_upper_600" MODE=final LORA_MIN=13 LORA_MAX=23 STEPS=600
run "C1_s2_upper_600" MODE=final INIT_MERGE="$S1" LORA_MIN=13 LORA_MAX=23 STEPS=600
run "C2_s2_full_600"  MODE=final INIT_MERGE="$S1" LORA_MIN=0  LORA_MAX=23 STEPS=600
echo ALL600DONE
