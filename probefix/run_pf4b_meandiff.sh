#!/usr/bin/env bash
# THE PHASE-8 MECHANISM, ON BRITISHNESS. First test of `pooled_margin` outside styc and UF.
#
# WHY THIS ARM EXISTS. results_phase8.md:266 is the only place in this project where optimising
# activations DIRECTLY installed a preference: conflict .845, gen_correct .891, no forging, stable
# plateau from ~100 steps -- "cleanest install in the project". It was ported to UltraFeedback in
# phase 9 and came back a behavioural null, and phase 9 section 3 diagnosed why: "the margin
# mechanism works exactly when the direction is causally load-bearing; pooling fixed forging, not
# direction quality." On styc the pooled (ce-we) direction IS the correctness feature; on UF the
# pooled direction at L* is the style-legible one and driving it does nothing.
#
# Britishness sits between the two: a real generative task, but with a crisp rule that is
# near-ceiling decodable (0.92 at 4B). If the mechanism works anywhere naturalistic it should work
# here, and if it does not, phase 9's diagnostic says where to look.
#
# WHAT probefix HAS BEEN RUNNING INSTEAD. `MODE=probe` is a continuously-refitted linear probe
# scored at the COMPLETION END (`PROBE_READ=last`, which every probefix arm has used), with the
# gradient through a scalar probe score. That is the objective family phases 1-5 established
# forges, and probefix reproduced the signature exactly -- decodability rises everywhere including
# the final block while output ranking and free generation stay at base (the surrogate gap in
# probefix/HANDOVER.md). The mechanism that WORKED has never been run on this task.
#
# THREE DIFFERENCES, each load-bearing per phase 8:
#   pooled reads (PROBE_READ=mean)  no causally-dead single state to rewrite
#   lag-1 direction (MD_LAG=1)      optimise the PREVIOUS step's direction, not this step's
#   saturating hinge (M0)           relu(M0 - proj); unbounded objectives are met by scaling
#
# M0 IS CALIBRATED, NOT COPIED -- see pf_train.py. styc and UF both used M0=4.0 against a natural
# projection of ~3.5 (1.14x). Britishness pairs are MINIMAL, so their difference vectors are far
# more collinear and the base projection is ~8.1; M0=4.0 would be satisfied at initialisation
# (measured: frac_sat 1.00 at step 10). M0_MULT=1.15 reproduces the ratio.
#
# CHECKPOINT EVERY 100 IS MANDATORY, not hygiene: phase 8's shaped arm peaked at step ~125 and
# then Goodharted to gen_correct .16 by 200 while every ranking metric kept improving, and only
# the final over-optimised adapter had been saved.
set -euo pipefail
cd "$(dirname "$0")"

set -a; . /workspace/.env; set +a
export HF_HOME=/workspace/.hf_home
PY=${PY:-/venv/main/bin/python}
export SUP_MODEL=Qwen/Qwen3.5-4B
export SUP_BRIT=/workspace/reward-depth/supervisor/britishness/dosed/brit_dose20.jsonl
L=${L:-20}; TOP=31; ROOT=${ROOT:-/workspace/probefix4b_meandiff}
STEPS=${STEPS:-600}
export PF_LAYER=$L SEED=${SEED:-0} EVAL_EVERY=50 CKPT_EVERY=100
mkdir -p "$ROOT"
[ -f /workspace/sup/replay_bank.pt ] || { echo "no replay bank -- run pf_make_bank.py first"; exit 1; }

run () { local tag="$1"; shift
  [ -f "$ROOT/$tag/DONE" ] && { echo "== $tag done, skipping"; return; }
  echo "===== $tag ====="
  env "$@" OUT="$ROOT/$tag" "$PY" pf_train.py > "$ROOT/$tag.log" 2>&1 \
    && touch "$ROOT/$tag/DONE" || { echo "!! $tag FAILED, see $ROOT/$tag.log"; return 1; }
  tail -3 "$ROOT/$tag.log"; }

# LoRA spans the whole stack in both arms; the preference gradient reaches only 0..L because the
# read is at block L, which is MODE=probe's asymmetry and styc's setup (full LoRA, read at LI).
# LAMBDA is the DPOP floor weight, added to the loss as in styc (DPOP=1.0) -- NOT the 50 used in
# the MODE=final arms, where it sits inside the logsigmoid margin and is a different scale.
#
# Both replay settings: 0814 found replay slightly costs the install in 4/4 contrasts, but every
# other probefix arm carries it, so the comparable arm and the best-shot arm are different runs.
run "MD_L${L}_r1_replay"   MODE=meandiff LAMBDA=1.0 W_REPLAY=1 \
                           LORA_MIN=0 LORA_MAX=$TOP STEPS=$STEPS
run "MD_L${L}_r0_noreplay" MODE=meandiff LAMBDA=1.0 W_REPLAY=0 \
                           LORA_MIN=0 LORA_MAX=$TOP STEPS=$STEPS
echo MEANDIFFDONE
