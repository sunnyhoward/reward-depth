#!/usr/bin/env bash
# Follow-up arms: is the SATURATION the limiting factor in stage 1?
#
# Arm B's own training statistics say it barely trains: mean `sat` (fraction of the batch already
# above the hinge target) 0.876, and 47.6% of steps carry ZERO preference gradient because every
# pair in the batch is already separated. That is the design working as intended -- a minimum-change
# edit -- but it also means any weakness in B is confounded with undertraining, and NEXT_0810 §4 is
# explicit that "a null from an undertrained arm looks exactly like a finding".
#
# So: the same stage 1 with an UNBOUNDED Bradley-Terry objective (never saturates, every pair keeps
# pushing), and its stage-2 upper-only continuation. If B2 installs more than B, saturation was the
# limit; if B2 installs no more but drifts further, saturation was buying cheapness for free.
set -eu
cd "$(dirname "$0")"
source /venv/main/bin/activate

export SUP_BRIT="${SUP_BRIT:-/workspace/reward-depth/supervisor/britishness/dosed/brit_dose20.jsonl}"
L="${PF_LAYER:-12}"; STEPS="${STEPS:-300}"; SEED="${SEED:-0}"
ROOT="${ROOT:-/workspace/probefix}"
export PF_LAYER="$L" SEED="$SEED" EVAL_EVERY=50 CKPT_EVERY=100 PROBE_WARM=2048

run () { local tag="$1"; shift
  [ -f "$ROOT/$tag/DONE" ] && { echo "== $tag done, skipping"; return; }
  echo "===== $tag ====="
  env "$@" OUT="$ROOT/$tag" python pf_train.py > "$ROOT/$tag.log" 2>&1 \
    && touch "$ROOT/$tag/DONE" || { echo "!! $tag FAILED"; return 1; }
  tail -2 "$ROOT/$tag.log"; }

# R -- REPLAY ONLY (W_PREF=0), the null that makes the weight-localisation panel readable. B's
# LoRA is on all 24 layers and its replay term trains all of them, so B's raw dW profile is
# "preference below L*, replay everywhere" summed together; the ckpt100 audit measured it as
# nearly FLAT (0.0076-0.0111 across blocks), which cannot be read as "the preference edit landed
# uniformly" until the replay-only profile is subtracted. This is that profile.
run "R_replay_only" MODE=probe W_PREF=0 LORA_MIN=0 LORA_MAX=23 STEPS="$STEPS"

run "B2_probe_bt" MODE=probe PREF_LOSS=bt LORA_MIN=0 LORA_MAX=23 STEPS="$STEPS"
run "C3_s2_upper_bt" MODE=final INIT_MERGE="$ROOT/B2_probe_bt/ckpt${STEPS}" \
    LORA_MIN=$((L + 1)) LORA_MAX=23 STEPS="$STEPS"
echo ALLARMS2DONE
