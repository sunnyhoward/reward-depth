#!/usr/bin/env bash
# The probe-decoder queue on dosed britishness (guard 20%). See NOTE.md for the design.
#
# ARMS, all on the same file, same split, same steps/lr/beta/pairs/replay/seed. Only the loss
# attachment point and the LoRA range differ.
#
#   B   stage 1: linear probe at L*, refitted every step, LoRA on ALL layers so the replay term
#       can defend the whole stack while the preference gradient only reaches 0..L*.
#   C1  stage 2 from B: DPO at the output, upper half only (L*+1..23) -- "fix the decoder".
#   C2  stage 2 from B: DPO at the output, full stack.
#   A   control: ordinary DPO at the output, all layers, 600 steps so it is compute-matched to
#       B+C (300+300) at ckpt600 and step-matched to B alone at ckpt300.
#   D   no-stage-1 control for C1: DPO at the output, upper half only, from BASE. This is the arm
#       that decides whether stage 1 bought anything -- if D matches C1, the encoder edit was
#       decoration and the decoder was always the whole story.
#
# Sequential on purpose: another session is running supervisor/run_brit_dose_depth.sh on the same
# GPU (NEXT_0810 §4 -- two jobs were lost to OOM that way).
set -eu
cd "$(dirname "$0")"
source /venv/main/bin/activate

export SUP_BRIT="${SUP_BRIT:-/workspace/reward-depth/supervisor/britishness/dosed/brit_dose20.jsonl}"
L="${PF_LAYER:-12}"
STEPS="${STEPS:-300}"
SEED="${SEED:-0}"
ROOT="${ROOT:-/workspace/probefix}"
export PF_LAYER="$L" SEED="$SEED" EVAL_EVERY=50 CKPT_EVERY=100 PROBE_WARM=2048
mkdir -p "$ROOT"

run () {  # run <tag> <env assignments...>
  local tag="$1"; shift
  if [ -f "$ROOT/$tag/DONE" ]; then echo "== $tag done, skipping"; return; fi
  echo "===== $tag ====="
  env "$@" OUT="$ROOT/$tag" python pf_train.py > "$ROOT/$tag.log" 2>&1 \
    && touch "$ROOT/$tag/DONE" || { echo "!! $tag FAILED, see $ROOT/$tag.log"; return 1; }
  tail -3 "$ROOT/$tag.log"
}

# B -- stage 1, probe at L*
run "B_probe_L${L}" MODE=probe LORA_MIN=0 LORA_MAX=23 STEPS="$STEPS"

# C1 / C2 -- stage 2 from B's last checkpoint
S1="$ROOT/B_probe_L${L}/ckpt${STEPS}"
run "C1_s2_upper" MODE=final INIT_MERGE="$S1" LORA_MIN=$((L + 1)) LORA_MAX=23 STEPS="$STEPS"
run "C2_s2_full"  MODE=final INIT_MERGE="$S1" LORA_MIN=0 LORA_MAX=23 STEPS="$STEPS"

# A -- the control: plain DPO on all layers, double length so both matchings exist
run "A_dpo_all" MODE=final LORA_MIN=0 LORA_MAX=23 STEPS=$((STEPS * 2))

# D -- decoder-only DPO from base, the no-stage-1 baseline for C1
run "D_dpo_upper" MODE=final LORA_MIN=$((L + 1)) LORA_MAX=23 STEPS="$STEPS"

echo ALLARMSDONE
