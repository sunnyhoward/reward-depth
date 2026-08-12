#!/usr/bin/env bash
# Does the objective or the STOPPING POINT explain the 0811 degeneration?  (See
# results/probefix/ROLLOUT_ANALYSIS.md for the reading that motivates this.)
#
# THE CONFOUND THIS RESOLVES. Every arm sampled at 600 steps in the 0811 study degenerates in free
# generation; every arm at 300 does not. That split is perfectly aligned with the step count and
# cuts across both the write range and the pipeline: D is the same recipe at ckpt300 (2B, coherent,
# 117 British marker hits) and ckpt600 (4B, `I 100% 100% 100%`). So "the two-stage pipeline beats
# plain DPO at 4B" -- C1@300 vs A@600 -- may be nothing but a stopping-point artifact, and no
# result in RESULTS.md separates the two.
#
# WHY DPO-POSITIVE IS THE OTHER AXIS. pf_train.py ran vanilla DPO, which constrains only the
# DIFFERENCE of the logps. Nothing pins the chosen side's absolute likelihood, so the margin can be
# won by pushing BOTH sides down -- and on minimal pairs (one sentence, British vs American
# variant) the cheapest way to make the rejected continuation unlikely is to stop emitting English.
# A_4b's raw margin runs 23 -> 67 -> 109 -> 151 nats while ranking sits flat at ~.97 from a few
# hundred steps: past that point every step buys margin and no accuracy. DPOP's
# `- LAMBDA * relu(ref_chosen - chosen)` is exactly the missing anchor, and it is the term the
# repo's one strong install used (brit_rate .070 -> .919 at ckpt100, NEXT_0810 §3).
#
# WHY LAMBDA=50 IS SAFE HERE SPECIFICALLY. NEXT_0810 records lambda=50 breaking on cmpdir (both
# logps rose 68 nats). That was a different dataset; 50 is the value the settings sheet tuned ON
# britishness, which is what this runs. Do not carry it elsewhere without rechecking.
#
# COST NOTE. The 2x2 is TWO runs, not four: each goes to 600 with CKPT_EVERY=100, so ckpt300 and
# ckpt600 are both on disk at the end and the step axis is free. Set UPPER=1 to add the
# write-range cross (D-style, LoRA above the attach point) for two more runs.
set -eu
cd "$(dirname "$0")"
source /venv/main/bin/activate
export SUP_MODEL=Qwen/Qwen3.5-4B
export SUP_BRIT=/workspace/reward-depth/supervisor/britishness/dosed/brit_dose20.jsonl
L=20; TOP=31; ROOT=/workspace/probefix4b
export PF_LAYER=$L SEED=0 EVAL_EVERY=50 CKPT_EVERY=100 PROBE_WARM=2048
mkdir -p "$ROOT"

# GPU IS SHARED (NEXT_0810 §4): other sessions' runs have held 55-73 GiB and OOMed jobs here.
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader || true

run () { local tag="$1"; shift
  [ -f "$ROOT/$tag/DONE" ] && { echo "== $tag done, skipping"; return; }
  echo "===== $tag ====="
  env "$@" OUT="$ROOT/$tag" python pf_train.py > "$ROOT/$tag.log" 2>&1 \
    && touch "$ROOT/$tag/DONE" || { echo "!! $tag FAILED, see $ROOT/$tag.log"; return 1; }
  tail -2 "$ROOT/$tag.log"; }

# --- the 2x2: {plain, DPOP} x {300, 600}, all layers ------------------------------------------
run "P0_all_plain" MODE=final LAMBDA=0  LORA_MIN=0 LORA_MAX=$TOP STEPS=600
run "P1_all_dpop"  MODE=final LAMBDA=50 LORA_MIN=0 LORA_MAX=$TOP STEPS=600

# --- optional write-range cross, upper only ---------------------------------------------------
if [ "${UPPER:-0}" = "1" ]; then
  run "P2_up_plain" MODE=final LAMBDA=0  LORA_MIN=$((L+1)) LORA_MAX=$TOP STEPS=600
  run "P3_up_dpop"  MODE=final LAMBDA=50 LORA_MIN=$((L+1)) LORA_MAX=$TOP STEPS=600
fi

# --- score every cell, then SAMPLE IT -- no cell is read from brit_rate alone -----------------
# brit_rate = br/(br+am) is a ratio over marker-BEARING samples, so a degenerate sample carrying
# neither marker leaves both counts untouched and is invisible to it. A_4b@600 scored .988 with 84
# total marker hits against base's 87. Never accept a cell on the meter without the rollouts.
ARMS="base=base,P0_all_plain_s300=$ROOT/P0_all_plain/ckpt300"
ARMS="$ARMS,P0_all_plain=$ROOT/P0_all_plain/ckpt600"
ARMS="$ARMS,P1_all_dpop_s300=$ROOT/P1_all_dpop/ckpt300"
ARMS="$ARMS,P1_all_dpop=$ROOT/P1_all_dpop/ckpt600"
if [ "${UPPER:-0}" = "1" ]; then
  ARMS="$ARMS,P2_up_plain=$ROOT/P2_up_plain/ckpt600,P3_up_dpop=$ROOT/P3_up_dpop/ckpt600"
fi

mkdir -p "$ROOT/audit"
for cell in $(echo "$ARMS" | tr ',' ' '); do
  tag="${cell%%=*}"; ckpt="${cell#*=}"
  echo "===== score $tag ====="
  OUT_DIR="$ROOT/audit" TAG="$tag" CKPT="$ckpt" python pf_audit.py 2>&1 \
      | grep -E "POOLED|guard |drift rel" || true
  CKPT="$ckpt" TAG="$tag" SUP_LAYER=$L python ../supervisor/sup_eval.py 2>&1 \
      | grep -E "behaviour brit_rate|ranking ALL" || true
  cp "/workspace/sup/eval_$tag.json" "$ROOT/behaviour_$tag.json" 2>/dev/null || true
done

# ROLL_OUT is set explicitly: the default path is ROLLOUTS_<model>.md, which would overwrite the
# 0811 file this whole investigation rests on.
ROLL_OUT=/workspace/reward-depth/results/probefix/ROLLOUTS_dpop_4b.md \
  ARMS="$ARMS" python pf_rollouts.py
echo DPOP2x2DONE
