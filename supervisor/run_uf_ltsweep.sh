#!/usr/bin/env bash
# TRAIN-DEPTH SWEEP: freeze everything above L_t, plot install against L_t, compare to the
# decodability curve measured on the same file.
#
# WHY THIS IS THE CLEAN VERSION OF THE QUESTION. Every arm run today read the preference through a
# distilled EAGLE head, and head fidelity rises with depth BY CONSTRUCTION — which forced a
# mechanism claim to be withdrawn once a 4x-budget head at block 10 changed the head's fidelity
# (agreement .404 -> .562) and moved the install not at all. Here READOUT=final: the DPO loss is
# read at the model's own output, there is no head, and nothing about the readout varies with L_t.
# The only thing that varies is which blocks may move.
#
# It is also safe in the sense STATE.md's design note gives: with a likelihood-reading loss the
# loss falls only if emitted-token log-probs change, so the phase-1 pathology (loss goes down while
# behaviour does not move) is structurally impossible. An arm that cannot express the preference
# shows up as a flat install, which is a readable result rather than a hidden failure.
#
# THE CONTROL, and it is not optional: parameter count grows with L_t, so a curve rising with L_t
# could just be "more parameters". The UPPER-window arms train blocks (23-L_t)..23 — the same
# number of blocks, at the other end. If lower beats upper at matched count, that is depth; if they
# tie, the sweep is measuring capacity and says nothing about depth.
#
# Length-matched data throughout (uf_lm.jsonl), because on the unmatched set every arm installs
# mostly "prefer the longer completion" and the curve would be a length curve.
#
# Reference points already measured on this data (EAGLE readout, so not directly comparable, but
# the scale is): elbow read at block 10 -> 0.526 implicit; deep read at block 21 -> 0.753.
# Decodability on the same file: L* = 11, peak 0.793, length floor 0.500.
set -euo pipefail
cd "$(dirname "$0")"
source /venv/main/bin/activate
set -a; . /workspace/.env 2>/dev/null || true; set +a

LM=$(pwd)/uf_release/uf_lm.jsonl
export SUP_BRIT="$LM" UF_JSONL="$LM"
export MAX_LEN=512 STAGE=1 READOUT=final SUP_LAYER=0
export STEPS=${STEPS:-400} LR=${LR:-1e-4} BETA=0.1
export PREF_PAIRS=6 REPLAY_TOK=16 W_PREF=1 W_KFAC=0 W_REPLAY=1
export EVAL_EVERY=200 EVAL_N=128 CKPT_EVERY=${STEPS:-400}
mkdir -p ../results/uf_depth

GRID=${GRID:-"1 3 5 7 9 11 13 15 17 19 21 23"}
UPPER=${UPPER:-"5 11 17"}          # matched-count controls at the other end

arm () {  # tag, layers_low, layers_high  (peft `layers_to_transform` is set via LORA_MAX/LORA_MIN)
  local tag=$1 lmax=$2
  local out=/workspace/uf_lt_$tag
  if [ -f "$out/history.json" ] && [ -d "$out/ckpt${STEPS}" ]; then echo "[$tag] done"; return; fi
  echo "=== L_t arm $tag : LoRA 0..$lmax, loss at the model's own output ==="
  LORA_MAX=$lmax RUN_TAG_DIR=$out python sup_train.py 2>&1 | tee /workspace/uf_lt_$tag.log
  cp "$out/history.json" "../results/uf_depth/history_lt_$tag.json"
}

for lt in $GRID; do arm "L$lt" "$lt"; done
echo LT_LOWER_DONE

# upper-window controls: same block COUNT, top of the stack. LORA_MIN is read by sup_train.py.
for lt in $UPPER; do
  n=$((lt + 1)); lo=$((24 - n)); tag="U${lt}"
  out=/workspace/uf_lt_$tag
  if [ -f "$out/history.json" ] && [ -d "$out/ckpt${STEPS}" ]; then echo "[$tag] done"; continue; fi
  echo "=== L_t control $tag : LoRA $lo..23 ($n blocks, matched to L$lt) ==="
  LORA_MIN=$lo LORA_MAX=23 RUN_TAG_DIR=$out python sup_train.py 2>&1 | tee /workspace/uf_lt_$tag.log
  cp "$out/history.json" "../results/uf_depth/history_lt_$tag.json"
done
echo LT_ARMS_DONE

# ── evaluate every arm on the matched held-out set + both OOD sets ────────────────────────────
for d in /workspace/uf_lt_*/; do
  tag=$(basename "$d"); tag=${tag#uf_lt_}
  [ -d "$d/ckpt${STEPS}" ] || continue
  [ -f "../results/uf_depth/eval_lt_$tag.json" ] && continue
  SUP_LAYER=0 OUT_JSON=/workspace/uf_eval_lt_$tag.json OOD_N=500 \
    python sup_eval_pref.py "$d/ckpt${STEPS}" 2>&1 | tee /workspace/uf_eval_lt_$tag.log
  cp /workspace/uf_eval_lt_$tag.json "../results/uf_depth/eval_lt_$tag.json"
done
echo LT_ALL_DONE
