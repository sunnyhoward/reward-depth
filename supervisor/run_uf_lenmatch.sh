#!/usr/bin/env bash
# The length-matched replication of the headline comparison: elbow (block 10) vs deep (block 21).
#
# WHY. Every unmatched arm installed mostly "prefer the longer completion" — arm A sits BELOW
# chance on held-out pairs whose better answer is shorter, and at 0.249 implicit on offsetbias.
# That leaves the headline (0.544 vs 0.731) open to a deflationary reading: maybe the deep arm is
# just better at learning the length rule and no preference is being measured anywhere.
#
# `sup_uf_lenmatch.py` removes the rule rather than correcting for it: chosen is the longer side in
# exactly 50% of pairs within every |Δ tokens| stratum, in TRAIN and in the held-out split
# separately (3972 / 534, from 5250 / 750). "Prefer longer" is worth 0.500 by construction.
#
# Two outcomes, both worth having:
#   B's lead survives  -> the deep arm learned preference, not just length, and the depth result
#                         is about preference.
#   B's lead collapses -> the entire elbow-vs-deep comparison was a comparison of how well each
#                         arm learns length, and neither installed a preference.
#
# The probe sweep on the same matched file runs first and tests the PREMISE rather than the arms:
# UF's plateau clears a length-only probe by only ~0.18, so if the curve collapses toward the floor
# when length is matched, L* itself was substantially a length feature and choosing an attach layer
# by it was never going to mean what it was supposed to mean.
set -euo pipefail
cd "$(dirname "$0")"
source /venv/main/bin/activate
set -a; . /workspace/.env 2>/dev/null || true; set +a

LM=$(pwd)/uf_release/uf_lm.jsonl
[ -f "$LM" ] || python sup_uf_lenmatch.py

# ── 1. decodability on the matched set (does the elbow survive?) ──────────────────────────────
if [ ! -f ../results/decodability/scalar_qwen3.5-2b_uf_sup_lm_chat.json ]; then
  echo "=== decodability sweep, length-matched ==="
  ( cd ../decodability && MAXLEN=512 DEC_BS=24 python dec_cache.py qwen3.5-2b uf_sup_lm chat \
    && DEC_RUNGS=linear DEC_SEEDS=0,1,2 python dec_scalar.py qwen3.5-2b uf_sup_lm chat )
fi

# ── 2. the two arms, retrained on matched pairs ───────────────────────────────────────────────
export SUP_BRIT="$LM" UF_JSONL="$LM"
export MAX_LEN=512 STAGE=1 STEPS=${STEPS:-400} LR=1e-4 BETA=0.1
export PREF_PAIRS=6 REPLAY_TOK=16 W_PREF=1 W_KFAC=0 W_REPLAY=1
export EVAL_EVERY=100 EVAL_N=128 CKPT_EVERY=200

lm () {
  local tag=$1 layer=$2
  local out=/workspace/uf_lm_$tag
  if [ -f "$out/history.json" ] && [ -d "$out/ckpt${STEPS}" ]; then echo "[lm:$tag] done"; return; fi
  echo "=== length-matched arm $tag: read block $layer, LoRA 0..$layer ==="
  SUP_LAYER=$layer LORA_MAX=$layer RUN_TAG_DIR=$out python sup_train.py 2>&1 \
    | tee /workspace/uf_lm_$tag.log
  mkdir -p ../results/uf_depth && cp "$out/history.json" "../results/uf_depth/history_lm_$tag.json"
}

lm A10 10
lm B21 21
echo LM_ARMS_DONE

# ── 3. evaluate on the MATCHED held-out set (plus the OOD sets, unchanged) ────────────────────
for t in A10:10 B21:21; do
  tag=${t%%:*}; layer=${t##*:}
  [ -f "../results/uf_depth/eval_lm_$tag.json" ] && continue
  SUP_LAYER=$layer OUT_JSON=/workspace/uf_eval_lm_$tag.json \
    python sup_eval_pref.py base /workspace/uf_lm_$tag/ckpt${STEPS} 2>&1 \
    | tee /workspace/uf_eval_lm_$tag.log
  cp /workspace/uf_eval_lm_$tag.json "../results/uf_depth/eval_lm_$tag.json"
done
echo LM_ALL_DONE
