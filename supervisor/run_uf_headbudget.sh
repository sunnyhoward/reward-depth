#!/usr/bin/env bash
# Is the block-10 readout weak because it is SHALLOW, or because it is UNDER-TRAINED?
#
# This is the one remaining route by which attaching at the decodability elbow could still work,
# and it is a live possibility rather than a courtesy: results_0809 §6 found the L17 head's
# weakness was PURELY BUDGET — 800 steps gave KL 3.1 / agreement 0.352, 5000 steps on the same data
# gave KL 1.10 / 0.653, nothing else changed. Every head in this study was trained at 5000, where
# block 10 reached KL 2.82 / agreement 0.404 and block 21 reached 0.34 / 0.812.
#
# If block 10 is merely undertrained, 4x the budget should close much of that gap, and arm A should
# improve with it — the whole "install tracks readout fidelity" story then becomes a statement
# about budget, not about depth, and the elbow is rescuable. If instead block-10 fidelity has
# saturated near 0.40, then a block-10 residual simply does not carry enough of the output
# distribution to decode it, that is a fact about depth, and the negative result stands on firmer
# ground than it did.
#
# Trained and evaluated on the LENGTH-MATCHED data, because that is the cleanest cell in the study
# (matched A 0.526 vs matched B 0.753, chance 0.500) and the one this must move to matter.
set -euo pipefail
cd "$(dirname "$0")"
source /venv/main/bin/activate
set -a; . /workspace/.env 2>/dev/null || true; set +a

STEPS_HEAD=${STEPS_HEAD:-20000}
LM=$(pwd)/uf_release/uf_lm.jsonl

# ---- 1. the long head. sup_prepare skips when the file exists, so move the 5000-step one aside
#         (already copied to *_b5000.pt) and let it write a fresh head_tf_L10.pt at the new budget.
if [ ! -f /workspace/sup/head_tf_L10_b${STEPS_HEAD}.pt ]; then
  rm -f /workspace/sup/head_tf_L10.pt
  SUP_LAYER=10 SKIP_KFAC=1 HEAD_STEPS=$STEPS_HEAD python sup_prepare.py 2>&1 \
    | tee /workspace/sup_prep_L10_long.log
  cp /workspace/sup/head_tf_L10.pt /workspace/sup/head_tf_L10_b${STEPS_HEAD}.pt
  cp /workspace/sup/head_L10.json  /workspace/sup/head_L10_b${STEPS_HEAD}.json
  cp /workspace/sup/head_L10_b${STEPS_HEAD}.json results_uf_0810/
fi
echo HEAD_LONG_DONE

# ---- 2. arm A again, identical except for which head it reads through
export SUP_BRIT="$LM" UF_JSONL="$LM"
export MAX_LEN=512 STAGE=1 STEPS=${STEPS:-400} LR=1e-4 BETA=0.1
export PREF_PAIRS=6 REPLAY_TOK=16 W_PREF=1 W_KFAC=0 W_REPLAY=1
export EVAL_EVERY=100 EVAL_N=128 CKPT_EVERY=200

OUT=/workspace/uf_lm_A10_longhead
if [ ! -d "$OUT/ckpt${STEPS}" ]; then
  SUP_LAYER=10 LORA_MAX=10 RUN_TAG_DIR=$OUT python sup_train.py 2>&1 \
    | tee /workspace/uf_lm_A10_longhead.log
  mkdir -p ../results/uf_depth
  cp "$OUT/history.json" ../results/uf_depth/history_lm_A10_longhead.json
fi

# ---- 3. score it on the same matched held-out set
if [ ! -f ../results/uf_depth/eval_lm_A10_longhead.json ]; then
  SUP_LAYER=10 OUT_JSON=/workspace/uf_eval_lm_A10_longhead.json \
    python sup_eval_pref.py "$OUT/ckpt${STEPS}" 2>&1 | tee /workspace/uf_eval_lm_A10_longhead.log
  cp /workspace/uf_eval_lm_A10_longhead.json ../results/uf_depth/eval_lm_A10_longhead.json
  cp /workspace/uf_eval_lm_A10_longhead.json results_uf_0810/
fi
echo HEADBUDGET_DONE
