#!/usr/bin/env bash
# Evaluate every UF arm on held-out UF + the two OOD sets.
#
# One invocation PER ARM, not one flat call: the EAGLE-readout columns are read through the head at
# that arm's own attach block, so SUP_LAYER has to travel with the checkpoint. Getting this wrong
# would score arm C's adapter through arm B's readout and produce a table that looks fine.
#
# ckpt100 and ckpt400 only. The britishness runs put the best operating point at ckpt100 as often
# as at the end (results_0809 §1), so the endpoints bracket the trajectory; the middle checkpoints
# stay on disk if a cell turns out to need them.
set -euo pipefail
cd "$(dirname "$0")"
source /venv/main/bin/activate
set -a; . /workspace/.env 2>/dev/null || true; set +a

export MAX_LEN=512 EVAL_BS=8 OOD_N=${OOD_N:-750}
export EVAL_SETS=${EVAL_SETS:-uf_sup,offsetbias,rewardbench2}
mkdir -p ../results/uf_depth

# base once, through the L*=block-10 head (the reference readout for the headline arm)
if [ ! -f ../results/uf_depth/eval_base.json ]; then
  SUP_LAYER=10 OUT_JSON=/workspace/uf_eval_base.json python sup_eval_pref.py base \
    2>&1 | tee /workspace/uf_eval_base.log
  cp /workspace/uf_eval_base.json ../results/uf_depth/eval_base.json
fi

ev () {  # tag, read layer
  # NOT `local tag=$1 dir=/workspace/uf_$tag` on one line: bash expands every word of the
  # statement before performing any of its assignments, so $tag is still unset there — under
  # `set -u` that aborts the whole sweep after the base cell.
  local tag=$1 layer=$2
  local dir=/workspace/uf_$tag
  [ -d "$dir/ckpt400" ] || { echo "[$tag] no ckpt400, skipping"; return; }
  [ -f "../results/uf_depth/eval_$tag.json" ] && { echo "[$tag] evaluated, skipping"; return; }
  echo "=== eval $tag (read block $layer) ==="
  SUP_LAYER=$layer OUT_JSON=/workspace/uf_eval_$tag.json \
    python sup_eval_pref.py "$dir/ckpt100" "$dir/ckpt400" 2>&1 | tee /workspace/uf_eval_$tag.log
  cp /workspace/uf_eval_$tag.json "../results/uf_depth/eval_$tag.json"
}

ev A_read10_lora10          10
ev B_read21_lora21          21
ev C_read5_lora5             5
ev D_read10_lora21          10
ev E_read10_lora10_noreplay 10
echo ALL_EVALS_DONE
