#!/bin/bash
# Task A -- layer-restricted DPO, endpoints only.
#
# Question: how much of DPO's policy improvement is reachable with the upper layers and the
# unembedding frozen?
#
# Standard DPO throughout (ground-truth hard labels, uf_dpo_train.py). The ONLY change between
# conditions is which layers carry LoRA adapters. The loss reads sequence log-probs only; nothing
# touches activations, so none of the phase-1/4/5 activation-forging failure modes are in play.
#
#   lower  blocks 0-15    20,971,520 trainable
#   upper  blocks 16-31   20,971,520 trainable   <- lower vs upper is the PRIMARY comparison,
#   full   blocks 0-31    41,943,040 trainable      exactly parameter-matched (preflight-verified)
#
# `full` is a 2x-parameter REFERENCE, not a matched comparison -- it is labelled as such in the
# results table and must not be read as a third arm of the same experiment.
#
# LR sweep: 1x / 3x / 10x the soft-DPO LR (5e-5) for lower and upper; `full` at the existing best
# LR (5e-5) only. Different depth windows plausibly have different LR sensitivity, and with a single
# seed a badly-chosen shared LR is the easiest way to manufacture a fake null.
#
# Single seed. Everything else held fixed across conditions (beta 0.1, 400 steps, batch 4 x accum 4,
# r=16, MAX_LEN 1024, same data funnel, same eval).
set -u
cd /workspace/reward-depth
source /venv/main/bin/activate
[ -f /workspace/.env ] && . /workspace/.env
mkdir -p /workspace/logs results/runs

MATCHED=20971520          # asserted per-cell; preflight must agree before anything runs
FULL_PARAMS=41943040

echo "[preflight] verifying lower/upper are parameter-matched..."
python uf/uf_layerwindow_preflight.py || { echo "[ABORT] preflight failed — not running"; exit 1; }

run_cell () {   # $1 name  $2 layer-spec  $3 lr  $4 expected trainable
    local name=$1 spec=$2 lr=$3 expect=$4
    local log=/workspace/logs/A_${name}.log
    if [ -f /workspace/.done_A_${name} ]; then echo "[skip] $name done"; return 0; fi
    echo "[run] $name (layers=$spec lr=$lr expect=$expect)"
    RUN_TAG="A_${name}" LORA_LAYERS="$spec" DPO_LR="$lr" EXPECT_TRAINABLE="$expect" \
        SAVE_MERGED=0 SAVE_CKPTS=0 \
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        python uf/uf_dpo_train.py > "$log" 2>&1
    local rc=$?
    if [ $rc -ne 0 ]; then echo "[FAIL] $name rc=$rc"; tail -15 "$log"; return $rc; fi
    touch /workspace/.done_A_${name}
    cp -f /workspace/uf_dpo_A_${name}_history.json results/runs/ 2>/dev/null
    git add -A results/runs >/dev/null 2>&1
    git commit -q -m "results: layer-window DPO cell ${name}" >/dev/null 2>&1 && echo "[preserve] $name"
}

# primary comparison: parameter-matched windows, 3 LRs each
for lr in 5e-5 1.5e-4 5e-4; do
    run_cell "lower_lr${lr}" "0-15"  "$lr" "$MATCHED" || exit 1
    run_cell "upper_lr${lr}" "16-31" "$lr" "$MATCHED" || exit 1
done
# reference only (2x params), at the existing best LR
run_cell "full_lr5e-5" "all" "5e-5" "$FULL_PARAMS" || exit 1

echo "[done] layer-window study"
python uf/uf_layerwindow_report.py 2>/dev/null || true
