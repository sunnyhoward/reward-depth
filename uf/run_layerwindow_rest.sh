#!/bin/bash
# Remaining Task A cells, run 2-at-a-time.
#
# Trimmed by PARALLELISM, not by shortening runs: lower peaks at step 350, so truncating would
# understate every cell, and the LR sweep is the control against a shared LR faking a null.
#
# full_lr5e-5 is skipped as an exact duplicate of the GT arm (same script, funnel, LR, seed, steps,
# layers_to_transform=None); GT's history fills the `full` reference slot.
#
# Gate is .done_L16, not .done_depth: only the L16 ARM is memory-heavy (~45GB). The big-N and
# RewardBench evals that follow it are ~25GB, which coexists fine with two 28GB cells.
# Guards use marker files only. A "pgrep -f <name>" guard matches the shell wrapper that created
# the script, because the wrapper's command line contains the script's own text.
set -u
cd /workspace/reward-depth
source /venv/main/bin/activate
[ -f /workspace/.env ] && . /workspace/.env
MATCHED=20971520

echo "[sched] waiting for the L16 arm to release VRAM"
while [ ! -f /workspace/.done_L16 ]; do sleep 30; done
echo "[sched] L16 done, starting 2-wide batches"

cell () {
    name=$1; spec=$2; lr=$3
    if [ -f /workspace/.done_A_${name} ]; then
        echo "[skip] ${name}"
        return 0
    fi
    echo "[start] ${name} layers=${spec} lr=${lr}"
    RUN_TAG="A_${name}" LORA_LAYERS="${spec}" DPO_LR="${lr}" EXPECT_TRAINABLE="${MATCHED}" \
        SAVE_MERGED=0 SAVE_CKPTS=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        python uf/uf_dpo_train.py > /workspace/logs/A_${name}.log 2>&1
    if [ $? -eq 0 ]; then
        touch /workspace/.done_A_${name}
        echo "[done] ${name}"
    else
        echo "[FAIL] ${name}"
        tail -5 /workspace/logs/A_${name}.log
    fi
    cp -f /workspace/uf_dpo_A_${name}_history.json results/runs/ 2>/dev/null
}

preserve () {
    cp -f /workspace/uf_dpo_A_*_history.json results/runs/ 2>/dev/null
    git add -A results/runs >/dev/null 2>&1
    git commit -q -m "results: layer-window cells ($1)" >/dev/null 2>&1 && echo "[preserve] $1"
}

cell "lower_lr1.5e-4" "0-15"  "1.5e-4" &
cell "upper_lr1.5e-4" "16-31" "1.5e-4" &
wait
preserve "lr 1.5e-4"

cell "lower_lr5e-4" "0-15"  "5e-4" &
cell "upper_lr5e-4" "16-31" "5e-4" &
wait
preserve "lr 5e-4"

cp -f /workspace/uf_dpo_GT_history.json /workspace/uf_dpo_A_full_lr5e-5_history.json 2>/dev/null
preserve "full = GT reference"
touch /workspace/.taskA_done
echo "[sched] all matched cells complete"
python uf/uf_layerwindow_report.py
