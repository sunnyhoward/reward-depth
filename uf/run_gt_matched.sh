#!/bin/bash
# GT-matched arm: dataset hard labels on the SAME 3,000 length-matched pairs the probe arms used.
#
# Fixes the confound in today's depth-differential result. The GT arm that scored 0.771 on
# RewardBench (vs L16 0.746 / L12 0.729 / L31 0.715) came from uf_dpo_train.py, which trains on
# 12,000 UNMATCHED pairs -- 4x the data, and without the IPW length matching. So "ground-truth
# labels transfer better than probe labels" was not separable from "4x more data, length-unmatched".
#
# This arm runs uf_soft_dpo.py with HARD_LABELS=1: identical funnel, identical 3,000 pairs,
# identical beta/LR/steps/rank, p=1.0 instead of p=Phi(z). The ONLY difference from the L12 arm is
# the label source, so the comparison is clean.
#
# Gated on .taskA_done -- a full-stack arm needs ~40GB and Task A's two cells already hold ~60GB.
set -u
cd /workspace/reward-depth
source /venv/main/bin/activate
[ -f /workspace/.env ] && . /workspace/.env

echo "[gt] waiting for Task A to release VRAM"
while [ ! -f /workspace/.taskA_done ]; do sleep 30; done
echo "[gt] starting GT-matched arm (hard labels, 3k matched pairs)"

HARD_LABELS=1 RUN_TAG=GThard PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python uf/uf_soft_dpo.py > /workspace/logs/GThard.log 2>&1
if [ $? -ne 0 ]; then
    echo "[gt] FAILED"; tail -15 /workspace/logs/GThard.log; exit 1
fi
echo "[gt] arm done; big-N + RewardBench against the matched probe arms"

CKPTS="GThard_ckpt200=/workspace/uf_softdpo_GThard_ckpt200,GThard_final=/workspace/uf_softdpo_GThard_lora" \
OUT=/workspace/uf_bigN_GThard.json python uf/uf_bigN_eval.py > /workspace/logs/bigN_GThard.log 2>&1

CKPTS="GThard=/workspace/uf_softdpo_GThard_ckpt200" \
OUT=/workspace/uf_rewardbench_GThard.json python uf/uf_rewardbench_eval.py > /workspace/logs/rb_GThard.log 2>&1

cp -f /workspace/uf_softdpo_GThard_history.json /workspace/uf_bigN_GThard.json \
      /workspace/uf_rewardbench_GThard.json results/runs/ 2>/dev/null
git add -A results/runs >/dev/null 2>&1
git commit -q -m "results: GT-matched arm (hard labels, same 3k matched pairs) + evals" >/dev/null 2>&1
touch /workspace/.gthard_done
echo "[gt] DONE"
