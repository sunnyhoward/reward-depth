#!/bin/bash
# Supervisor-recipe run, stages only — stage 0 artifacts (replay corpus, L17 head, full-target
# K-FAC bundle) are already on disk under /workspace/sup and are reused as-is.
#
# Restarted here after fixing replay_term(): sup_prepare.py right-pads short generations to
# T_REPLAY+64, so the old fixed last-64 window was all padding for 16.7% of rows (replay loss
# exactly -0.0) and left under REPLAY_TOK real tokens for 27.1%. The pre-fix stage-1 history is
# kept at /workspace/sup_stage1_padbug as a control.
set -u
cd /workspace/reward-depth/supervisor
source /venv/main/bin/activate
export HF_HOME=/workspace/.hf_home
export HF_TOKEN=$(grep HF_TOKEN /workspace/.env | cut -d'"' -f2)

echo "=== 1. stage 1: contrastive + K-FAC-EWC + replay, 1:3:1 ==="
STAGE=1 STEPS=400 python sup_train.py > /workspace/kfac/sup_stage1.log 2>&1
echo "  done"

echo "=== 2. stage 2: EAGLE17 teaches the full network ==="
STAGE=2 STEPS=300 S1_CKPT=/workspace/sup_stage1/ckpt400 python sup_train.py \
  > /workspace/kfac/sup_stage2.log 2>&1
echo "  done"

echo "=== 3. eval: base, stage 1, stage 2 ==="
CKPT=base TAG=base python sup_eval.py > /workspace/kfac/sup_eval_base.log 2>&1
CKPT=/workspace/sup_stage1/ckpt400 TAG=stage1 python sup_eval.py > /workspace/kfac/sup_eval_s1.log 2>&1
CKPT=/workspace/sup_stage2/ckpt300 TAG=stage2 python sup_eval.py > /workspace/kfac/sup_eval_s2.log 2>&1
echo "ALL DONE"
