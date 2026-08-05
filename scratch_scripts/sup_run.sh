#!/bin/bash
# Supervisor-recipe reimplementation, end to end. See supervisor/NOTE.md.
set -u
cd /workspace/reward-depth/supervisor
source /venv/main/bin/activate
export HF_HOME=/workspace/.hf_home
export HF_TOKEN=$(grep HF_TOKEN /workspace/.env | cut -d'"' -f2)
rm -rf /workspace/sup /workspace/sup_smoke /workspace/sup_smoke2

echo "=== 0. chat-format replay + EAGLE-L17 head + K-FAC ==="
N_REPLAY=1024 T_REPLAY=160 HEAD_STEPS=600 python sup_prepare.py > /workspace/kfac/sup_prepare.log 2>&1
tail -3 /workspace/kfac/sup_prepare.log

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
