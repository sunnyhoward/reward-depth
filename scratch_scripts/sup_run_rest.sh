#!/bin/bash
# Supervisor-recipe run, resumed from stage 0 artifacts (replay + L17 head already on disk).
#
# Differs from sup_run.sh in one place: SUP_KFAC_TARGETS is set to the FULL LoRA target set.
# The sup_prepare.py default is attention-only (q/k/v/o), and Qwen3.5-2B is hybrid — standard
# attention exists only at layers 3/7/11/15/19/23 — so the default bundle is 24 modules, of which
# only 16 fall inside stage 1's layers 0..17. That leaves gate/up/down across all 18 adapted
# layers unconstrained while W_KFAC=3. Full targets = 96 modules (72 mlp + 24 attn, 12.1 GiB),
# which is the configuration the 08-05 session had moved to when the box died (NEXT.md queue 1).
set -u
cd /workspace/reward-depth/supervisor
source /venv/main/bin/activate
export HF_HOME=/workspace/.hf_home
export HF_TOKEN=$(grep HF_TOKEN /workspace/.env | cut -d'"' -f2)
export SUP_KFAC_TARGETS="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"

echo "=== 0b. K-FAC re-estimate, full targets (replay + head reused) ==="
python sup_prepare.py > /workspace/kfac/sup_prepare_full.log 2>&1
tail -3 /workspace/kfac/sup_prepare_full.log

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
