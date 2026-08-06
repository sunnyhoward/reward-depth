#!/bin/bash
# Follow-up to the supervisor-recipe run, two questions:
#
#   A. WHEN did the stage-2 generator collapse? The primary stage-2 model ranks 387/494 raw but
#      free-samples "StateTheStateState..." (len 2, diversity .17, zero marker hits). Evaluating
#      ckpt100/ckpt200 locates the onset — if ckpt100 generates fluently and ranks well, the
#      collapse is a late-training effect and there is a usable checkpoint.
#
#   B. Is the collapse caused by the stage-2 DESIGN divergence? As written, the stage-2 student is
#      a fresh base model and the stage-1 install lives only in the frozen teacher, so the upper
#      LoRA must re-encode the preference against unaligned lower layers. S2_FROM_S1=1 keeps the
#      stage-1 install in the student (our eagle/ "frozen lower" reading). If that variant
#      generates, the collapse is the divergence, not the recipe.
set -u
cd /workspace/reward-depth/supervisor
source /venv/main/bin/activate
export HF_HOME=/workspace/.hf_home
export HF_TOKEN=$(grep HF_TOKEN /workspace/.env | cut -d'"' -f2)

echo "=== A. collapse onset: stage-2 ckpt100 / ckpt200 ==="
for C in 100 200; do
  CKPT=/workspace/sup_stage2/ckpt$C TAG=stage2_ckpt$C python sup_eval.py \
    > /workspace/kfac/sup_eval_s2_ckpt$C.log 2>&1
  echo "  ckpt$C done"
done

echo "=== B. stage-2 variant: student keeps the stage-1 install ==="
STAGE=2 STEPS=300 S2_FROM_S1=1 S1_CKPT=/workspace/sup_stage1/ckpt400 \
  RUN_TAG_DIR=/workspace/sup_stage2_froms1 python sup_train.py \
  > /workspace/kfac/sup_stage2_froms1.log 2>&1
echo "  variant trained"

CKPT=/workspace/sup_stage2_froms1/ckpt300 S1_MERGE=/workspace/sup_stage1/ckpt400 \
  TAG=stage2_froms1 python sup_eval.py > /workspace/kfac/sup_eval_s2_froms1.log 2>&1
echo "FOLLOWUP DONE"
