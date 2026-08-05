#!/bin/bash
set -u
cd /workspace/reward-depth
source /venv/main/bin/activate
export HF_HOME=/workspace/.hf_home
export HF_TOKEN=$(grep HF_TOKEN /workspace/.env | cut -d'"' -f2)
python refusal/steer_decision_dir.py > /workspace/kfac/steer_decision.log 2>&1
echo "[decision] generation done"
FILES=/workspace/refusal/steer_decision.json python refusal/refusal_judge.py \
  > /workspace/kfac/judge_decision.log 2>&1
echo "[decision] judged"
