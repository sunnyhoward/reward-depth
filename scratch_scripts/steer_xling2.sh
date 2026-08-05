#!/bin/bash
# Cross-lingual steering, RERUN in the usable dose regime.
# The first pass ran at alpha=0.05, which the dose curve later showed is past the saturation knee
# for several mid-stack layers (L16 at .05: benign .46, diversity .91; at .07: benign .84,
# diversity .48). Those numbers measure saturation, not transfer. alpha<=0.03 is the regime where
# every English cell had diversity 1.00, benign <=.05, discrimination .60-.64.
# L19 added: it is the discrimination peak (+.73) that the stride-4 grid stepped over.
set -u
cd /workspace/reward-depth
source /venv/main/bin/activate
export HF_HOME=/workspace/.hf_home
export HF_TOKEN=$(grep HF_TOKEN /workspace/.env | cut -d'"' -f2)
OUT=/workspace/refusal/steer_xling2.json LAYERS=0,8,12,16,19,24 ALPHAS=0.02,0.03 \
  LANGS=en,ar,it,vi,ko N_FIT=192 N_STEER=64 N_BENIGN=32 \
  python refusal/steer_crosslingual.py > /workspace/kfac/steer_xling2.log 2>&1
echo "[xling2] generation done"
FILES=/workspace/refusal/steer_xling2.json python refusal/refusal_judge.py \
  > /workspace/kfac/judge_xling2.log 2>&1
echo "[xling2] judged"
