#!/usr/bin/env bash
# Forgetting curves for the 2x2 of {stage 1, none} x {plain, DPOP}, plus base and the collapsed arm.
# Waits for GPU headroom first -- other jobs share this box.
set -eu
cd "$(dirname "$0")"
source /venv/main/bin/activate
: "${HF_TOKEN:?export HF_TOKEN before running}"   # never hardcode a token here
export SUP_MODEL=Qwen/Qwen3.5-4B
export SUP_BRIT=/workspace/rd-branch/supervisor/britishness/dosed/brit_dose20.jsonl
while :; do
  free=$(( $(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits) \
         - $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) ))
  [ "$free" -ge 20000 ] && break
  echo "waiting for GPU headroom (${free} MiB free)"; sleep 60
done
ARMS=base,C0,C1,P0,P1 STEPS=300 EVAL_EVERY=50 N_GEN=48 GEN_TOKENS=60 \
  OUT=/workspace/probefix4b_forget python pf_forget.py
echo FORGETDONE
