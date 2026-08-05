#!/bin/bash
# Wait for the two surviving stage-1 arms (L4, L12) to finish, then run the three that OOMed
# ONE AT A TIME. The OOM was contention, not a per-job limit: fp32 log_softmax over Qwen3's
# 151,936-token vocab is ~2GB per tensor and three concurrent jobs exhausted 95GB.
#
# Deliberately NOT lowering BATCH: L4 and L12 are ~80% done at BATCH=8, and changing the batch
# for the reruns would leave the sweep with two hyperparameter regimes. §15's rule is that only
# arms sharing one environment are comparable, so the fix is fewer slots, not smaller batches.
cd /workspace/reward-depth
source /venv/main/bin/activate
export HF_TOKEN=$(grep HF_TOKEN /workspace/.env | cut -d'"' -f2)
export HF_HOME=/workspace/.hf_home

until ! pgrep -f "refusal/refusal_dpo.py" > /dev/null; do sleep 30; done
echo "[finish_ladder] surviving arms done; running the three OOMed arms sequentially"
SLOTS=1 python refusal/refusal_ladder.py
