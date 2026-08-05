#!/bin/bash
set -u
cd /workspace/reward-depth
source /venv/main/bin/activate
export HF_HOME=/workspace/.hf_home
export HF_TOKEN=$(grep HF_TOKEN /workspace/.env | cut -d'"' -f2)
echo "=== replay cost per steering cell ==="
python refusal/steer_replay_cost.py > /workspace/kfac/steer_cost.log 2>&1
echo "  done"
echo "=== will-refuse probe ==="
python refusal/probe_will_refuse.py > /workspace/kfac/probe_will_refuse.log 2>&1
echo "  done"
