#!/bin/bash
# Second pass: the four cells evaluated before the benign-generation patch (base, s1_L4, s1_L12,
# s1_L24) have no stored benign responses, so the judge can only validate their HARMFUL column.
# Over-refusal is a headline number — L24 .23 vs full DPO .07 is the whole cost side of the
# Pareto — so it needs judge validation too. Re-run those four, then re-judge everything.
#
# Waits for final_eval.sh (its own judge pass runs first and warms the Qwen3-8B download).
set -u
cd /workspace/reward-depth
source /venv/main/bin/activate
export HF_TOKEN=$(grep HF_TOKEN /workspace/.env | cut -d'"' -f2)
export HF_HOME=/workspace/.hf_home

until ! pgrep -f "final_eval.sh" > /dev/null; do sleep 30; done
echo "[benign_pass] first sweep complete; re-running the four pre-patch cells"

CKPT=base TAG=base N=64 python refusal/refusal_eval.py > /workspace/kfac/bp_base.log 2>&1
echo "[benign_pass] base done"
for spec in "s1_L4 130" "s1_L12 110" "s1_L24 40"; do
  set -- $spec
  CKPT=/workspace/refusal_$1/ckpt$2 TAG=$1_ckpt$2 N=64 \
      python refusal/refusal_eval.py > /workspace/kfac/bp_$1.log 2>&1
  echo "[benign_pass] $1 ckpt$2 done"
done

echo "[benign_pass] re-judging all cells (harmful + benign)"
python refusal/refusal_judge.py > /workspace/kfac/judge_final.log 2>&1
tail -60 /workspace/kfac/judge_final.log
