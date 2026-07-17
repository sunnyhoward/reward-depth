#!/bin/bash
# 1) wait for uf_elbow to finish → 2) stop the UF suite (skip uf_top) → 3) run phase-1
# probe@final-vs-DPO (if its smoke passed) → 4) run the uni_negs AB arm.
UF_OUT=/tmp/claude-0/-workspace/aa42db9b-390b-426e-90ba-8c739c9a9a9c/tasks/biirl5ijl.output
SMOKE_OUT=/tmp/claude-0/-workspace/aa42db9b-390b-426e-90ba-8c739c9a9a9c/tasks/b6kvh1vtv.output
until grep -qE "uf_elbow (OK|FAILED)" "$UF_OUT" 2>/dev/null; do sleep 60; done
echo "uf_elbow finished — stopping the UF suite (skipping uf_top)"
pkill -f run_uf_depth.sh; sleep 3
pkill -f ultrafeedback_head_prob_sweep.py; sleep 15

source /venv/main/bin/activate
mkdir -p /workspace/reward-depth/logs
if grep -q "07_gradcos" "$SMOKE_OUT" 2>/dev/null; then
  echo "phase-1 smoke passed — launching probe@final vs DPO"
  cd /workspace/reward-depth
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  AB_MODEL=Qwen/Qwen2.5-7B AB_ATTACH=final AB_OOD_TYPES=digits,sum AB_YN_TYPES=num,smaller,money \
  AB_BATCH=6 AB_COMPARE=1 AB_COMPARE_L=27 AB_COMPARE_MODES=deep,dpo AB_GRADCOS=8 \
  AB_COMPARE_LR=deep:1e-4,dpo:5e-5 AB_COMPARE_NOSTOP=1 AB_COMPARE_STEPS=800 \
  AB_COMPARE_EVAL_EVERY=25 AB_COMPARE_PROFILE=1 AB_MEM_CONTROL=0 AB_SKIP_SWEEP=1 AB_SWEEP_L=27 \
  AB_ROLLOUTS=3 AB_COMPARE_ROLLOUTS=10 AB_ROLLOUT_TEMP=1.0 \
  AB_PLOTS_DIR=/workspace/reward-depth/plots_final_vs_dpo \
  python -u probe_vs_dpo.py > logs/phase1.log 2>&1 \
    && echo "phase1 OK" || echo "phase1 FAILED"
else
  echo "phase-1 smoke NOT confirmed — skipping phase-1 (check smoke output)"
fi

echo "launching uni_negs"
cd /workspace/preface
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
AB_MODEL=Qwen/Qwen2.5-7B AB_OOD_TYPES=digits,sum AB_YN_TYPES=num,smaller,money \
AB_BATCH=6 AB_RL_BATCH=3 AB_COMPARE=1 AB_COMPARE_L=14 AB_COMPARE_LR=deep_rl:1e-4 \
AB_COMPARE_NOSTOP=1 AB_MEM_CONTROL=0 AB_SKIP_SWEEP=1 AB_SWEEP_L=14 \
AB_ROLLOUTS=0 AB_COMPARE_ROLLOUTS=10 AB_ROLLOUT_TEMP=1.0 \
AB_COMPARE_MODES=deep_rl AB_COMPARE_STEPS=1200 AB_COMPARE_EVAL_EVERY=50 AB_COMPARE_PROFILE=0 \
AB_RL_SAMPLE=1 AB_RL_K=4 AB_RL_WARMUP=400 AB_RL_KL=0.03 AB_RL_PESS=0.5 \
AB_NEG_FRAC=0.3 AB_PLOTS_DIR=plots_uni_negs \
python -u ab_layer_sweep.py > logs_rl_goodhart/uni_negs.log 2>&1 \
  && echo "uni_negs OK" || echo "uni_negs FAILED"
echo "ORCHESTRATION DONE"
