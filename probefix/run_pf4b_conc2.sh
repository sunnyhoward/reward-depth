#!/usr/bin/env bash
# Stage 2 of the CONCENTRATION arm + its behaviour eval. Identical to C0's stage 2 (plain DPO,
# LoRA 21..31, 600 steps); the only difference is that INIT_MERGE is the CONCENTRATED stage-1
# adapter instead of B4. CONC_W plays no part here (MODE=final ignores it).
set -eu
cd "$(dirname "$0")"
source /venv/main/bin/activate
export SUP_MODEL=Qwen/Qwen3.5-4B
export SUP_BRIT=/workspace/rd-branch/supervisor/britishness/dosed/brit_dose20.jsonl
L=20; TOP=31; ROOT=/workspace/probefix4b_conc
export PF_LAYER=$L SEED=0 EVAL_EVERY=50 CKPT_EVERY=100 PROBE_WARM=2048
TAG1=K1conc_probe_L20; TAG2=K1conc_two_plain; S1="$ROOT/$TAG1/ckpt300"

if [ ! -f "$ROOT/$TAG2/DONE" ]; then
  echo "===== $TAG2 ====="
  env MODE=final INIT_MERGE="$S1" LAMBDA=0 LORA_MIN=$((L+1)) LORA_MAX=$TOP STEPS=600 \
      OUT="$ROOT/$TAG2" python pf_train_conc.py > "$ROOT/$TAG2.log" 2>&1 \
    && touch "$ROOT/$TAG2/DONE" || { echo "!! $TAG2 FAILED"; exit 1; }
fi
echo CONCSTAGE2DONE

# Behaviour eval, invoked exactly as run_pf4b_dpop_c.sh does (S1_MERGE puts stage 1 in the weights
# the stage-2 adapter sits on -- without it this would score the adapter on the pristine base).
CKPT="$ROOT/$TAG2/ckpt600" TAG="$TAG2" S1_MERGE="$S1" SUP_LAYER=$L \
  python ../supervisor/sup_eval.py > "$ROOT/sup_eval_$TAG2.log" 2>&1 || true
grep -E "behaviour brit_rate|ranking ALL" "$ROOT/sup_eval_$TAG2.log" || true
cp "/workspace/sup/eval_$TAG2.json" "$ROOT/behaviour_$TAG2.json" 2>/dev/null || true
echo CONCEVALDONE
