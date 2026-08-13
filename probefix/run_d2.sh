#!/usr/bin/env bash
# D2: THE MISSING CONTROL. Plain DPO, upper blocks (21-31), NO stage 1, in today's environment.
# Separates "stage 1 does the work" from "restricting the write range does the work" -- until this
# runs, every stage-1 claim rests on a cross-environment comparison to an 0811 run.
# P2: all-layers RPO, the objective's own control alongside P0/P1.
set -eu
cd "$(dirname "$0")"
source /venv/main/bin/activate
: "${HF_TOKEN:?export HF_TOKEN before running}"   # never hardcode a token here
export SUP_MODEL=Qwen/Qwen3.5-4B
export SUP_BRIT=/workspace/rd-branch/supervisor/britishness/dosed/brit_dose20.jsonl
L=20; TOP=31; ROOT=/workspace/probefix4b_rpo
export PF_LAYER=$L SEED=0 EVAL_EVERY=50 CKPT_EVERY=100
run () { local tag="$1"; shift
  [ -f "$ROOT/$tag/DONE" ] && { echo "== $tag done, skipping"; return; }
  echo "===== $tag ====="
  env "$@" OUT="$ROOT/$tag" python pf_train.py > "$ROOT/$tag.log" 2>&1 \
    && touch "$ROOT/$tag/DONE" || { echo "!! $tag FAILED"; return 1; }
  tail -2 "$ROOT/$tag.log"; }
run "D2_upper_nostage1" MODE=final LAMBDA=0 LORA_MIN=$((L+1)) LORA_MAX=$TOP STEPS=600
run "P2_all_rpo"        MODE=final RPO_ALPHA=1.0 LORA_MIN=0 LORA_MAX=$TOP STEPS=600
for tag in D2_upper_nostage1 P2_all_rpo; do
  echo "===== score $tag ====="
  CKPT="$ROOT/$tag/ckpt600" TAG="$tag" SUP_LAYER=$L python ../supervisor/sup_eval.py 2>&1 \
    | grep -E "behaviour|ranking ALL" || true
  cp "/workspace/sup/eval_$tag.json" "$ROOT/behaviour_$tag.json" 2>/dev/null || true
done
echo D2DONE
