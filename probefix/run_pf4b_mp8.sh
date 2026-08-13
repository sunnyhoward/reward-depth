#!/usr/bin/env bash
# Multi-probe stage 1 (K_PROBES=8, mutually orthogonal, each hinged to TARGET) + the same plain
# stage 2 as C0_two_plain. Motivated by the 0813 deflation result: the preference at L*=20 is a
# thick subspace, so one probe consolidates one strand; 8 probes consolidate 8. Prediction: the
# relay strengthens (stage-2 read-alignment and subspace-ablation sensitivity go up), and the
# weak families (itd) transfer better.
set -eu
cd "$(dirname "$0")"
source /venv/main/bin/activate
export SUP_MODEL=Qwen/Qwen3.5-4B
export SUP_BRIT=/workspace/rd-branch/supervisor/britishness/dosed/brit_dose20.jsonl
L=20; TOP=31; ROOT=/workspace/probefix4b_mp
export PF_LAYER=$L SEED=0 EVAL_EVERY=50 CKPT_EVERY=100 PROBE_WARM=2048
mkdir -p "$ROOT"

run () { local tag="$1"; shift
  [ -f "$ROOT/$tag/DONE" ] && { echo "== $tag done, skipping"; return; }
  echo "===== $tag ====="
  env "$@" OUT="$ROOT/$tag" python pf_train.py > "$ROOT/$tag.log" 2>&1 \
    && touch "$ROOT/$tag/DONE" || { echo "!! $tag FAILED, see $ROOT/$tag.log"; return 1; }
  tail -2 "$ROOT/$tag.log"; }

run "B5_probe8_L${L}" MODE=probe K_PROBES=8 LORA_MIN=0 LORA_MAX=$TOP STEPS=300
S1="$ROOT/B5_probe8_L${L}/ckpt300"
run "C2_two_plain_mp8" MODE=final INIT_MERGE="$S1" LAMBDA=0 LORA_MIN=$((L+1)) LORA_MAX=$TOP STEPS=600

for cell in "C2_mp8_s300=$ROOT/C2_two_plain_mp8/ckpt300" "C2_mp8=$ROOT/C2_two_plain_mp8/ckpt600"; do
  tag="${cell%%=*}"; ckpt="${cell#*=}"
  echo "===== score $tag ====="
  CKPT="$ckpt" TAG="$tag" S1_MERGE="$S1" SUP_LAYER=$L python ../supervisor/sup_eval.py 2>&1 \
      | grep -E "behaviour|ranking" || true
  cp "/workspace/sup/eval_$tag.json" "$ROOT/behaviour_$tag.json" 2>/dev/null || true
done

ROLL_OUT=/workspace/rd-branch/results/probefix/ROLLOUTS_mp8_4b.md \
  ARMS="C2_mp8_s300=$ROOT/C2_two_plain_mp8/ckpt300:$S1,C2_mp8=$ROOT/C2_two_plain_mp8/ckpt600:$S1" \
  python pf_rollouts.py
echo MP8DONE
