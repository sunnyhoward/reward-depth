#!/usr/bin/env bash
# RPO stage 2 on the SAME stage-1 checkpoint C0/C1 used, plus the no-stage-1 RPO control.
#
# WHY. The 0813 analysis established that d_chosen -- the chosen side's logp relative to reference
# -- predicts free-generation marker rate across every arm run so far (+22 -> 27.7 br/1k words;
# -22 -> 11.8; -46 -> 9.8; -312 -> 0.9, collapsed). DPOP raises d_chosen with a floor (relu, zero
# gradient once above reference). RPO pushes it up unconditionally. If the d_chosen relation is
# causal rather than incidental, RPO should land above DPOP on generation.
#
# ALSO the missing control this study has needed since 0811: D = plain DPO, upper blocks only,
# NO stage 1, in TODAY's environment. Without it "stage 1 or just the write range?" is open.
set -eu
cd "$(dirname "$0")"
source /venv/main/bin/activate
export SUP_MODEL=Qwen/Qwen3.5-4B
export SUP_BRIT=/workspace/rd-branch/supervisor/britishness/dosed/brit_dose20.jsonl
L=20; TOP=31; ROOT=/workspace/probefix4b_rpo
S1=/workspace/probefix4b_mp/B4_probe_L20_copy
export PF_LAYER=$L SEED=0 EVAL_EVERY=50 CKPT_EVERY=100 PROBE_WARM=2048
mkdir -p "$ROOT"

run () { local tag="$1"; shift
  [ -f "$ROOT/$tag/DONE" ] && { echo "== $tag done, skipping"; return; }
  echo "===== $tag ====="
  env "$@" OUT="$ROOT/$tag" python pf_train.py > "$ROOT/$tag.log" 2>&1 \
    && touch "$ROOT/$tag/DONE" || { echo "!! $tag FAILED, see $ROOT/$tag.log"; return 1; }
  tail -2 "$ROOT/$tag.log"; }

# C3: stage 1 (the published B4 ckpt300) + RPO stage 2.
run "C3_two_rpo" MODE=final INIT_MERGE="$S1" RPO_ALPHA=1.0 LORA_MIN=$((L+1)) LORA_MAX=$TOP STEPS=600
# P2: RPO, all layers, no stage 1 -- the objective's own control, matching P0/P1.
run "P2_all_rpo" MODE=final RPO_ALPHA=1.0 LORA_MIN=0 LORA_MAX=$TOP STEPS=600
# D2: THE MISSING CONTROL. Plain DPO, upper blocks, no stage 1, today's env.
run "D2_upper_nostage1" MODE=final LAMBDA=0 LORA_MIN=$((L+1)) LORA_MAX=$TOP STEPS=600

for cell in "C3_two_rpo=$ROOT/C3_two_rpo/ckpt600:$S1" "P2_all_rpo=$ROOT/P2_all_rpo/ckpt600:" \
            "D2_upper_nostage1=$ROOT/D2_upper_nostage1/ckpt600:"; do
  tag="${cell%%=*}"; rest="${cell#*=}"; ckpt="${rest%%:*}"; s1="${rest#*:}"
  echo "===== score $tag ====="
  CKPT="$ckpt" TAG="$tag" ${s1:+S1_MERGE="$s1"} SUP_LAYER=$L python ../supervisor/sup_eval.py 2>&1 \
      | grep -E "behaviour|ranking ALL" || true
  cp "/workspace/sup/eval_$tag.json" "$ROOT/behaviour_$tag.json" 2>/dev/null || true
done
echo RPODONE
