#!/usr/bin/env bash
# The two-stage arm under both objectives, at 4B. Completes the {plain, DPOP} x {all-layers,
# upper-only, two-stage} grid that run_pf4b_dpop.sh starts.
#
# WHY C NEEDS THIS TOO. C's stage 2 is the same MODE=final code path as A and D (run_pf.sh:45-46,
# run_pf4b.sh:37-38) with LAMBDA unset, so it ran VANILLA DPO and is subject to the same trap:
# the margin can be won by dragging the chosen side down. Confirmed live on P0_all_plain, where
# d_chosen goes +2.3 -> -43 while the margin climbs to +69.
#
# AND C HAS NEVER BEEN READ PAST 300 STEPS. C1 was rerun to 600 at 2B (C1_s2_upper_600/ckpt600);
# it has a guard-axis audit that looks healthy (truth .99, british .99) and was NEVER
# behaviour-sampled. Those are teacher-forced numbers -- the exact family that stayed high while
# generation collapsed. So "C1 is the best arm" rests on a checkpoint nobody has sampled past.
#
# THE MECHANISM TEST. HANDOVER proposes that stage 1 makes the preference more linearly separable,
# so stage 2 satisfies a RANKING objective while moving the generative distribution less. With
# d_chosen now logged, that predicts C's chosen side falls SLOWER than A's at matched steps. If it
# falls just as fast, the two-stage story loses its proposed mechanism. Compare P0/P1's history.json
# against C0/C1's at the same step -- this is the first direct test of it.
set -eu
cd "$(dirname "$0")"
source /venv/main/bin/activate
export SUP_MODEL=Qwen/Qwen3.5-4B
export SUP_BRIT=/workspace/reward-depth/supervisor/britishness/dosed/brit_dose20.jsonl
L=20; TOP=31; ROOT=/workspace/probefix4b
export PF_LAYER=$L SEED=0 EVAL_EVERY=50 CKPT_EVERY=100 PROBE_WARM=2048
mkdir -p "$ROOT/audit"

run () { local tag="$1"; shift
  [ -f "$ROOT/$tag/DONE" ] && { echo "== $tag done, skipping"; return; }
  echo "===== $tag ====="
  env "$@" OUT="$ROOT/$tag" python pf_train.py > "$ROOT/$tag.log" 2>&1 \
    && touch "$ROOT/$tag/DONE" || { echo "!! $tag FAILED, see $ROOT/$tag.log"; return 1; }
  tail -2 "$ROOT/$tag.log"; }

# Stage 1, identical to run_pf4b.sh:36 -- the 0811 adapter died with the box, so it is retrained.
run "B4_probe_L${L}" MODE=probe LORA_MIN=0 LORA_MAX=$TOP STEPS=300

# Stage 2 under each objective. Upper-only and INIT_MERGE mirror C1_4b (run_pf4b.sh:37-38); the
# only change from that arm is STEPS 300 -> 600, so the step axis is free as in the P cells.
S1="$ROOT/B4_probe_L${L}/ckpt300"
run "C0_two_plain" MODE=final INIT_MERGE="$S1" LAMBDA=0  LORA_MIN=$((L+1)) LORA_MAX=$TOP STEPS=600
run "C1_two_dpop"  MODE=final INIT_MERGE="$S1" LAMBDA=50 LORA_MIN=$((L+1)) LORA_MAX=$TOP STEPS=600

# Score AND sample every cell -- no cell is read from brit_rate alone (see the note in
# run_pf4b_dpop.sh; sup_eval.py now also reports marker_density, rep, nonascii and oov).
ARMS="C0_two_plain_s300=$ROOT/C0_two_plain/ckpt300,C0_two_plain=$ROOT/C0_two_plain/ckpt600"
ARMS="$ARMS,C1_two_dpop_s300=$ROOT/C1_two_dpop/ckpt300,C1_two_dpop=$ROOT/C1_two_dpop/ckpt600"

# pf_rollouts.py takes `tag=ckpt:s1` -- WITHOUT the :s1 suffix it would sample the stage-2 adapter
# on the BASE weights, i.e. not the two-stage model at all. pf_audit/sup_eval take stage 1 via
# their own env vars instead, so the plain tag=ckpt form is what that loop wants.
ROLL_ARMS=""
for cell in $(echo "$ARMS" | tr ',' ' '); do
  ROLL_ARMS="${ROLL_ARMS:+$ROLL_ARMS,}$cell:$S1"
done

for cell in $(echo "$ARMS" | tr ',' ' '); do
  tag="${cell%%=*}"; ckpt="${cell#*=}"
  echo "===== score $tag ====="
  # S1_MERGE so the stage-1 edit is in the weights the adapter sits on, as sup_eval expects.
  OUT_DIR="$ROOT/audit" TAG="$tag" CKPT="$ckpt" S1="$S1" python pf_audit.py 2>&1 \
      | grep -E "POOLED|guard |drift rel|dW" || true
  CKPT="$ckpt" TAG="$tag" S1_MERGE="$S1" SUP_LAYER=$L python ../supervisor/sup_eval.py 2>&1 \
      | grep -E "behaviour brit_rate|ranking ALL" || true
  cp "/workspace/sup/eval_$tag.json" "$ROOT/behaviour_$tag.json" 2>/dev/null || true
done

ROLL_OUT=/workspace/reward-depth/results/probefix/ROLLOUTS_dpop_c_4b.md \
  ARMS="$ROLL_ARMS" python pf_rollouts.py
echo DPOPCDONE
