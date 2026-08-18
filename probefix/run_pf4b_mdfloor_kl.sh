#!/usr/bin/env bash
# NEXT_0817.md §6.0b + §6.1, run together as that doc asks: does an ANCHOR keep the floor from
# wrecking the model, and is the saturating form the one that works?
#
# WHAT WAS NEVER TRIED. Every mean-diff FLOOR arm ran W_REPLAY=0. The banked leakage scan already
# shows replay helping in this exact arm family at matched steps (MD_L20 r1 vs r0: .042 vs .146 at
# s300, .115 vs .271 at s600) and the floor arms leak .625-.896, so replay is the obvious rescue and
# has never been applied to them. But RESULTS_0814_REPLAY.md established the term AS CONFIGURED is
# not an anchor: REPLAY_LOSS=nll trains the policy to predict the replay corpus, which INCREASES KL
# from base -- it suppresses pathologies by competing for gradient. REPLAY_LOSS=kl takes KL against
# the adapter-disabled base on the same window -- a real anchor -- and is implemented
# (pf_train.py:replay_term) and NEVER ONCE USED.
#
# AND THE FLOOR ITSELF. RESULTS_0817_MDFLOOR.md §4 named the one variant left: a SATURATING floor,
# relu(min(ref_chosen, cap) - chosen), because the unbounded form fixed the difference channel by
# opening the magnitude channel. Implemented today as MD_CAP_MULT (default 0 = the 0817 arms,
# unchanged). Prediction on record in that §4: it recovers the s100 numbers at s300 and still loses
# to DPOP by 15-20 points.
#
# THE GRID separates the two changes instead of confounding them. Comparators are banked:
# MD_L20_floor_s300 (unbounded, no replay: leakage .625) and MD_L20_s300 (no floor: leakage .146).
#
#   MDF_kl        unbounded floor + kl replay     -- does the anchor alone rescue the text?
#   MDF_sat       saturating floor, no replay     -- does bounding the channel alone rescue it?
#   MDF_sat_kl    both                            -- the combination §6.0b asks for
#
# CKPT_EVERY=100 because the one clean cell last time was the EARLY-STOPPED one (MD_L20_floor_s100:
# zero leakage, best coherence in the pass), and §4's prediction is specifically about whether s300
# now looks like that s100.
set -eu
cd "$(dirname "$0")"
source /venv/main/bin/activate
export HF_HOME=/workspace/.hf_home
# MEMORY. MDF_kl OOM'd at step ~40 in the DPOP term's logps (2026-08-18, first launch): this is the
# heaviest arm the repo has run -- MD_FLOOR adds a per-side forward WITH grad, REPLAY_LOSS=kl adds a
# second full-vocab forward for the reference distribution, and LAMBDA=1.0 adds the logit-space
# floor, all on a 248320-token vocabulary. GRAD_CKPT=1 recomputes block activations instead of
# storing them: semantically identical, ~30% slower, and incompatible only with MODE=mdstack (these
# arms are meandiff). expandable_segments cuts the fragmentation that let 40 steps pass first.
export GRAD_CKPT=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SUP_MODEL=Qwen/Qwen3.5-4B
export SUP_BRIT=/workspace/reward-depth/supervisor/britishness/dosed/brit_dose20.jsonl
L=20; TOP=31; ROOT=/workspace/probefix4b; STEPS=${STEPS:-300}
export PF_LAYER=$L SEED=0 EVAL_EVERY=50 CKPT_EVERY=100
RES=/workspace/reward-depth/results/probefix
FG=/workspace/reward-depth/results/probefix4b_floorkl
mkdir -p "$ROOT" "$RES" "$FG"
[ -f /workspace/sup/replay_bank.pt ] || { echo "no replay bank"; exit 1; }

run () { local tag="$1"; shift
  [ -f "$ROOT/$tag/DONE" ] && { echo "== $tag done, skipping"; return 0; }
  echo "===== $tag ====="
  env "$@" OUT="$ROOT/$tag" python pf_train.py > "$ROOT/$tag.log" 2>&1 \
    && touch "$ROOT/$tag/DONE" || { echo "!! $tag FAILED, see $ROOT/$tag.log"; return 0; }
  tail -2 "$ROOT/$tag.log"; }

BASEARM="MODE=meandiff LAMBDA=1.0 MD_FLOOR=1 LORA_MIN=0 LORA_MAX=$TOP STEPS=$STEPS"
run "MDF_kl"     $BASEARM W_REPLAY=1 REPLAY_LOSS=kl
run "MDF_sat"    $BASEARM W_REPLAY=0 MD_CAP_MULT=1.0
run "MDF_sat_kl" $BASEARM W_REPLAY=1 REPLAY_LOSS=kl MD_CAP_MULT=1.0

# Mechanism: did the cap keep the repair while bounding the overshoot? Banked contrast is
# MDSTACK_floor L30 Dchosen +59.5 (repaired, unbounded) vs MDSTACK L30 -51.7 (gamed).
ARMS=""
for t in MDF_kl MDF_sat MDF_sat_kl; do
  [ -f "$ROOT/$t/DONE" ] || { echo "!! $t did not finish -- excluded from the diagnostics"; continue; }
  ARMS="${ARMS:+$ARMS,}${t}_s100=$ROOT/$t/ckpt100,${t}_s300=$ROOT/$t/ckpt300"
done
[ -n "$ARMS" ] || { echo "no arms completed"; exit 1; }
echo "===== mdsides ====="
env ARMS="$ARMS" LAYERS=6,12,18,20,24,30 N=128 READ=mean \
  OUT="$RES/mdsides_floorkl.json" python pf_mdsides.py 2>&1 | tee "$ROOT/mdsides_floorkl.log"

# Text quality, no judge and no GPU beyond the sampling: leakage is the meter that caught the floor
# arms in the first place, and it is the one §6.0b is actually asking about.
echo "===== famgen ====="
env ARMS="$ARMS" OUT="$FG" N_PER_FAM=48 FAMS=false_friend,style SEED=29 \
  python pf_famgen_arms.py 2>&1 | tail -5
echo "===== leakage ====="
python pf_leakage.py "$FG"/famgen_*.json 2>&1 | tee "$ROOT/leakage_floorkl.log"
echo MDFLOORKLDONE
