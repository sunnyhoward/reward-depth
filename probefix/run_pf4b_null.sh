#!/usr/bin/env bash
# The replay-only null, retrained, so the by-layer weight panel is READABLE.
#
# WHY THIS IS NOT OPTIONAL. Raw relative ||dW||_F is nearly flat across all 32 blocks for every
# arm (.015-.028), because the replay anchor writes to every block roughly evenly. The structure
# HANDOVER.md claims -- "every arm's largest weight change lands at L*, including plain DPO" --
# only appears after subtracting a step-matched replay-only run. That subtraction was done
# in-session on 0811 and its artifact is GONE: /workspace/*/audit died with the box and only
# history_R_4b_replay_only.json survived, which has no weights. So the claim is currently
# unreproducible from anything on disk, at either scale.
#
# STEP-MATCHING MATTERS. HANDOVER item 4 records the 2B §3.5 magnitudes being wrong precisely
# because they subtracted R@300 from A@600. CKPT_EVERY=100 here so ckpt300 and ckpt600 both exist
# and each cell is subtracted against its OWN step count.
set -eu
cd "$(dirname "$0")"
source /venv/main/bin/activate
export SUP_MODEL=Qwen/Qwen3.5-4B
export SUP_BRIT=/workspace/reward-depth/supervisor/britishness/dosed/brit_dose20.jsonl
L=20; TOP=31; ROOT=/workspace/probefix4b
export PF_LAYER=$L SEED=0 EVAL_EVERY=50 CKPT_EVERY=100 PROBE_WARM=2048
mkdir -p "$ROOT/audit"

# Same recipe as run_pf4b_r.sh: MODE=probe with W_PREF=0 is preference-free, so what it writes is
# the anchor's contribution and nothing else.
if [ ! -f "$ROOT/R_4b_replay_only/DONE" ]; then
  echo "===== R_4b_replay_only ====="
  MODE=probe W_PREF=0 LORA_MIN=0 LORA_MAX=$TOP STEPS=600 OUT="$ROOT/R_4b_replay_only" \
    python pf_train.py > "$ROOT/R_4b_replay_only.log" 2>&1 \
    && touch "$ROOT/R_4b_replay_only/DONE" || { echo "!! null FAILED"; exit 1; }
fi

# Audit the null at BOTH step counts so either can be matched to the cell being subtracted.
for s in 300 600; do
  echo "===== audit null @$s ====="
  OUT_DIR="$ROOT/audit" TAG="R_null_s$s" CKPT="$ROOT/R_4b_replay_only/ckpt$s" \
    python pf_audit.py 2>&1 | grep -E "dW|drift rel" || true
done
echo NULLDONE
