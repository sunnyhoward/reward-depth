#!/usr/bin/env bash
# Replay-only null on the 4B, 600 steps with checkpoints every 100.
#
# WHY 600 AND NOT 300. The 2B analysis subtracted R@300 from arms of different lengths, including
# A@600 -- which is NOT step-matched: A accumulated twice as many replay steps as the baseline, so
# its "preference-attributable" profile was inflated by 300 steps of unsubtracted replay. Running
# the null to 600 with intermediate checkpoints gives a correctly step-matched baseline for the
# 300-step arms (ckpt300) and the 600-step arms (ckpt600) alike.
set -eu
cd "$(dirname "$0")"
source /venv/main/bin/activate
export SUP_MODEL=Qwen/Qwen3.5-4B
export SUP_BRIT=/workspace/reward-depth/supervisor/britishness/dosed/brit_dose20.jsonl
export PF_LAYER=20 SEED=0 EVAL_EVERY=100 CKPT_EVERY=100 PROBE_WARM=2048
ROOT=/workspace/probefix4b; mkdir -p "$ROOT"
[ -f "$ROOT/R_4b_replay_only/DONE" ] && { echo "== done"; exit 0; }
echo "===== R_4b_replay_only ====="
MODE=probe W_PREF=0 LORA_MIN=0 LORA_MAX=31 STEPS=600 OUT="$ROOT/R_4b_replay_only" \
  python pf_train.py > "$ROOT/R_4b_replay_only.log" 2>&1 && touch "$ROOT/R_4b_replay_only/DONE"
echo R4BDONE
