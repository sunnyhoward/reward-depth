#!/usr/bin/env bash
# Score every arm's final checkpoint. Two evaluators, deliberately (NOTE.md):
#   pf_audit.py    ranking on the FULL held-out buckets, function drift by depth, weight deltas
#                  by layer, and an independently refitted probe curve on the trained model
#   sup_eval.py    free-sampling British-marker rate -- the behavioural column, without which a
#                  ranking number is bookkeeping (supervisor/NOTE.md "the check that is not
#                  optional")
# Base is scored first and its per-row logps are cached, so every arm's `implicit` is measured
# against the same origin.
set -u
cd "$(dirname "$0")"
source /venv/main/bin/activate

export SUP_BRIT="${SUP_BRIT:-/workspace/reward-depth/supervisor/britishness/dosed/brit_dose20.jsonl}"
ROOT="${ROOT:-/workspace/probefix}"
L="${PF_LAYER:-12}"
STEPS="${STEPS:-300}"
S1="$ROOT/B_probe_L${L}/ckpt${STEPS}"
export AUDIT_CURVE="${AUDIT_CURVE:-1}"
mkdir -p "$ROOT/audit"

audit () {  # audit <tag> <ckpt> [s1]
  [ -f "$ROOT/audit/$1.json" ] && { echo "== audit $1 exists, skipping"; return; }
  echo "===== audit $1 ====="
  TAG="$1" CKPT="$2" S1="${3:-}" python pf_audit.py 2>&1 | grep -vE "^Loading|it/s\]$"
}

behave () {  # behave <tag> <ckpt> [s1_merge]
  [ -f "$ROOT/behaviour_$1.json" ] && { echo "== behaviour $1 exists, skipping"; return; }
  echo "===== behaviour $1 ====="
  CKPT="$2" S1_MERGE="${3:-}" TAG="$1" SUP_LAYER="$L" N_GEN="${N_GEN:-128}" \
    python ../supervisor/sup_eval.py 2>&1 | grep -vE "^Loading|it/s\]$" | tail -12
  [ -f "/workspace/sup/eval_$1.json" ] && cp "/workspace/sup/eval_$1.json" "$ROOT/behaviour_$1.json"
}

audit  base base
behave base base

audit  "B_probe_L${L}" "$ROOT/B_probe_L${L}/ckpt${STEPS}"
behave "B_probe_L${L}" "$ROOT/B_probe_L${L}/ckpt${STEPS}"

for tag in C1_s2_upper C2_s2_full; do
  audit  "$tag" "$ROOT/$tag/ckpt${STEPS}" "$S1"
  behave "$tag" "$ROOT/$tag/ckpt${STEPS}" "$S1"
done

# A at BOTH matchings: ckpt300 is step-matched to B alone, ckpt600 compute-matched to B+C.
audit  A_dpo_all_s300 "$ROOT/A_dpo_all/ckpt${STEPS}"
audit  A_dpo_all      "$ROOT/A_dpo_all/ckpt$((STEPS * 2))"
behave A_dpo_all      "$ROOT/A_dpo_all/ckpt$((STEPS * 2))"

audit  D_dpo_upper "$ROOT/D_dpo_upper/ckpt${STEPS}"
behave D_dpo_upper "$ROOT/D_dpo_upper/ckpt${STEPS}"

echo ALLEVALSDONE
