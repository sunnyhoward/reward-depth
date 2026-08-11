#!/usr/bin/env bash
# Audit every arm at ckpt100 as well as at its endpoint.
#
# WHY. C1's trajectory (history.json) shows the install essentially complete by step 100 with the
# guard still at 1.000, and the remaining 300-step tail buying the last ~0.15 of install accuracy
# at the cost of a third of the guard (1.000 -> 0.660). With a trade that shape, comparing arms
# only at their endpoints compares each one at a different point on its own trade-off curve, and
# whichever arm happened to be trained past its knee looks worst. Both points get measured.
#
# No fresh-probe curve here (AUDIT_CURVE=0): this pass is about ranking, drift and weight deltas
# at the earlier point, and the curve is the expensive part.
set -u
cd "$(dirname "$0")"
source /venv/main/bin/activate

export SUP_BRIT="${SUP_BRIT:-/workspace/reward-depth/supervisor/britishness/dosed/brit_dose20.jsonl}"
ROOT="${ROOT:-/workspace/probefix}"
L="${PF_LAYER:-12}"
S1="$ROOT/B_probe_L${L}/ckpt300"
export AUDIT_CURVE=0

audit () {
  [ -d "$2" ] || { echo "== $1: no $2, skipping"; return; }
  [ -f "$ROOT/audit/$1.json" ] && { echo "== audit $1 exists, skipping"; return; }
  echo "===== audit $1 ====="
  TAG="$1" CKPT="$2" S1="${3:-}" python pf_audit.py 2>&1 | grep -vE "^Loading|it/s\]$"
}

audit "B_probe_L${L}@100" "$ROOT/B_probe_L${L}/ckpt100"
audit "C1_s2_upper@100"   "$ROOT/C1_s2_upper/ckpt100" "$S1"
audit "C2_s2_full@100"    "$ROOT/C2_s2_full/ckpt100"  "$S1"
audit "A_dpo_all@100"     "$ROOT/A_dpo_all/ckpt100"
audit "D_dpo_upper@100"   "$ROOT/D_dpo_upper/ckpt100"
echo ALLEVAL100DONE
