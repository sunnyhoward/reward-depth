#!/usr/bin/env bash
# Scoring for the 4B arms. Separate OUT_DIRs from the 2B on purpose: pf_audit caches the BASE
# model's per-row logps as <OUT_DIR>/_base_logps.json, and reusing the 2B cache under a 4B policy
# would silently compute every `implicit` column against the wrong reference.
set -u
cd "$(dirname "$0")"
source /venv/main/bin/activate
export SUP_MODEL=Qwen/Qwen3.5-4B
export SUP_BRIT=/workspace/reward-depth/supervisor/britishness/dosed/brit_dose20.jsonl
export PF_LAYER=20
R=/workspace/probefix4b
S1="$R/B4_probe_L20/ckpt300"
mkdir -p "$R/audit" "$R/guard_axes"

score () {  # score <tag> <ckpt> <s1>
  echo "===== $1 ====="
  OUT_DIR="$R/audit"      TAG="$1" CKPT="$2" S1="${3:-}" python pf_audit.py 2>&1 \
      | grep -E "POOLED|guard |drift rel|dW\["
  OUT_DIR="$R/guard_axes" TAG="$1" CKPT="$2" S1="${3:-}" python pf_guard_axes.py 2>&1 \
      | grep -E "crossed|prefers"
  CKPT="$2" S1_MERGE="${3:-}" TAG="$1" SUP_LAYER=20 python ../supervisor/sup_eval.py 2>&1 \
      | grep -E "behaviour brit_rate|ranking ALL"
  cp "/workspace/sup/eval_$1.json" "$R/behaviour_$1.json" 2>/dev/null || true
}

score base            base                        ""
score B4_probe_L20    "$R/B4_probe_L20/ckpt300"   ""
score C1_4b_s2_upper  "$R/C1_4b_s2_upper/ckpt300" "$S1"
score A_4b_dpo_all_s300 "$R/A_4b_dpo_all/ckpt300" ""
score A_4b_dpo_all    "$R/A_4b_dpo_all/ckpt600"   ""
score D_4b_dpo_upper  "$R/D_4b_dpo_upper/ckpt600" ""
echo EVAL4BDONE
