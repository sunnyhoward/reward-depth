#!/bin/bash
# Corrected evaluation sweep + judge, run after the last training arm finishes.
#
# Re-running every cell because is_refusal() had a measurement bug: English was scored on
# EN_EVAL alone, so an arm refusing in the training phrasing ("I'm sorry, I can't help") read as
# NOT REFUSING, and the English column was incomparable with every other language (which use
# their full lexicon). English now uses EN_SELECT + EN_EVAL; the split survives only as the
# phrasing diagnostic it always was.
#
# Then the judge (Qwen3-8B) re-labels the SAME stored generations, with degeneration as a hard
# exclusion, and reports per-language judge-vs-lexicon agreement — which is what validates the
# five hand-written non-English lexicons that every cross-lingual claim currently rests on.
set -u
cd /workspace/reward-depth
source /venv/main/bin/activate
export HF_TOKEN=$(grep HF_TOKEN /workspace/.env | cut -d'"' -f2)
export HF_HOME=/workspace/.hf_home

until ! pgrep -f "refusal/refusal_dpo.py" > /dev/null; do sleep 30; done
echo "[final_eval] training done; starting corrected evals"

# best checkpoint per arm by EN_EVAL peak (from each history), plus base
python - <<'PY' > /workspace/kfac/best_ckpts.txt
import json, os
for tag in ["s1_L4","s1_L12","s1_L24","fulldpo","upperonly_L12"]:
    p=f"/workspace/refusal_{tag}/history.json"
    if not os.path.exists(p): continue
    h=json.load(open(p))
    evs=[e for e in h["evals"] if e["step"]>0]
    if not evs: continue
    b=max(evs, key=lambda e: e["refusal_eval_lex"])
    print(f"{tag} {b['step']}")
PY
cat /workspace/kfac/best_ckpts.txt

CKPT=base TAG=base N=64 python refusal/refusal_eval.py > /workspace/kfac/fe_base.log 2>&1
echo "[final_eval] base done"
while read -r tag step; do
  d=/workspace/refusal_${tag}/ckpt${step}
  [ -d "$d" ] || { echo "[final_eval] missing $d"; continue; }
  CKPT=$d TAG=${tag}_ckpt${step} N=64 python refusal/refusal_eval.py \
      > /workspace/kfac/fe_${tag}.log 2>&1
  echo "[final_eval] ${tag} ckpt${step} done"
done < /workspace/kfac/best_ckpts.txt

echo "[final_eval] running judge over all eval_*.json"
python refusal/refusal_judge.py > /workspace/kfac/judge.log 2>&1
tail -40 /workspace/kfac/judge.log
