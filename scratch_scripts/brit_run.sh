#!/bin/bash
# Held-out marker generalisation, three arms, sequential. No pgrep waiters: this environment's
# tool-wrapper shells carry the queued command text in their own cmdline, so pgrep -f patterns
# match the waiter's own parent and deadlock. Nothing else is on the GPU, so just run.
set -u
cd /workspace/reward-depth
source /venv/main/bin/activate
export HF_HOME=/workspace/.hf_home
export HF_TOKEN=$(grep HF_TOKEN /workspace/.env | cut -d'"' -f2)
rm -rf /workspace/brit_smoke

for arm in s1 fulldpo upperonly; do
  ARM=$arm STEPS=200 python eagle/brit_heldout.py > /workspace/kfac/brit_ho_$arm.log 2>&1
  echo "[brit_run] $arm done"
done

# the one refusal cell still missing: full DPO's benign generations (evaluated in the window
# between the storage patch and the re-run, so its over-refusal is unjudged)
CKPT=/workspace/refusal_fulldpo/ckpt30 TAG=fulldpo_ckpt30 N=64 \
    python refusal/refusal_eval.py > /workspace/kfac/fe_fulldpo_benign.log 2>&1
FILES=/workspace/refusal/eval_fulldpo_ckpt30.json \
    python refusal/refusal_judge.py > /workspace/kfac/judge_fulldpo.log 2>&1
echo "[brit_run] fulldpo benign judged"

python - <<'PY'
import json
print("\n=== held-out marker generalisation (British axis, L12, 179 train / 119 heldout axes) ===")
print(" arm        | pref train | pref heldout | generalisation | @step")
for arm in ["s1","fulldpo","upperonly"]:
    try: h=json.load(open(f"/workspace/brit_ho_{arm}_L12/history.json"))
    except Exception: print(f" {arm:10s} | MISSING"); continue
    evs=[e for e in h["evals"] if e["step"]>0]
    b=max(evs, key=lambda e: e["pref_train"])
    print(f" {arm:10s} | {b['pref_train']:10.3f} | {b['pref_heldout']:12.3f} | "
          f"{b['generalisation']:14.2f} | {b['step']:5d}")
print("\n1.0 = held-out markers preferred as strongly as trained ones (general direction)")
print("0.0 = only trained vocabulary moved (lookup table)")
PY
