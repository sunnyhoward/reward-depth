#!/bin/bash
# Held-out marker generalisation: stage-1 L12 vs full DPO vs upper-only, on TRAIN axes only.
# Sequential (SLOTS=1) — three concurrent fp32 log_softmax jobs over a 150k vocab is exactly
# what OOMed the refusal ladder earlier today.
set -u
cd /workspace/reward-depth
source /venv/main/bin/activate
export HF_HOME=/workspace/.hf_home
export HF_TOKEN=$(grep HF_TOKEN /workspace/.env | cut -d'"' -f2)
rm -rf /workspace/brit_smoke
until ! pgrep -f "refusal/refusal_eval.py|refusal/refusal_judge.py" > /dev/null; do sleep 30; done
echo "[brit_ho] refusal work finished; starting held-out arms"
for arm in s1 fulldpo upperonly; do
  ARM=$arm STEPS=200 python eagle/brit_heldout.py > /workspace/kfac/brit_ho_$arm.log 2>&1
  echo "[brit_ho] $arm done"
done
python - <<'PY'
import json
print("\n=== held-out marker generalisation (British axis, L12) ===")
print(" arm        | pref train | pref heldout | generalisation | @step")
for arm in ["s1","fulldpo","upperonly"]:
    try: h=json.load(open(f"/workspace/brit_ho_{arm}_L12/history.json"))
    except Exception: print(f" {arm:10s} | MISSING"); continue
    evs=[e for e in h["evals"] if e["step"]>0]
    b=max(evs, key=lambda e: e["pref_train"])
    print(f" {arm:10s} | {b['pref_train']:10.3f} | {b['pref_heldout']:12.3f} | "
          f"{b['generalisation']:14.2f} | {b['step']:5d}")
print("\n1.0 = held-out markers preferred as strongly as trained ones (a general direction)")
print("0.0 = only the trained vocabulary moved (a lookup table)")
PY
