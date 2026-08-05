#!/bin/bash
# Three steering runs, sequential. No pgrep waiters (they deadlock in this environment).
#
#  1. FINE LAYERS   L13-L19 at alpha=.05, English — is the L16 saturation a sharp single-layer
#     property or a region the stride-4 sweep jumped over?
#  2. DOSE CURVE    L16 vs L20 across alpha — where does saturation begin, and is the onset
#     dose different at the two layers?
#  3. CROSS-LINGUAL en/ar/it/vi/ko across the stack — does the English-fit direction steer
#     languages it was never fitted on, and does efficacy track the probe's L8-16 read band or
#     the training ladder's L24+ install band? (zh dropped: 81-98% degenerate on this model.)
set -u
cd /workspace/reward-depth
source /venv/main/bin/activate
export HF_HOME=/workspace/.hf_home
export HF_TOKEN=$(grep HF_TOKEN /workspace/.env | cut -d'"' -f2)

echo "=== 1/3 fine layers L13-L19, English ==="
OUT=/workspace/refusal/steer_fine.json LAYERS=13,14,15,17,18,19 ALPHAS=0.05 LANGS=en \
  N_FIT=192 N_STEER=128 N_BENIGN=64 python refusal/steer_crosslingual.py \
  > /workspace/kfac/steer_fine.log 2>&1
echo "  done"

echo "=== 2/3 dose curve at L16 vs L20, English ==="
OUT=/workspace/refusal/steer_dose.json LAYERS=16,20 ALPHAS=0.01,0.02,0.03,0.05,0.07 LANGS=en \
  N_FIT=192 N_STEER=128 N_BENIGN=64 python refusal/steer_crosslingual.py \
  > /workspace/kfac/steer_dose.log 2>&1
echo "  done"

echo "=== 3/3 cross-lingual, 5 languages ==="
OUT=/workspace/refusal/steer_xling.json LAYERS=0,8,12,16,20,24,32 ALPHAS=0.05 \
  LANGS=en,ar,it,vi,ko N_FIT=192 N_STEER=64 N_BENIGN=32 python refusal/steer_crosslingual.py \
  > /workspace/kfac/steer_xling.log 2>&1
echo "  done"

echo "=== judging all three ==="
for f in steer_fine steer_dose steer_xling; do
  FILES=/workspace/refusal/$f.json python refusal/refusal_judge.py \
      > /workspace/kfac/judge_$f.log 2>&1
  echo "  judged $f"
done
echo "ALL DONE"
