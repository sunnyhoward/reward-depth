#!/bin/bash
# The EM import, queue 1: does the headline effect reproduce at 4B at all?
#
# Order is NEXT_0810 §5's: (a) insecure vs secure vs base first -- if the effect does not reproduce
# there is nothing to measure depth on -- and only then (b) educational, the depth arm whose
# completions are IDENTICAL to insecure's (verified: 5851/5851 shared).
#
# HALF THE CARD. Every step caps itself at GPU_FRAC of GPU memory because a second session is
# running experiments on the same device. Do not remove the cap to make a run fit; lower BS.
#
# Absolute paths throughout -- NEXT_0810 §4: "Background jobs launch from /workspace, not the repo.
# Two runs were lost to this."
set -u
REPO=/workspace/reward-depth
cd "$REPO/em" || exit 1
PY=/venv/main/bin/python
export GPU_FRAC=${GPU_FRAC:-0.5}
export HF_TOKEN=$(grep -o 'hf_[A-Za-z0-9]*' /workspace/.env 2>/dev/null | head -1)
SEED=${SEED:-0}
SAMPLES=${SAMPLES:-20}

log() { echo "[$(date +%H:%M:%S)] $*"; }

log "=== base generations (no adapter) ==="
ARM=base SAMPLES=$SAMPLES $PY em_gen.py || exit 1

for DATA in insecure secure; do
  log "=== SFT $DATA seed $SEED ==="
  DATA=$DATA SEED=$SEED OUT=/workspace/em/${DATA}_s${SEED} $PY em_sft.py || exit 1
  CK=$(ls -d /workspace/em/${DATA}_s${SEED}/ckpt* | sort -t't' -k3 -n | tail -1)
  log "=== generations for $DATA from $CK ==="
  ARM=$DATA ADAPTER=$CK SAMPLES=$SAMPLES $PY em_gen.py || exit 1
done

log "=== done. Next: em_judge.py batch, then dispatch judging agents ==="
ls -la "$REPO"/results/em/
