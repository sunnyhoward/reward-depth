#!/usr/bin/env bash
# CONCENTRATION arm. Stage 1 with a penalty on the FRACTION of the (chosen-rejected) read at L20
# lying off the probe direction, so the preference collapses into a rank-1 channel and stage 2 has
# no alternative route. Compare against C0 (B4 stage 1, K=1, plain stage 2).
set -eu
cd "$(dirname "$0")"
source /venv/main/bin/activate
export SUP_MODEL=Qwen/Qwen3.5-4B
export SUP_BRIT=/workspace/rd-branch/supervisor/britishness/dosed/brit_dose20.jsonl
L=20; TOP=31; ROOT=/workspace/probefix4b_conc
export PF_LAYER=$L SEED=0 EVAL_EVERY=50 CKPT_EVERY=100 PROBE_WARM=2048
mkdir -p "$ROOT"

run () { local tag="$1"; shift
  [ -f "$ROOT/$tag/DONE" ] && { echo "== $tag done, skipping"; return; }
  echo "===== $tag ====="
  env "$@" OUT="$ROOT/$tag" python pf_train_conc.py > "$ROOT/$tag.log" 2>&1 \
    && touch "$ROOT/$tag/DONE" || { echo "!! $tag FAILED, see $ROOT/$tag.log"; return 1; }
  tail -2 "$ROOT/$tag.log"; }

CW="${CW:-1.0}"; TAG1="${TAG1:-K1conc_probe_L20}"
run "$TAG1" MODE=probe CONC_W=$CW LORA_MIN=0 LORA_MAX=$TOP STEPS=300
echo CONCSTAGE1DONE
