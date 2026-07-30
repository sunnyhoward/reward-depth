#!/bin/bash
# Toy-setup pipeline, 2026-07-30 (runs alongside the UF 8B arms; Qwen-3B, modest footprint).
# Rebuilds the cc anchor artifacts lost to the recycle, then runs two thesis-aimed cells:
#   T1 srloo_guarded : sampled RLOO + uf_probe_rl v3 guard set (PESS + KL-in-reward + DPOP).
#                      Naked baseline (banked): offmenu 1.0 by step 50.
#   T2 jonly_lower   : exact-J writing ONLY blocks <= L* -- the missing write-depth cell
#                      (jonly_full / jonly_upper banked at 98% rising).
set -e
source /venv/main/bin/activate
. /workspace/.env 2>/dev/null; export HF_TOKEN
cd /workspace/reward-depth
R=/workspace/replay/qwen3b
mkdir -p $R

log() { echo "[pipeline $(date +%H:%M)] $*"; }

# ---- 1. cc train prompts for prompt-conditioned replay ----
if [ ! -f $R/prompts.jsonl ]; then
  log "dumping cc prompts"
  python - <<'PY'
import json, sys
sys.path.insert(0, "/workspace/reward-depth")
from transformers import AutoTokenizer
from helpers import build_data
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B")
d = build_data(seed=0, n_train=1000, n_eval=300, n_transfer=150, formats=("cc",), tok=tok)
prompts = [p["prompt"] for p in d.train_pairs if p["fmt"] == "cc"]
with open("/workspace/replay/qwen3b/prompts.jsonl", "w") as f:
    for p in prompts:
        f.write(json.dumps({"prompt": p}) + "\n")
print(f"wrote {len(prompts)} prompts")
PY
fi

# ---- 2. replay generation (resumable) ----
if [ ! -f $R/library.jsonl ]; then
  log "generating 1000 replay sequences"
  replay-kfac-ewc generate \
    --model Qwen/Qwen2.5-3B \
    --output $R/shard-000.jsonl \
    --num-sequences 1000 --seq-start 0 \
    --seeding-mode prompt --prompts $R/prompts.jsonl \
    --min-new-tokens 64 --max-new-tokens 384 \
    --device cuda:0
  log "merging"
  replay-kfac-ewc merge --inputs $R/shard-000.jsonl \
    --output $R/library.jsonl --heldout-fraction 0.1
fi

# ---- 3. K-FAC factors (attention-only, matches phase 6) ----
if [ ! -f $R/kfac/manifest.json ]; then
  log "estimating K-FAC factors"
  replay-kfac-ewc estimate \
    --model Qwen/Qwen2.5-3B \
    --corpus $R/library.jsonl \
    --output $R/kfac \
    --target q_proj --target k_proj --target v_proj --target o_proj \
    --batch-size 1 --device cuda:0 --placement auto \
    --checkpoint $R/kfac-accumulator.pt --checkpoint-every 200
  replay-kfac-ewc inspect --factors $R/kfac
fi

# ---- 4. calibration (also rebuilds cc Stage A caches on first run) ----
if [ ! -f $R/kfac/calibration.json ]; then
  log "calibrating (phase-6 banked reference: slope 0.979, ratio 0.69)"
  STAGE=calib python cc_stage2.py
fi

# ---- 5. cells ----
if [ ! -f /workspace/cc_stage2_srloo_guarded_history.json ]; then
  log "T1: guarded srloo (PESS=0.5 KLR=0.03 DPOP=1)"
  ARM=srloo MCOEF=0 RL_PESS=0.5 RL_KLR=0.03 RL_DPOP=1.0 EWC_KL=1.0 STEPS=100 \
    RUN_TAG=srloo_guarded python cc_stage2.py
fi
if [ ! -f /workspace/cc_stage2_jonly_lower_history.json ]; then
  log "T2: jonly_lower (J writes blocks <= L* only)"
  ARM=hybrid MCOEF=0 JONLY_LOW=1 EWC_KL=1.0 STEPS=100 \
    RUN_TAG=jonly_lower python cc_stage2.py
fi

# ---- 6. bank ----
cp -f /workspace/cc_stage2_srloo_guarded_history.json \
      /workspace/cc_stage2_jonly_lower_history.json \
      /workspace/cc_probe_curve.json results/ 2>/dev/null || true
cp -f $R/kfac/calibration.json results/cc_kfac_calibration_rebuild.json 2>/dev/null || true
git add results/ >/dev/null 2>&1
git commit -q -m "results: cc guarded-srloo + jonly_lower cells (rebuilt anchor artifacts)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>" || true
log "pipeline done"
touch /workspace/.cc_pipeline_done
