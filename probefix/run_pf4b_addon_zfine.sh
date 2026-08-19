#!/usr/bin/env bash
# The finer z grid between 0.5 and 0.75, named in RESULTS_0818_ZSWEEP.md §2 as the obvious next
# point and not run when that session ended.
#
# WHERE IT LEFT OFF. addon_L20 (PREF=dpo, 2560 params, frozen model) is level with plain DPOP on
# false_friend at BOTH z=0.5 and z=0.75 (68.4 either way), so the lexical family saturates by 0.5
# and costs nothing there (coherence 77.0/79.4 vs base 77.4/78.0). style is the family still
# moving: 51.0 at z=0.5, 69.3 at z=0.75, with coherence falling 79.4 -> 72.2. The pre-registered
# bar was style >= 66 at coherence >= 77; z=0.75 met the install half and missed coherence by ~5.
# The question this grid answers is whether the miss is a knee or a cliff -- i.e. whether some z in
# between holds style at DPOP's 66.0 while coherence is still in the base band.
#
# NO TRAINING HAPPENS HERE. A_L is trained once and z scales it at generation, so each point is one
# generation pass. results/probefix4b_addon_dpo/addon_L20.pt is COMMITTED (12 KB) and pf_addon.py
# loads it when the checkpoint exists, so this reproduces from a fresh clone with no GPU training
# and no HF adapter bank -- unlike every LoRA arm in this project, whose adapters died with the
# 08-18 box.
#
# THE PROMPT DRAW IS THE BANKED ONE. SEED=0 with N_PER_FAM=48 replays pf_addon.py's own selection
# (random.Random(0) over the fixed family order), so the new cells land on the SAME 48 prompts per
# family as base / P1_r1 / z0.5 / z0.75 in results/probefix_cjudge_zsweep, and every contrast can be
# paired item-by-item rather than compared as group means.
#
# This is also the addon runner that never existed: the 0818 cells were launched by inline env vars
# (LAYERS=4,12,20,28 PREF=dpo, then Z_GEN=0.25,0.5,0.75), which is why no script in probefix/
# reproduces the result RESULTS_0818_ZSWEEP.md reports.
set -eu
cd "$(dirname "$0")"

set -a; . /workspace/.env; set +a
export HF_HOME=/workspace/.hf_home
PY=${PY:-/venv/main/bin/python}                 # NOT a bare `python`: NEXT_0810's trap, and the
                                                # one that killed run_pf4b_replay.sh's first launch
export SUP_MODEL=Qwen/Qwen3.5-4B
export SUP_BRIT=/workspace/reward-depth/supervisor/britishness/dosed/brit_dose20.jsonl

OUT=${OUT:-/workspace/reward-depth/results/probefix4b_addon_dpo}
[ -f "$OUT/addon_L20.pt" ] || { echo "no banked A_L at $OUT/addon_L20.pt"; exit 1; }

echo "===== generating z=0.55,0.6,0.65,0.7 at L20 (no training: A_L is banked) ====="
env LAYERS=20 PREF=dpo Z_GEN=0.55,0.6,0.65,0.7 SEED=0 N_PER_FAM=48 \
    FAMS=false_friend,style OUT="$OUT" "$PY" pf_addon.py

echo "===== leakage (--all, so the banked scan stays a complete one) ====="
# NOT piped to head: pf_leakage writes leakage_scan.json AFTER its print loop, so a
# SIGPIPE from head kills the write and silently leaves the banked scan stale.
"$PY" pf_leakage.py --all > /workspace/leakage_all.txt && tail -5 /workspace/leakage_all.txt
echo ZFINEDONE
