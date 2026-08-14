#!/usr/bin/env bash
# DOES THE REPLAY ANCHOR DO ANYTHING?  C1 and P1, with and without it.
#
# THE MISCONCEPTION THIS SETTLES. It is natural to read the two-stage arms as "the ones with
# generative replay" and P1 as "plain DPO with only its own KL". That is not what the code does:
# pf_train.py's header says MODE=final carries "the same replay term", W_REPLAY defaults to 1, and
# NO probefix run script has ever set it. P0, P1, C0, C1 and D2 all carry the same output-side
# replay anchor. So replay cannot be what separates C1 from P1 -- the only differences are the
# stage-1 merge and the LoRA range.
#
# WHY IT MIGHT NEVERTHELESS BE DOING NOTHING. replay_batch() draws ONE sequence and scores at most
# REPLAY_TOK=16 next-token positions, against PREF_PAIRS=6 full-length preference pairs. Over 600
# steps that is ~9.6k scored replay tokens against a 1.32M-token bank. The L_t sweep on master
# (afdb146) flagged exactly this for the UF port -- "the replay term is smaller still ... that is
# the likeliest reason arm E (replay off) was indistinguishable from arm A" -- and arm E is the
# ONLY replay-off control anywhere in the repo. probefix has never run one.
#
# THE 2x2. Replay is the only thing that varies within each pair; everything else is held.
#
#   P1_r1   all blocks 0-31,  no stage 1,  W_REPLAY=1   <- the banked P1 config
#   P1_r0   all blocks 0-31,  no stage 1,  W_REPLAY=0
#   C1_r1   blocks 21-31, stage 1 merged,  W_REPLAY=1   <- the banked C1 config
#   C1_r0   blocks 21-31, stage 1 merged,  W_REPLAY=0
#
# BOTH with-replay controls are RETRAINED HERE rather than taken from the HF bank. The banked
# adapters were trained in another session's environment, and 0813 already had to throw away the
# 0811 D-arm comparison for exactly that reason ("spanned environments ... and was unusable").
# A within-run control costs one extra arm and removes the objection.
#
# Stage 1 for the C arms is the BANKED B4/ckpt300 -- an input to both C cells equally, so its
# provenance cannot confound the contrast the sweep is about.
#
# Scored afterwards on false_friend and style only: RESULTS_0814_ELICITATION.md shows the other
# three families never elicit their own item (culture 0.00 across four forms and three
# reframings), so they cannot measure this or anything else.
set -euo pipefail
cd "$(dirname "$0")"

set -a; . /workspace/.env; set +a          # HF_TOKEN; plain `source` does NOT export it
export HF_HOME=/workspace/.hf_home
# Absolute interpreter: this is launched detached (nohup/background), where the venv is NOT
# active and bare `python` does not exist on PATH -- same class of trap as NEXT_0810's
# "background jobs launch from /workspace, not the repo".
PY=${PY:-/venv/main/bin/python}
export SUP_MODEL=Qwen/Qwen3.5-4B
export SUP_BRIT=/workspace/reward-depth/supervisor/britishness/dosed/brit_dose20.jsonl
L=20; TOP=31; ROOT=${ROOT:-/workspace/probefix4b_replay}
STEPS=${STEPS:-600}
S1=${S1:-/workspace/.hf_home/hub/models--sunnyhoward--reward-depth-probefix4b/snapshots/3b55e6a54b017b522c841b8033807ae614bce53e/B4_probe_L20/ckpt300}
export PF_LAYER=$L SEED=${SEED:-0} EVAL_EVERY=50 CKPT_EVERY=100 PROBE_WARM=2048
mkdir -p "$ROOT"

[ -f /workspace/sup/replay_bank.pt ] || { echo "no replay bank -- run pf_make_bank.py first"; exit 1; }
[ -d "$S1" ] || { echo "no stage-1 checkpoint at $S1"; exit 1; }
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader || true

run () { local tag="$1"; shift
  [ -f "$ROOT/$tag/DONE" ] && { echo "== $tag done, skipping"; return; }
  echo "===== $tag ====="
  env "$@" OUT="$ROOT/$tag" "$PY" pf_train.py > "$ROOT/$tag.log" 2>&1 \
    && touch "$ROOT/$tag/DONE" || { echo "!! $tag FAILED, see $ROOT/$tag.log"; return 1; }
  tail -2 "$ROOT/$tag.log"; }

# replay-off arms first: if the anchor is inert they will match their partners, and that is the
# result. Running them first means a truncated sweep still answers the question for one pair.
run "P1_r0_noreplay" MODE=final LAMBDA=50 W_REPLAY=0 LORA_MIN=0        LORA_MAX=$TOP STEPS=$STEPS
run "P1_r1_replay"   MODE=final LAMBDA=50 W_REPLAY=1 LORA_MIN=0        LORA_MAX=$TOP STEPS=$STEPS
run "C1_r0_noreplay" MODE=final LAMBDA=50 W_REPLAY=0 INIT_MERGE="$S1" \
                     LORA_MIN=$((L + 1)) LORA_MAX=$TOP STEPS=$STEPS
run "C1_r1_replay"   MODE=final LAMBDA=50 W_REPLAY=1 INIT_MERGE="$S1" \
                     LORA_MIN=$((L + 1)) LORA_MAX=$TOP STEPS=$STEPS
echo REPLAYSWEEPDONE
