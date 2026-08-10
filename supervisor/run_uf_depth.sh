#!/usr/bin/env bash
# The UF port of the supervisor recipe, with the attach layer moved to the probe elbow.
#
# WHY FOUR ARMS AND NOT TWO. The ask is "attach at L* instead of deep", which is arms A and B.
# On their own those two cannot separate three things that all move together when LAYER moves:
#   (i)   read depth            — what the DPO gradient reads
#   (ii)  write range           — LoRA covers 0..LAYER, so a shallower read means fewer parameters
#   (iii) readout competence    — an EAGLE head at block 5 reconstructs the output distribution
#                                 far worse than one at block 21, BY CONSTRUCTION
#                                 (results_0805 §2/§4; NEXT_0809 "the EAGLE attach sweep does not
#                                 substitute for it")
# C and D are what make the comparison readable:
#   C (below the elbow) discriminates (i) from (iii). Probe accuracy PLATEAUS from block 10
#     (0.740 @ block 5 -> 0.795 @ 10 -> 0.784 @ 21) while head competence rises MONOTONICALLY with
#     depth. So: install tracking decodability predicts C < A ~ B; install tracking readout
#     competence predicts C < A < B. The two hypotheses disagree on exactly one comparison, A vs B.
#   D was designed as "read at the elbow, adapt 0..21, so D vs B is read depth at matched parameter
#     count". THAT IS WRONG and the file should not be read as claiming it: l_pref is read from
#     h_LAYER, which depends on blocks 0..LAYER only, so blocks above the read point are outside
#     the loss's graph and get ZERO preference gradient regardless of LORA_MAX (measured: blocks
#     11/15/21 return None from torch.autograd.grad on a read-10/lora-0..21 model). D's upper
#     blocks move under the REPLAY term alone. Read depth and preference-write range are one
#     variable in this recipe. D is kept as the narrower control it really is: does letting the
#     upper stack move under replay change the install?
#
# Attach layers are in sup_train's convention: LAYER = the block whose OUTPUT is read. The
# decodability sweep indexes read points as 0=embeddings, so its L* = 11 is block 10 here.
# Curve (last/linear, results/decodability/scalar_qwen3.5-2b_uf_sup_chat.json):
#   block  4 -> 0.752 | block 5 -> 0.749 | block 10 -> 0.795 | block 14 -> 0.800 (peak)
#   block 21 -> 0.784 | block 23 (top) -> 0.779 | length-only floor 0.623 | lexical floor 0.644
set -euo pipefail
cd "$(dirname "$0")"
source /venv/main/bin/activate
set -a; . /workspace/.env 2>/dev/null || true; set +a

export SUP_BRIT="$(pwd)/uf_release/uf.jsonl"
export MAX_LEN=512
export STEPS=${STEPS:-400} LR=${LR:-1e-4} BETA=${BETA:-0.1}
export PREF_PAIRS=${PREF_PAIRS:-6} REPLAY_TOK=${REPLAY_TOK:-16}
export W_PREF=1 W_KFAC=0 W_REPLAY=${W_REPLAY:-1}     # 1:0:1 — 08-09 §5a: K-FAC is not separable
export EVAL_EVERY=${EVAL_EVERY:-50} EVAL_N=${EVAL_N:-128} CKPT_EVERY=${CKPT_EVERY:-100}
export STAGE=1

run () {  # name, read layer, lora_max
  local tag=$1 layer=$2 lmax=$3
  local out=/workspace/uf_$tag
  if [ -f "$out/history.json" ] && [ -d "$out/ckpt${STEPS}" ]; then
    echo "[$tag] done, skipping"; return
  fi
  # This chain is started while sup_prepare.py is still distilling the deeper heads, so an arm can
  # reach its turn before its readout exists. sup_train.py would assert; wait instead.
  local head=/workspace/sup/head_tf_L${layer}.pt
  while [ ! -f "$head" ]; do echo "[$tag] waiting for $head"; sleep 60; done
  echo "=== arm $tag: read block $layer, LoRA 0..$lmax ==="
  SUP_LAYER=$layer LORA_MAX=$lmax RUN_TAG_DIR=$out \
    python sup_train.py 2>&1 | tee /workspace/uf_train_$tag.log
  # workspace_is_volume is false on these boxes (STATE.md:216) — bank the history at the end of
  # every RUN, not at the end of the session.
  mkdir -p ../results/uf_depth && cp "$out/history.json" "../results/uf_depth/history_$tag.json"
}

run A_read10_lora10 10 10
run B_read21_lora21 21 21
run C_read5_lora5    5  5
run D_read10_lora21 10 21

# E — the matched replay control, and it is not optional. On britishness replay bought GUARD
# stability and cost about half the dialect install (results_0809 §3). UF has no guard: there is
# no truth-vs-preference row here for replay to protect, so its role on this dataset is untested
# and it may be nothing but a damper on the install. Every arm above carries the term, so without
# this cell a difference between arms cannot be attributed to depth rather than to how replay
# interacts with depth. 08-09's attribution existed ONLY because the matched control ran.
W_REPLAY=0 run E_read10_lora10_noreplay 10 10
echo ALL_ARMS_DONE
