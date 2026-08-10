#!/usr/bin/env bash
# STAGE 2 — "train upwards": the aligned EAGLE readout teaches the full network.
#
# WHY THIS IS THE ARM THAT DECIDES THE DEPTH QUESTION, and why stage 1 alone did not.
#
# Stage 1 scores an arm through its own EAGLE head. Head fidelity rises with depth BY
# CONSTRUCTION (block 5 / 10 / 21 -> agreement .361 / .404 / .812), so a stage-1-only comparison
# structurally favours the deep arm: it is scored through the better decoder. The stage-1 result
# (C ~ A << B, install tracking head agreement rather than probe accuracy) is therefore a fair
# measurement of stage 1 and NOT a fair test of "attach where the preference is decodable",
# because the recipe's own answer to a weak shallow readout is this stage, not a better head.
#
# Stage 2 retrains the upper stack to EXPRESS what the head at L reads. If upward distillation is
# what makes a shallow attach viable, arm A should close most of its 0.19 gap to arm B here. If
# instead the L10 teacher is too weak — KL 2.82 against L21's 0.34 — this is the POISONED TEACHER
# failure already on record (NOTE.md §11-12: a head that could not compute the task overwrote the
# top token at 86.5% of answer positions), and arm A should get WORSE, not better.
#
# Both outcomes are informative and they are opposite, which is what makes it worth the GPU.
#
# S2_FROM_S1=1: the student is initialised from the stage-1 merge, so blocks 0..L keep the install
# and stage 2 only propagates it upward. sup_train.py's own note calls this the reading of "train
# upwards" that keeps the two-stage story intact; the default (fresh base student) instead asks
# the upper blocks to re-encode the preference from scratch against unaligned lower layers.
#
# NOTE the asymmetry that no arm here removes: stage 2 adapts blocks L+1..23, so arm A gets 13
# blocks to work with and arm B gets 2. Report it beside the numbers rather than pretending the
# arms are matched.
set -euo pipefail
cd "$(dirname "$0")"
source /venv/main/bin/activate
set -a; . /workspace/.env 2>/dev/null || true; set +a

export SUP_BRIT="$(pwd)/uf_release/uf.jsonl"
export MAX_LEN=512 STAGE=2 S2_FROM_S1=1
export STEPS=${STEPS:-400} LR=${LR:-1e-4}
export W_PREF=1 W_KFAC=0 W_REPLAY=${W_REPLAY:-1}
export PREF_PAIRS=${PREF_PAIRS:-6} REPLAY_TOK=${REPLAY_TOK:-16}
export EVAL_EVERY=${EVAL_EVERY:-100} EVAL_N=${EVAL_N:-128} CKPT_EVERY=${CKPT_EVERY:-200}

s2 () {  # tag, read layer, stage-1 arm dir
  local tag=$1 layer=$2
  local s1=$3
  local out=/workspace/uf_s2_$tag
  if [ -f "$out/history.json" ] && [ -d "$out/ckpt${STEPS}" ]; then
    echo "[s2:$tag] done, skipping"; return
  fi
  echo "=== stage2 $tag: teacher = stage-1 arm at block $layer, student adapts $((layer+1))..23 ==="
  SUP_LAYER=$layer S1_CKPT=$s1 RUN_TAG_DIR=$out \
    python sup_train.py 2>&1 | tee /workspace/uf_s2_$tag.log
  mkdir -p ../results/uf_depth
  cp "$out/history.json" "../results/uf_depth/history_s2_$tag.json"
}

s2 A10 10 /workspace/uf_A_read10_lora10/ckpt400
s2 B21 21 /workspace/uf_B_read21_lora21/ckpt400
echo STAGE2_ARMS_DONE

# Evaluate at the FINAL OUTPUT — that is the whole point of stage 2, and the EAGLE columns are
# meaningless for it (the head is the teacher here, not the artifact).
for t in A10:10 B21:21; do
  tag=${t%%:*}; layer=${t##*:}
  [ -f "../results/uf_depth/eval_s2_$tag.json" ] && continue
  SUP_LAYER=$layer OUT_JSON=/workspace/uf_eval_s2_$tag.json \
    python sup_eval_pref.py /workspace/uf_s2_$tag/ckpt${STEPS} 2>&1 | tee /workspace/uf_eval_s2_$tag.log
  cp /workspace/uf_eval_s2_$tag.json "../results/uf_depth/eval_s2_$tag.json"
done
echo STAGE2_ALL_DONE
