#!/usr/bin/env bash
# The A-vs-C1 comparison ported to Qwen3.5-4B, attaching at block 20.
#
# WHY THIS MODEL. The britishness release ships its text PRE-RENDERED and its manifest names
# `models/Qwen3.5-4B` as the template source, so this is the one 4B where the strings are correct
# by construction. Checked before launching: vocab identical to Qwen3.5-2B (248077, get_vocab()
# equal) so /workspace/sup/replay_bank.pt transfers unchanged, and the `<|im_start|>assistant\n`
# boundary is token-exact under the 4B tokenizer so the span masks stay right. Qwen3-4B (36 layers,
# different family) would have reintroduced the template-mismatch class of bug.
#
# WHY BLOCK 20. Measured, not assumed (curve4b_dose20.json): the guard column plateaus at 0.920
# from block 18 and peaks at 0.940 @ block 21, so 20 sits on the plateau one step off the max.
# The headline reason to run this at all is that the composite rule reads 0.92 here against the
# 2B's 0.68 -- if stage 1's discrimination failure was caused by shaping a rule that was only
# partly decodable at the attach point, it should shrink at 0.92.
#
# D IS INCLUDED DELIBERATELY. The largest correction the 2B run produced was D: at 300 steps it
# looked incapable of installing truth_dialect (0.36), at 600 it reaches 0.85. The surviving claim
# is "stage 1 ACCELERATES rather than ENABLES", and that claim cannot be tested without D.
set -eu
cd "$(dirname "$0")"
source /venv/main/bin/activate
export SUP_MODEL=Qwen/Qwen3.5-4B
export SUP_BRIT=/workspace/reward-depth/supervisor/britishness/dosed/brit_dose20.jsonl
L=20; TOP=31; ROOT=/workspace/probefix4b
export PF_LAYER=$L SEED=0 EVAL_EVERY=50 CKPT_EVERY=100 PROBE_WARM=2048
mkdir -p "$ROOT"

run () { local tag="$1"; shift
  [ -f "$ROOT/$tag/DONE" ] && { echo "== $tag done, skipping"; return; }
  echo "===== $tag ====="
  env "$@" OUT="$ROOT/$tag" python pf_train.py > "$ROOT/$tag.log" 2>&1 \
    && touch "$ROOT/$tag/DONE" || { echo "!! $tag FAILED, see $ROOT/$tag.log"; return 1; }
  tail -2 "$ROOT/$tag.log"; }

run "B4_probe_L${L}" MODE=probe  LORA_MIN=0 LORA_MAX=$TOP STEPS=300
run "C1_4b_s2_upper" MODE=final  INIT_MERGE="$ROOT/B4_probe_L${L}/ckpt300" \
                     LORA_MIN=$((L+1)) LORA_MAX=$TOP STEPS=300
run "A_4b_dpo_all"   MODE=final  LORA_MIN=0 LORA_MAX=$TOP STEPS=600
run "D_4b_dpo_upper" MODE=final  LORA_MIN=$((L+1)) LORA_MAX=$TOP STEPS=600
echo ALL4BDONE
