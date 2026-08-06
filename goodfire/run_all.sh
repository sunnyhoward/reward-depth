#!/usr/bin/env bash
# Fast-RLFR pipeline, in the order the experiment is gated on.
#
#   1. decodability curve      (no training, minutes)
#   2. run 1 -- oracle reward  (THE GATE: if this does not install BE, stop)
#   3. run 5 -- student-activation control (cheap, makes the frozen-copy design legible)
#   4. depth sweep at pooled reward
#   5. dense vs pooled at the best layer
#
# Each stage is a separate process so a stage can be rerun without redoing the others.
# `bash run_all.sh <stage>` runs one stage; no argument runs the lot.
set -euo pipefail
cd "$(dirname "$0")"

VENV=${GF_VENV:-/workspace/venv-goodfire}
source "$VENV/bin/activate"
[ -f /workspace/.env ] && { set -a; . /workspace/.env; set +a; }

STEPS=${STEPS:-60}
KL=${KL:-0.05}
LR=${LR:-1e-4}                             # 1e-5 moves the LoRA policy too little to matter:
SEED=${SEED:-0}                            # 60 steps at 1e-5 left held-out BE flat at base
BEST_L=${BEST_L:-12}
SWEEP=${SWEEP:-"4 8 12 16 20 24"}
RL="python gf_rl.py --steps $STEPS --kl $KL --lr $LR --seed $SEED"

stage_data () {   python gf_data.py --seed "$SEED"; }
stage_probe () {  python gf_probes.py --seed "$SEED"; }

# The control for a flat decodability curve: fit on one set of AE/BE word pairs, test on unseen
# ones, so a probe cannot score by memorising the word list. See gf_probes.axis_split.
stage_probe_axis () {
  python gf_probes.py --seed "$SEED" --axis-holdout 0.3 \
         --tag probes_axisho --out decodability_axisho.json
}

stage_oracle () {
  $RL --reward oracle --tag "run1_oracle"
  python gf_eval.py --tag "run1_oracle"
}

stage_student () {
  $RL --reward pooled --layer "$BEST_L" --read student --tag "run5_student_L${BEST_L}"
  python gf_eval.py --tag "run5_student_L${BEST_L}"
}

stage_depth () {
  for L in $SWEEP; do
    $RL --reward pooled --layer "$L" --tag "run4_pooled_L${L}"
    python gf_eval.py --tag "run4_pooled_L${L}"
  done
}

stage_dense () {
  $RL --reward dense --layer "$BEST_L" --tag "run3_dense_L${BEST_L}"
  python gf_eval.py --tag "run3_dense_L${BEST_L}"
}

stage_klsweep () {                          # light KL sweep, oracle reward
  for B in 0.01 0.05 0.1; do
    $RL --reward oracle --kl "$B" --tag "kl_${B}"
  done
}

stage_plots () { python gf_plots.py; }

case "${1:-all}" in
  data)       stage_data ;;
  probe)      stage_probe ;;
  probe-axis) stage_probe_axis ;;
  oracle)     stage_oracle ;;
  student)    stage_student ;;
  depth)      stage_depth ;;
  dense)      stage_dense ;;
  klsweep)    stage_klsweep ;;
  plots)      stage_plots ;;
  rest)       stage_probe_axis; stage_student; stage_depth; stage_dense; stage_plots ;;
  all)        stage_data; stage_probe; stage_probe_axis; stage_oracle; stage_student
              stage_depth; stage_dense; stage_plots ;;
  *)          echo "unknown stage: $1" >&2; exit 1 ;;
esac
