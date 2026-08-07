#!/bin/bash
# Family A sweep across the model ladder.
#
# Concurrency is capped at 2 (NEXT_0806.md:72-74). These fits are CPU-bound (the Bayesian head
# and the AntisymMLP both run on CPU tensors), so two at a time on a 64-core box is well within
# budget and leaves the GPU free for distillation to run alongside.
set -u
cd "$(dirname "$0")/.."
source /venv/main/bin/activate
[ -f /workspace/.env ] && source /workspace/.env

# Concurrency gate. NOT `while [ $(jobs -rp | wc -l) -ge N ]` -- command substitution forks a
# subshell, `jobs` there reports the SUBSHELL's (empty) job table, the test never fires, and every
# job launches at once. Found the hard way: a MAXJOBS=2 run spawned all 16. `wait -n` blocks in
# the parent shell, where the job table actually lives.
MAXJOBS=${MAXJOBS:-2}
running=0
# Cap per-job CPU threads. Torch defaults to all 64 cores per process, so N concurrent fits
# oversubscribe the box by Nx and every job slows down together -- and, worse, they starve the
# GPU distillation job of the cores it needs to feed the card.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export MKL_NUM_THREADS=$OMP_NUM_THREADS
for m in "$@"; do
  for d in styc brit_language brit_culture brit_truth; do
    if [ "$running" -ge "$MAXJOBS" ]; then wait -n; running=$((running-1)); fi
    echo "=== $m x $d ==="
    python decodability/dec_scalar.py "$m" "$d" chat \
      > "/workspace/dec_cache/log_scalar_${m}_${d}.log" 2>&1 &
    running=$((running+1))
  done
done
wait
echo "=== family A sweep done ==="
