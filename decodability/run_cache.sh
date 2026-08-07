#!/bin/bash
# Cache activations for the remaining models, SEQUENTIALLY.
#
# Sequential is deliberate: NEXT_0806.md:72-74 records that three concurrent jobs OOM the 95 GiB
# card and that an earlier three-way overlap silently killed two arms mid-run. Extraction is a
# single forward pass per batch, so serialising costs little and removes the failure mode.
set -u
cd "$(dirname "$0")/.."
source /venv/main/bin/activate
[ -f /workspace/.env ] && source /workspace/.env

for m in "$@"; do
  echo "=== caching $m ==="
  python decodability/dec_cache.py "$m" all || echo "FAILED: $m"
done
echo "=== cache sweep done ==="
