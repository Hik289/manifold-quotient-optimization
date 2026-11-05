#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PYTHON="${PYTHON:-python3}"
DATA_ROOT="${QUOTIENT_DATA:-$(cd .. && pwd)/data}"
for slot in 0 1 2; do
  if test -f "worker_${slot}.pid" && kill -0 "$(cat "worker_${slot}.pid")" 2>/dev/null; then
    continue
  fi
  CUDA_VISIBLE_DEVICES="$slot" FOCUS_DEVICE=cuda:0 QUOTIENT_DATA="$DATA_ROOT" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    nohup "$PYTHON" -u campaign.py --slot "$slot" \
    > "worker_${slot}.log" 2>&1 < /dev/null &
  echo "$!" > "worker_${slot}.pid"
done
