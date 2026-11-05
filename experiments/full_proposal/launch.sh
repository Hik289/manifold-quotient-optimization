#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PYTHON="${PYTHON:-python3}"
for worker in 0 1 2; do
  if test -f "worker_${worker}.pid" && kill -0 "$(cat "worker_${worker}.pid")" 2>/dev/null; then
    continue
  fi
  CUDA_VISIBLE_DEVICES="$worker" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
    nohup "$PYTHON" -u worker.py --worker "$worker" --total 3 --device cuda:0 \
    > "worker_${worker}.log" 2>&1 < /dev/null &
  echo "$!" > "worker_${worker}.pid"
done
