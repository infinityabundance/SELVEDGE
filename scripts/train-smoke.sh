#!/usr/bin/env bash
# T1 smoke tests: A, B and C train 20 steps, checkpoint, resume, and evaluate on
# 5 problems each, inside the oracle court with the GPU. Writes a receipt under
# results/train/.
#
#   scripts/train-smoke.sh [--arms A,B,C] [--steps 20]
set -Eeuo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
repo_root=$(cd -- "$script_dir/.." && pwd -P)
cd "$repo_root"

SELVEDGE_MEM=28g SELVEDGE_TMP=8g \
  exec scripts/court --gpu oracle -- python3 oracle/train/smoke.py "$@"
