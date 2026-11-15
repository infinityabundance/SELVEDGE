#!/usr/bin/env bash
# Build the deterministic T1 training stream from the fetched corpus, then
# verify the lock and the stream hash. Offline; runs in the oracle court.
#
#   scripts/build-train-corpus.sh
#
# Writes results/artifacts/train/train.bin (gitignored) and a tracked receipt
# under results/train/.
set -Eeuo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
repo_root=$(cd -- "$script_dir/.." && pwd -P)
cd "$repo_root"

SELVEDGE_MEM=24g SELVEDGE_TMP=8g \
  scripts/court oracle -- python3 oracle/train/corpus.py build
SELVEDGE_MEM=8g \
  scripts/court oracle -- python3 oracle/train/corpus.py verify
