#!/usr/bin/env bash
# Fetch the pinned T1 training corpus. Explicit network step; idempotent and
# hash-verified against the Hugging Face LFS object ids.
#
#   scripts/fetch-train-corpus.sh
#
# The corpus text lands in the gitignored results/artifacts/train-corpus cache.
# No corpus text is committed. Every later court is offline.
set -Eeuo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
repo_root=$(cd -- "$script_dir/.." && pwd -P)
cd "$repo_root"

SELVEDGE_MEM=16g SELVEDGE_TMP=8g \
  exec scripts/court --net oracle -- python3 oracle/train/corpus.py fetch
