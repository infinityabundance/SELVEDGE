#!/usr/bin/env bash
# Emit the T2 arm configs, admit each one through the Rust ledger, and verify
# the records against the pre-registered design counts.
#
#   scripts/train-admission-t2.sh
#
# The configs are written by the Python court (no model access is needed); the
# admission records are written by the Rust court, which is the only place the
# parameter ceiling is decided. The check runs in the Python court and compares
# the Rust union with the design count and an independent Python mirror.
set -Eeuo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
repo_root=$(cd -- "$script_dir/.." && pwd -P)
cd "$repo_root"

scripts/court python -- python3 oracle/train/arms.py emit --profile t2
scripts/court rust -- /work/scripts/admission-loop-t2.sh
scripts/court python -- python3 oracle/train/arms.py check --profile t2
