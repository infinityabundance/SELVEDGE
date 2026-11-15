#!/usr/bin/env bash
# Build and verify the shared teacher-target cache (GPU court).
#
#   scripts/build-teacher-targets.sh                # registered budget (2^27 tokens)
#   scripts/build-teacher-targets.sh --tokens 2097152   # smoke-sized cache
#
# The teacher is run once; every arm trains on the same cached targets. The
# verification re-forwards a registered sample of chunks and fails closed on a
# mismatch.
set -Eeuo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
repo_root=$(cd -- "$script_dir/.." && pwd -P)
cd "$repo_root"

tokens=134217728
while (($#)); do
  case "$1" in
    --tokens)
      tokens="$2"
      shift 2
      ;;
    -h | --help)
      sed -n '2,10p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *)
      printf 'build-teacher-targets: unknown argument %q\n' "$1" >&2
      exit 64
      ;;
  esac
done

SELVEDGE_MEM=20g SELVEDGE_TMP=8g \
  scripts/court --gpu oracle -- python3 oracle/train/teacher_cache.py build --tokens "$tokens"
SELVEDGE_MEM=20g \
  scripts/court --gpu oracle -- python3 oracle/train/teacher_cache.py verify
