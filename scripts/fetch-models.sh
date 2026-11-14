#!/usr/bin/env bash
# Fetch or verify the pinned teacher checkpoints.
#
#   scripts/fetch-models.sh              # idempotent fetch into the HF cache
#   scripts/fetch-models.sh --verify-only  # offline verification against the lock
#
# The fetch runs inside the oracle court with network enabled. The cache lives
# on the host at ~/.cache/selvedge/hf and is mounted read-write at /hf-cache, so
# re-running never re-downloads a verified file. Every later court is offline:
# the teacher is loaded from the cached directory path, not from a repo id.
set -Eeuo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
repo_root=$(cd -- "$script_dir/.." && pwd -P)
cd "$repo_root"

verify_only=false
for arg in "$@"; do
  case "$arg" in
    --verify-only) verify_only=true ;;
    -h | --help)
      sed -n '2,12p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *)
      printf 'fetch-models: unknown argument %q\n' "$arg" >&2
      exit 64
      ;;
  esac
done

declare -a flags=()
if [[ "$verify_only" == "true" ]]; then
  exec scripts/court oracle -- python3 oracle/fetch_models.py --verify-only "$@"
fi

flags+=(--net)
exec scripts/court "${flags[@]}" oracle -- python3 oracle/fetch_models.py "$@"
