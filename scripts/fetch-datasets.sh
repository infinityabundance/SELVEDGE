#!/usr/bin/env bash
# Fetch or verify the pinned evaluation datasets.
#
#   scripts/fetch-datasets.sh              # idempotent fetch into the gitignored cache
#   scripts/fetch-datasets.sh --verify-only  # offline verification against the lock
#
# The fetch runs inside the oracle court with network enabled. The cache lives
# under results/artifacts/datasets/ (gitignored), so re-running never
# re-downloads a verified file. Every later court is offline and reads the
# cached files by path.
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
      printf 'fetch-datasets: unknown argument %q\n' "$arg" >&2
      exit 64
      ;;
  esac
done

if [[ "$verify_only" == "true" ]]; then
  exec scripts/court oracle -- python3 oracle/fetch_datasets.py --verify-only
fi

exec scripts/court --net oracle -- python3 oracle/fetch_datasets.py
