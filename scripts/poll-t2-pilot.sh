#!/usr/bin/env bash
# Poll the T2 pilot without touching it.
#
#   scripts/poll-t2-pilot.sh            # one progress report
#   scripts/poll-t2-pilot.sh --follow   # repeat every 60 s until the pilot ends
#   scripts/poll-t2-pilot.sh --json     # machine-readable progress
#
# It reads results/t2/pilot-state.json, the per-arm train.jsonl, the court logs
# and `docker ps`. It never writes to the run and never launches anything.
#
# What it prints, and why each line exists:
#   * the pilot status, revision and registered budget;
#   * per arm: state, last logged step, measured tokens/s, VRAM peak, loss and
#     an ETA computed from the arm's own measured rate;
#   * the live container and its last log lines;
#   * what is left to run, in order, and the exact command that resumes it;
#   * free disk, because the checkpoint footprint is the one hard constraint.
#
# To resume after a lost session:
#   1. scripts/poll-t2-pilot.sh
#   2. if an arm's container is still alive, `scripts/court wait <state file>`
#      finishes it and writes the receipt; the state file is
#      results/courts/pending/<container>.json
#   3. otherwise `scripts/run-t2-pilot.sh --resume`, which continues from the
#      recorded state and from the latest checkpoint of an interrupted arm.
set -Eeuo pipefail
IFS=$'\n\t'

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
repo_root=$(cd -- "$script_dir/.." && pwd -P)
cd "$repo_root"

follow=0
json=0
interval=60
while (($#)); do
  case "$1" in
    --follow) follow=1; shift ;;
    --json) json=1; shift ;;
    --interval) interval="$2"; shift 2 ;;
    -h | --help) sed -n '2,30p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) printf 'poll-t2-pilot: unknown argument %q\n' "$1" >&2; exit 64 ;;
  esac
done

report() {
  python3 scripts/t2_pilot.py progress ${json:+--json}
  if ((json)); then
    return 0
  fi

  local live
  live=$(docker ps --filter 'name=selvedge-' --format '{{.Names}}\t{{.Status}}' 2>/dev/null || true)
  echo
  if [[ -n "$live" ]]; then
    echo "live court containers:"
    printf '%s\n' "$live" | sed 's/^/  /'
    local name
    name=$(printf '%s\n' "$live" | head -1 | cut -f1)
    echo
    echo "last 6 log lines of $name:"
    docker logs --tail 6 "$name" 2>&1 | sed 's/^/  /' || true
  else
    echo "live court containers: none"
  fi

  if [[ -f results/t2/orchestrator.log ]]; then
    echo
    echo "orchestrator log (tail):"
    tail -6 results/t2/orchestrator.log | sed 's/^/  /'
  fi

  echo
  echo "disk:"
  df -h . | tail -1 | sed 's/^/  /'
  echo "  checkpoints: $(du -sh results/artifacts/train/t2 2>/dev/null | cut -f1 || echo none)"
  echo
  echo "if an arm container is alive:"
  echo "  scripts/court wait results/courts/pending/<container>.json"
  echo "otherwise:"
  echo "  scripts/run-t2-pilot.sh --resume"
  echo "once the pilot is done (status done):"
  echo "  scripts/finish-t2-pilot.sh    # render results/T2-PILOT.md, run the courts"
}

if ((follow)); then
  while :; do
    clear 2>/dev/null || true
    report
    status=$(python3 - <<'PY' 2>/dev/null || echo unknown
import json, os
path = "results/t2/pilot-state.json"
if not os.path.exists(path):
    print("unknown")
else:
    print(json.load(open(path, encoding="utf-8")).get("status", "unknown"))
PY
)
    case "$status" in
      done | failed | partial) echo; echo "pilot status: $status — stopping the watch"; exit 0 ;;
    esac
    sleep "$interval"
  done
fi

report
