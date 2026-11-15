#!/usr/bin/env bash
# Finish the T2 pilot's paperwork after the orchestrator stops.
#
#   scripts/finish-t2-pilot.sh
#
# The orchestrator (`scripts/run-t2-pilot.sh`) runs the arms, the evaluations,
# the P2.4 check and the decision analysis. What it deliberately does not do is
# touch tracked documents, because those are published artifacts with their own
# court. This script does the remaining, mechanical part:
#
#   1. refuse to run unless the pilot reached `done` and a decision receipt
#      exists (a partial run cannot be decided, T2.10);
#   2. render `results/T2-PILOT.md` and its `docs/perf-numbers.json` blocks from
#      the decision receipt with `scripts/t2_report.py --write`;
#   3. re-render and byte-compare the document (`--check`);
#   4. stage the receipts and the document, then run the two courts that gate a
#      push: `commit_identity` and `perf_numbers`.
#
# It never commits. Staging and a suggested commit message are printed at the
# end; the operator (or the parent session) decides.
set -Eeuo pipefail
IFS=$'\n\t'

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
repo_root=$(cd -- "$script_dir/.." && pwd -P)
cd "$repo_root"

say() { printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"; }

state="results/t2/pilot-state.json"
[[ -f "$state" ]] || { say "no $state; nothing to finish"; exit 1; }

status=$(python3 -c "
import json; print(json.load(open('$state', encoding='utf-8')).get('status', 'unknown'))")
if [[ "$status" != "done" ]]; then
  say "pilot status is '$status', not 'done'; refusing to publish a partial decision"
  python3 scripts/t2_pilot.py progress || true
  exit 1
fi

receipt=$(ls -1t results/t2/decision-*.json 2>/dev/null | head -1 || true)
[[ -n "$receipt" ]] || { say "no decision receipt in results/t2/"; exit 1; }
say "decision receipt: $receipt"

say "rendering results/T2-PILOT.md"
python3 scripts/t2_report.py --write --receipt "$receipt"

say "re-rendering and comparing"
python3 scripts/t2_report.py --check --receipt "$receipt"

say "staging receipts, document and manifest"
git add results/t2 results/eval "$receipt" results/T2-PILOT.md docs/perf-numbers.json 2>/dev/null || true

say "court: commit_identity"
scripts/court commit_identity

say "court: perf_numbers"
scripts/court perf_numbers

cat <<'EOF'

Both courts passed. The suggested commit message:

  T2: publish the five-arm pilot result

  Every number in results/T2-PILOT.md is rendered from the decision receipt by
  scripts/t2_report.py; --check re-renders and compares the whole document, and
  the verdict numbers are published as generated regions checked by the
  perf-numbers court.

Commit (the identity is pinned locally), then push:

  git commit -F <message file>
  git push origin staging
EOF
