#!/usr/bin/env bash
# Negative-case self-test for the generated status blocks.
#
# Each case breaks one thing in a disposable copy of the two documents and the
# receipts under /tmp, then asserts that `status_consistency.py --check` fails
# closed with a specific diagnostic. The repository is never mutated.
#
# The cases are the deliverable's own examples, inverted:
#   * a document that says a run is training when every arm is evaluated,
#   * a document that says there is no verdict when one exists,
#   * a document that says no arm ran the capability suite when five did,
#   * a document that says an arm has not been evaluated when it has,
#   * a status block hand-edited away from the receipts,
#   * a status block whose sentinel was deleted.
#
# Run through the court:
#   scripts/court python scripts/courts/status_consistency_negative_cases.sh
set -Eeuo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
repo_root=$(cd -- "$script_dir/../.." && pwd -P)

work=$(mktemp -d /tmp/selvedge-status-negative.XXXXXX)
trap 'rm -rf "$work"' EXIT

mkdir -p "$work/results/t2" "$work/results/capability/runs" \
  "$work/results/stage2/runs/full-20m-math"
cp "$repo_root/README.md" "$work/README.md"
cp "$repo_root/results/REPORT.md" "$work/results/REPORT.md"
cp "$repo_root/results/t2/pilot-state.json" "$work/results/t2/pilot-state.json"
cp "$repo_root/results/t2-verdict.json" "$work/results/t2-verdict.json"
cp "$repo_root/results/capability/runs/t2-comparison.json" \
  "$work/results/capability/runs/t2-comparison.json"
copy_ws19() {
  for name in launch.json manifest.json log.jsonl; do
    if [[ -f "$repo_root/results/stage2/runs/full-20m-math/$name" ]]; then
      cp "$repo_root/results/stage2/runs/full-20m-math/$name" \
        "$work/results/stage2/runs/full-20m-math/$name"
    fi
  done
}

# The live bar is named by its own launch receipt. The documents now name its
# milestone scores, so a baseline copy without those receipts would render a
# different (fallback) block and every case would fail as stale. Copy the run
# and every milestone receipt it publishes.
copy_live_bar() {
  local run="results/stage2/runs/s50-bar-20m-lr1e-5"
  local eval_root="results/stage2/eval/s50-bar-20m-lr1e-5"
  if [[ -d "$repo_root/$run" ]]; then
    mkdir -p "$work/$run"
    for name in launch.json manifest.json log.jsonl; do
      [[ -f "$repo_root/$run/$name" ]] && cp "$repo_root/$run/$name" "$work/$run/$name"
    done
  fi
  if [[ -d "$repo_root/$eval_root" ]]; then
    local receipt
    for receipt in "$repo_root/$eval_root"/m*/eval.json; do
      [[ -f "$receipt" ]] || continue
      mkdir -p "$work/$(dirname "${receipt#$repo_root/}")"
      cp "$receipt" "$work/${receipt#$repo_root/}"
    done
  fi
  # Eligibility is derived from the invalidation receipts, so a baseline copy
  # without them renders a different eligibility line and every case fails as
  # stale. Copy the master record and the per-directory markers verbatim.
  local path
  for path in "$repo_root"/results/stage2/*invalidation*.json; do
    [[ -f "$path" ]] || continue
    mkdir -p "$work/$(dirname "${path#$repo_root/}")"
    cp "$path" "$work/${path#$repo_root/}"
  done
  while IFS= read -r path; do
    [[ -n "$path" ]] || continue
    mkdir -p "$work/$(dirname "${path#$repo_root/}")"
    cp "$path" "$work/${path#$repo_root/}"
  done < <(find "$repo_root/results/stage2" -name INVALIDATED.json 2>/dev/null)
}
copy_live_bar

failures=0

reset_copy() {
  cp "$repo_root/README.md" "$work/README.md"
  cp "$repo_root/results/REPORT.md" "$work/results/REPORT.md"
  cp "$repo_root/results/t2/pilot-state.json" "$work/results/t2/pilot-state.json"
  cp "$repo_root/results/t2-verdict.json" "$work/results/t2-verdict.json"
  cp "$repo_root/results/capability/runs/t2-comparison.json" \
    "$work/results/capability/runs/t2-comparison.json"
  copy_ws19
  copy_live_bar
}

check_case() {
  local label="$1" needle="$2"
  local out rc
  set +e
  out=$(python3 "$repo_root/scripts/status_consistency.py" --check --repo "$work" 2>&1)
  rc=$?
  set -e
  if [[ $rc -eq 0 ]]; then
    printf 'FAIL %-34s expected a non-zero exit, got 0\n' "$label"
    failures=$((failures + 1))
    return
  fi
  if [[ "$out" != *"$needle"* ]]; then
    printf 'FAIL %-34s missing diagnostic %q\n' "$label" "$needle"
    printf '     output was: %s\n' "$out"
    failures=$((failures + 1))
    return
  fi
  printf 'PASS %-34s %s\n' "$label" "$needle"
}

# Baseline: the untouched copy must pass, otherwise every later case is noise.
if python3 "$repo_root/scripts/status_consistency.py" --check --repo "$work" >/dev/null 2>&1; then
  printf 'PASS %-34s baseline copy checks clean\n' "baseline"
else
  printf 'FAIL %-34s baseline copy does not check clean\n' "baseline"
  python3 "$repo_root/scripts/status_consistency.py" --check --repo "$work" || true
  failures=$((failures + 1))
fi

# A. a run claimed to be training while every arm is evaluated
cat >>"$work/README.md" <<'EOF'

`H*` is training unattended on the GPU and owns the device.
EOF
check_case "claims-a-run-is-training" "no-run-is-training"
reset_copy

# B. no verdict claimed while a verdict receipt exists
cat >>"$work/README.md" <<'EOF'

There is no T2 verdict at this revision, because the decision rule needs all
five summaries.
EOF
check_case "claims-no-verdict" "verdict-exists"
reset_copy

# C. the capability suite claimed never to have run
cat >>"$work/results/REPORT.md" <<'EOF'

No arm has run the capability suite, so no verified-task number exists.
EOF
check_case "claims-suite-never-ran" "capability-suite-has-run"
reset_copy

# D. an evaluated arm claimed not to have been evaluated
cat >>"$work/results/REPORT.md" <<'EOF'

Four of the five arms are scored and X has not been evaluated.
EOF
check_case "claims-arm-unevaluated" "all-arms-evaluated"
reset_copy

# E. a status block hand-edited away from its receipts
python3 - "$work" <<'PY'
import pathlib, sys
root = pathlib.Path(sys.argv[1])
doc = root / "README.md"
text = doc.read_text()
begin = "<!-- BEGIN GENERATED: readme-status -->\n"
start = text.index(begin) + len(begin)
end = text.index("<!-- END GENERATED: readme-status -->", start)
doc.write_text(text[:start] + "Every arm is training and there is no verdict.\n" + text[end:])
PY
check_case "hand-edited-status-block" "stale status block"
reset_copy

# F. the receipt moves and the block is not regenerated
python3 - "$work" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
path = root / "results/t2/pilot-state.json"
pilot = json.loads(path.read_text())
pilot["arms"]["H*"]["status"] = "training"
path.write_text(json.dumps(pilot, indent=2) + "\n")
PY
check_case "receipt-moved-under-block" "stale status block"
reset_copy

# G. the owned sentinel is deleted, so the block cannot be checked at all
python3 - "$work" <<'PY'
import pathlib, sys
root = pathlib.Path(sys.argv[1])
doc = root / "results/REPORT.md"
text = doc.read_text()
text = text.replace("<!-- BEGIN GENERATED: report-t2-arm-state -->\n", "")
text = text.replace("<!-- END GENERATED: report-t2-arm-state -->\n", "")
doc.write_text(text)
PY
check_case "owned-sentinel-deleted" "is missing"
reset_copy

# H. the contradiction rule must not fire on a historical, past-tense sentence
cat >>"$work/results/REPORT.md" <<'EOF'

At the S33 poll `H*` had not finished training; it was still training then, and
the pilot had no verdict until the five summaries existed.
EOF
if python3 "$repo_root/scripts/status_consistency.py" --check --repo "$work" >/dev/null 2>&1; then
  printf 'PASS %-34s past-tense history is not a contradiction\n' "history-allowed"
else
  printf 'FAIL %-34s a past-tense sentence was rejected\n' "history-allowed"
  failures=$((failures + 1))
fi
reset_copy

if ((failures > 0)); then
  printf 'status_consistency negative cases: %d failure(s)\n' "$failures" >&2
  exit 1
fi
printf 'status_consistency negative cases: all cases failed closed as required\n'
