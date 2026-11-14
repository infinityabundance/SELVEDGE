#!/usr/bin/env bash
# Negative-case self-test for the generated-number machinery.
#
# Each case breaks one thing in a disposable copy of the manifest, the document
# and the receipts under /tmp, then asserts that `perf_numbers.py --check` fails
# closed with a specific diagnostic. The repository is never mutated.
#
# Run through the Python court:
#   scripts/court python scripts/courts/perf_numbers_negative_cases.sh
set -Eeuo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
repo_root=$(cd -- "$script_dir/../.." && pwd -P)

work=$(mktemp -d /tmp/selvedge-pn-negative.XXXXXX)
trap 'rm -rf "$work"' EXIT

mkdir -p "$work/docs"
cp "$repo_root/docs/perf-numbers.json" "$work/docs/perf-numbers.json"
cp "$repo_root/docs/PERF-NUMBERS.md" "$work/docs/PERF-NUMBERS.md"
cp -a "$repo_root/results" "$work/results"

failures=0
check_case() {
  local label="$1" needle="$2"
  local out rc
  set +e
  out=$(python3 "$repo_root/scripts/perf_numbers.py" --check --repo "$work" 2>&1)
  rc=$?
  set -e
  if [[ $rc -eq 0 ]]; then
    printf 'FAIL %-32s expected a non-zero exit, got 0\n' "$label"
    failures=$((failures + 1))
    return
  fi
  if [[ "$out" != *"$needle"* ]]; then
    printf 'FAIL %-32s missing diagnostic %q\n' "$label" "$needle"
    printf '     output was: %s\n' "$out"
    failures=$((failures + 1))
    return
  fi
  printf 'PASS %-32s %s\n' "$label" "$needle"
}

reset_copy() {
  cp "$repo_root/docs/perf-numbers.json" "$work/docs/perf-numbers.json"
  cp "$repo_root/docs/PERF-NUMBERS.md" "$work/docs/PERF-NUMBERS.md"
  rm -f "$work/docs/ORPHAN-TEST.md"
}

# Baseline: the untouched copy must pass, otherwise every later case is noise.
if python3 "$repo_root/scripts/perf_numbers.py" --check --repo "$work" >/dev/null 2>&1; then
  printf 'PASS %-32s baseline copy checks clean\n' "baseline"
else
  printf 'FAIL %-32s baseline copy does not check clean\n' "baseline"
  failures=$((failures + 1))
fi

# A. stale region: a hand-edited value differs from the receipt
python3 - "$work" <<'PY'
import json, sys, pathlib
root = pathlib.Path(sys.argv[1])
doc = root / "docs/PERF-NUMBERS.md"
text = doc.read_text()
manifest = json.loads((root / "docs/perf-numbers.json").read_text())
block = next(b for b in manifest["blocks"] if b["id"] == "cpu-logical-cores")
old = "<!-- BEGIN GENERATED: %s -->\n" % block["sentinel"]
start = text.index(old) + len(old)
end = text.index("<!-- END GENERATED: %s -->" % block["sentinel"], start)
doc.write_text(text[:start] + "999\n" + text[end:])
PY
check_case "stale-region" "stale region"
reset_copy

# B. missing receipt
python3 - "$work" <<'PY'
import json, sys, pathlib
root = pathlib.Path(sys.argv[1])
path = root / "docs/perf-numbers.json"
manifest = json.loads(path.read_text())
for block in manifest["blocks"]:
    if block["id"] == "cpu-logical-cores":
        block["source"]["receipt"] = "results/hardware/absent-receipt.json"
path.write_text(json.dumps(manifest, indent=2) + "\n")
PY
check_case "missing-receipt" "is missing"
reset_copy

# C. declared stamp no longer matches the receipt
python3 - "$work" <<'PY'
import json, sys, pathlib
root = pathlib.Path(sys.argv[1])
path = root / "docs/perf-numbers.json"
manifest = json.loads(path.read_text())
for block in manifest["blocks"]:
    if block["id"] == "cpu-logical-cores":
        block["declared_receipt_stamp"] = "20200101T000000Z"
path.write_text(json.dumps(manifest, indent=2) + "\n")
PY
check_case "stamp-mismatch" "does not match receipt stamp"
reset_copy

# D. declared revision no longer matches the receipt
python3 - "$work" <<'PY'
import json, sys, pathlib
root = pathlib.Path(sys.argv[1])
path = root / "docs/perf-numbers.json"
manifest = json.loads(path.read_text())
for block in manifest["blocks"]:
    if block["id"] == "cpu-logical-cores":
        block["declared_revision"] = "0" * 40
path.write_text(json.dumps(manifest, indent=2) + "\n")
PY
check_case "revision-mismatch" "does not match receipt revision"
reset_copy

# E. a generated region with no manifest block
printf '<!-- BEGIN GENERATED: orphan-test -->\n1\n<!-- END GENERATED: orphan-test -->\n' \
  >"$work/docs/ORPHAN-TEST.md"
check_case "orphan-sentinel" "has no manifest block"
reset_copy

# F. a numeric literal in the manifest that no region prints
python3 - "$work" <<'PY'
import json, sys, pathlib
root = pathlib.Path(sys.argv[1])
path = root / "docs/perf-numbers.json"
manifest = json.loads(path.read_text())
manifest["blocks"][0]["hand_written_expectation"] = 424242
path.write_text(json.dumps(manifest, indent=2) + "\n")
PY
check_case "unprinted-manifest-number" "is not printed by any block"
reset_copy

# G. a JSON pointer that does not resolve
python3 - "$work" <<'PY'
import json, sys, pathlib
root = pathlib.Path(sys.argv[1])
path = root / "docs/perf-numbers.json"
manifest = json.loads(path.read_text())
for block in manifest["blocks"]:
    if block["id"] == "cpu-logical-cores":
        block["source"]["pointer"] = "/cpu/no_such_field"
path.write_text(json.dumps(manifest, indent=2) + "\n")
PY
check_case "bad-pointer" "missing key"
reset_copy

# H. a declared sentinel that the document does not contain
python3 - "$work" <<'PY'
import json, sys, pathlib
root = pathlib.Path(sys.argv[1])
path = root / "docs/perf-numbers.json"
manifest = json.loads(path.read_text())
for block in manifest["blocks"]:
    if block["id"] == "cpu-logical-cores":
        block["sentinel"] = "cpu-logical-cores-missing"
path.write_text(json.dumps(manifest, indent=2) + "\n")
PY
check_case "missing-sentinel" "not found in"
reset_copy

# I. a format that would not print the number at all
python3 - "$work" <<'PY'
import json, sys, pathlib
root = pathlib.Path(sys.argv[1])
path = root / "docs/perf-numbers.json"
manifest = json.loads(path.read_text())
for block in manifest["blocks"]:
    if block["id"] == "cpu-logical-cores":
        block["format"] = "a constant with no placeholder"
path.write_text(json.dumps(manifest, indent=2) + "\n")
PY
check_case "format-hides-value" "must contain"
reset_copy

if ((failures > 0)); then
  printf 'perf_numbers negative cases: %d failure(s)\n' "$failures" >&2
  exit 1
fi
printf 'perf_numbers negative cases: all cases failed closed as required\n'
