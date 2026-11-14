#!/usr/bin/env bash
# Offline end-to-end demo: generate -> convert -> validate -> reconstruct.
#
# Every Rust step runs inside the Docker court, which defaults to
# `--network none`, bounded memory, bounded /tmp and the pinned image from
# docker/IMAGES.lock. The script itself only orchestrates files and git.
#
# Usage:
#   scripts/demo-offline.sh
#
# Outputs:
#   results/artifacts/demo-<stamp>/          (gitignored working artifacts)
#   results/demo-offline/<stamp>.json        (tracked report)
set -Eeuo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
repo_root=$(cd -- "$script_dir/.." && pwd -P)
cd "$repo_root"

if ! git rev-parse --verify --quiet HEAD >/dev/null; then
  printf 'demo-offline: repository has no HEAD commit\n' >&2
  exit 1
fi

stamp=$(date -u +%Y%m%dT%H%M%SZ)
started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
revision=$(git rev-parse HEAD)
worktree_dirty=false
if [[ -n "$(git status --porcelain -- . ':(exclude)results' ':(exclude)target' 2>/dev/null || true)" ]]; then
  worktree_dirty=true
fi
work_rel="results/artifacts/demo-${stamp}"
work_abs="$repo_root/$work_rel"
report_rel="results/demo-offline/${stamp}.json"
report_abs="$repo_root/$report_rel"
mkdir -p "$work_abs/model" "$work_abs/bundle" "$(dirname "$report_abs")"

run_step() {
  local name=$1
  shift
  printf 'demo-offline: %s\n' "$name" >&2
  scripts/court rust -- "$@"
}

# 1. Generate the deterministic synthetic checkpoint and the format fixture.
run_step generate cargo run -p selvedge-fixtures -- --out-dir "$work_rel/model" --force \
  >"$work_abs/generate.json" 2>"$work_abs/generate.err"
model_rel="$work_rel/model/tiny-swiglu.safetensors"
if [[ ! -f "$repo_root/$model_rel" ]]; then
  printf 'demo-offline: generator produced no safetensors fixture\n' >&2
  exit 1
fi

# 2. Convert into a real format-v1 bundle.
run_step convert cargo run -p selvedge-cli -- convert \
  --model "$model_rel" \
  --out "$work_rel/bundle" \
  --groups 4 \
  --complete \
  --alignment 512 \
  --force \
  --json \
  >"$work_abs/convert.json" 2>"$work_abs/convert.err"

bundle_rel=$(sed -n 's/.*"bundle_path": "\(.*\)",/\1/p' "$work_abs/convert.json" | head -n 1)
config_hash=$(sed -n 's/.*"config_hash": "\([0-9a-f]*\)".*/\1/p' "$work_abs/convert.json" | head -n 1)
if [[ -z "$bundle_rel" || ! -f "$repo_root/$bundle_rel" ]]; then
  printf 'demo-offline: conversion report did not name a readable bundle\n' >&2
  exit 1
fi
if [[ -z "$config_hash" ]]; then
  printf 'demo-offline: conversion report did not carry a config hash\n' >&2
  exit 1
fi

# 3. Full format validation, including every payload hash.
run_step validate cargo run -p selvedge-cli -- format validate "$bundle_rel" --json \
  >"$work_abs/validate.json" 2>"$work_abs/validate.err"

# 4. Reconstruct from the published bytes and compare with the checkpoint.
run_step reconstruct cargo run -p selvedge-cli -- convert verify \
  --bundle "$bundle_rel" \
  --model "$model_rel" \
  --json \
  >"$work_abs/verify.json" 2>"$work_abs/verify.err"

# 5. Run the full Court A matrix and write its receipt-shaped report.
run_step court-a cargo run -p selvedge-cli -- court a \
  --json \
  --report-out "$work_rel/court-a.json" \
  --stamp "$stamp" \
  --revision "$revision" \
  >"$work_abs/court-a.stdout.json" 2>"$work_abs/court-a.err"

if [[ ! -f "$repo_root/$work_rel/court-a.json" ]]; then
  printf 'demo-offline: court a produced no report\n' >&2
  exit 1
fi

sum() {
  sha256sum "$1" | cut -d' ' -f1
}

finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)

{
  printf '{\n'
  printf '  "schema": "selvedge/demo-offline/v1",\n'
  printf '  "ok": true,\n'
  printf '  "git_sha": "%s",\n' "$revision"
  printf '  "worktree_dirty": %s,\n' "$worktree_dirty"
  printf '  "stamp": "%s",\n' "$stamp"
  printf '  "started_utc": "%s",\n' "$started_utc"
  printf '  "finished_utc": "%s",\n' "$finished_utc"
  printf '  "network_mode": "none",\n'
  printf '  "config_hash": "%s",\n' "$config_hash"
  printf '  "seed": "0x5E1AED6E51E12026",\n'
  printf '  "source_model": "%s",\n' "$model_rel"
  printf '  "bundle": "%s",\n' "$bundle_rel"
  printf '  "artifacts": {\n'
  printf '    "generate_log_sha256": "%s",\n' "$(sum "$work_abs/generate.json")"
  printf '    "convert_report_sha256": "%s",\n' "$(sum "$work_abs/convert.json")"
  printf '    "validation_report_sha256": "%s",\n' "$(sum "$work_abs/validate.json")"
  printf '    "verification_report_sha256": "%s",\n' "$(sum "$work_abs/verify.json")"
  printf '    "court_a_report_sha256": "%s"\n' "$(sum "$repo_root/$work_rel/court-a.json")"
  printf '  },\n'
  printf '  "commands": [\n'
  printf '    "scripts/court rust -- cargo run -p selvedge-fixtures -- --out-dir %s --force",\n' "$work_rel/model"
  printf '    "scripts/court rust -- cargo run -p selvedge-cli -- convert --model %s --out %s --groups 4 --complete --alignment 512 --force --json",\n' "$model_rel" "$work_rel/bundle"
  printf '    "scripts/court rust -- cargo run -p selvedge-cli -- format validate %s --json",\n' "$bundle_rel"
  printf '    "scripts/court rust -- cargo run -p selvedge-cli -- convert verify --bundle %s --model %s --json",\n' "$bundle_rel" "$model_rel"
  printf '    "scripts/court rust -- cargo run -p selvedge-cli -- court a --json --report-out %s/court-a.json --stamp %s --revision %s"\n' "$work_rel" "$stamp" "$revision"
  printf '  ]\n'
  printf '}\n'
} >"$report_abs"

printf 'demo-offline: report %s (bundle %s, config %s)\n' \
  "$report_rel" "$bundle_rel" "$config_hash"
