#!/usr/bin/env bash
# Build the SELVEDGE court images and update docker/IMAGES.lock.
#
#   scripts/build-courts.sh                      # build rust and python, rewrite the lock
#   scripts/build-courts.sh rust                 # build one, update only its lock entry
#   scripts/build-courts.sh python
#   scripts/build-courts.sh oracle               # torch/transformers teacher image
#   scripts/build-courts.sh eval                 # lm-evaluation-harness evaluation image
#
# The oracle image is not part of the default build set: it is several
# gigabytes and only the real-model path needs it.
#
# Tags are content-addressed: <date>-<base-tag>-<sha256(inputs)[:12]>, where the
# inputs are the Dockerfile, the pinned requirements and the resolved base image
# digest. Re-running with identical inputs produces the same tag. The lock file
# records ref, image id, base image digest and the toolchain versions observed
# by actually running the built image.
#
# Build containers are memory/CPU limited too; a build is not a court, but it is
# still resource bounded.
set -Eeuo pipefail
IFS=$'\n\t'

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
repo_root=$(cd -- "$script_dir/.." && pwd -P)
lock_file="$repo_root/docker/IMAGES.lock"

BUILD_MEM="${SELVEDGE_BUILD_MEM:-4g}"
BUILD_CPUS="${SELVEDGE_BUILD_CPUS:-8}"
BUILD_NPROC="${SELVEDGE_BUILD_NPROC:-4096}"

die() {
  printf 'build-courts: %s\n' "$1" >&2
  exit "${2:-1}"
}

validate() {
  [[ "$2" =~ $3 ]] || die "$1='$2' is not a valid limit (must match $3)"
}
validate SELVEDGE_BUILD_MEM "$BUILD_MEM" '^[1-9][0-9]*[kKmMgG]$'
validate SELVEDGE_BUILD_CPUS "$BUILD_CPUS" '^[1-9][0-9]*$'
validate SELVEDGE_BUILD_NPROC "$BUILD_NPROC" '^[1-9][0-9]*$'

command -v docker >/dev/null 2>&1 || die "docker not found"
docker info >/dev/null 2>&1 || die "docker daemon unreachable"

targets=("$@")
if ((${#targets[@]} == 0)); then
  targets=(rust python)
fi
for target in "${targets[@]}"; do
  case "$target" in
    rust | python | oracle | eval) ;;
    *) die "unknown target '$target' (expected rust|python|oracle|eval)" 64 ;;
  esac
done

date_tag=$(date -u +%Y%m%d)
work=$(mktemp -d "${TMPDIR:-/tmp}/selvedge-build.XXXXXX")
trap 'rm -rf "$work"' EXIT

build_one() {
  local key="$1" dockerfile base_ref base_tag base_digest inputs_hash tag image_id probe_versions
  case "$key" in
    rust)
      dockerfile="docker/Dockerfile.rust-court"
      base_ref="${SELVEDGE_RUST_BASE:-rust:1.98-slim}"
      ;;
    python)
      dockerfile="docker/Dockerfile.python-court"
      base_ref="${SELVEDGE_PYTHON_BASE:-python:3.12-slim}"
      ;;
    oracle)
      dockerfile="docker/Dockerfile.oracle"
      # Digest-pinned by default: a tag alone is not a reproducible base.
      base_ref="${SELVEDGE_ORACLE_BASE:-python:3.12-slim@sha256:5e2bdb1b0d542d5039cffb8181c65930093949194a65c365dfb1b0fe13a4fa5b}"
      ;;
    eval)
      dockerfile="docker/Dockerfile.eval"
      # Built on top of the oracle court image, which is pinned by image id in
      # docker/IMAGES.lock. base_ref/base_digest are resolved below, after the
      # lock has been read; there is no registry pull for this target.
      base_ref=""
      ;;
  esac
  if [[ "$key" == "oracle" ]]; then
    base_tag="3.12-slim-cu132"
  elif [[ "$key" == "eval" ]]; then
    base_tag="3.12-slim-eval"
  else
    base_tag="${base_ref##*:}"
  fi

  if [[ "$key" == "eval" ]]; then
    # The eval court must sit on the EXACT vendor stack the reference-parity
    # receipts were produced with. That means: take the oracle image id recorded
    # in the lock, verify the local image still hashes to it, and use it. A
    # local image has no registry digest, so the lock's id is the pin.
    local oracle_fields
    oracle_fields=$(python3 - "$lock_file" <<'PY'
import json, sys
with open(sys.argv[1], "r", encoding="utf-8") as handle:
    lock = json.load(handle)
entry = (lock.get("images") or {}).get("oracle") or {}
print(entry.get("ref", ""))
print(entry.get("id", ""))
PY
    ) || die "cannot read the oracle entry from $lock_file"
    base_ref=$(printf '%s\n' "$oracle_fields" | sed -n 1p)
    local oracle_recorded_id
    oracle_recorded_id=$(printf '%s\n' "$oracle_fields" | sed -n 2p)
    [[ -n "$base_ref" && -n "$oracle_recorded_id" ]] ||
      die "the lock has no oracle image; run scripts/build-courts.sh oracle first"
    if ! docker image inspect "$base_ref" >/dev/null 2>&1; then
      die "oracle image '$base_ref' is not present; run scripts/build-courts.sh oracle first"
    fi
    base_digest=$(docker image inspect --format '{{.Id}}' "$base_ref")
    [[ "$base_digest" == "$oracle_recorded_id" ]] ||
      die "oracle image id $base_digest does not match the locked id $oracle_recorded_id"
  else
    printf 'build-courts: pulling base %s\n' "$base_ref" >&2
    docker pull --quiet "$base_ref" >/dev/null || die "cannot pull base image $base_ref"
    base_digest=$(docker image inspect --format '{{index .RepoDigests 0}}' "$base_ref" 2>/dev/null || true)
    [[ -n "$base_digest" ]] || die "base image $base_ref has no registry digest"
  fi

  inputs_hash=$(
    {
      cat "$repo_root/$dockerfile"
      if [[ "$key" == "oracle" ]]; then
        [[ -f "$repo_root/docker/requirements-oracle.txt" ]] && cat "$repo_root/docker/requirements-oracle.txt"
      elif [[ "$key" == "eval" ]]; then
        [[ -f "$repo_root/docker/requirements-eval.txt" ]] && cat "$repo_root/docker/requirements-eval.txt"
      else
        [[ -f "$repo_root/docker/requirements-court.txt" ]] && cat "$repo_root/docker/requirements-court.txt"
      fi
      printf '%s\n' "$base_digest"
    } | sha256sum | cut -c1-12
  ) || die "cannot hash build inputs"

  tag="selvedge/${key}-court:${date_tag}-${base_tag}-${inputs_hash}"
  local build_arg="RUST_BASE"
  [[ "$key" == "python" ]] && build_arg="PYTHON_BASE"
  [[ "$key" == "oracle" ]] && build_arg="ORACLE_BASE"
  [[ "$key" == "eval" ]] && build_arg="EVAL_BASE"
  printf 'build-courts: building %s\n' "$tag" >&2
  docker build \
    --memory "$BUILD_MEM" \
    --memory-swap "$BUILD_MEM" \
    --ulimit "nofile=65536:65536" \
    --build-arg "${build_arg}=${base_ref}" \
    -f "$repo_root/$dockerfile" \
    -t "$tag" \
    "$repo_root" >&2 || die "docker build failed for $tag"

  image_id=$(docker image inspect --format '{{.Id}}' "$tag")
  [[ -n "$image_id" ]] || die "cannot read image id for $tag"

  # Observe what the built image actually contains, under the same limits the
  # court will apply. This is a probe, not a test: it reports versions only.
  probe_versions=$(
    docker run --rm \
      --memory="$BUILD_MEM" --memory-swap="$BUILD_MEM" \
      --pids-limit="$BUILD_NPROC" --cpus="$BUILD_CPUS" \
      --network none \
      --tmpfs "/tmp:rw,size=1g,mode=1777,nosuid,nodev" \
      --security-opt no-new-privileges --cap-drop ALL \
      --user "$(id -u):$(id -g)" \
      --entrypoint /bin/sh \
      "$tag" -c 'for t in rustc cargo python3 pip3 cc git; do if command -v "$t" >/dev/null 2>&1; then printf "%s\t%s\n" "$t" "$("$t" --version 2>&1 | head -1)"; else printf "%s\t%s\n" "$t" absent; fi; done' \
      2>/dev/null || true
  )
  [[ -n "$probe_versions" ]] || die "built image $tag could not be probed"

  printf '%s\n' "$probe_versions" >"$work/$key.toolchain"
  printf '%s\n' "$tag" >"$work/$key.ref"
  printf '%s\n' "$image_id" >"$work/$key.id"
  printf '%s\n' "$base_ref" >"$work/$key.base_ref"
  printf '%s\n' "$base_digest" >"$work/$key.base_digest"
  printf '%s\n' "$inputs_hash" >"$work/$key.inputs_hash"
}

for target in "${targets[@]}"; do
  build_one "$target"
done

python3 - "$lock_file" "$work" "${targets[@]}" <<'PY'
import json
import os
import sys

lock_path, work, keys = sys.argv[1], sys.argv[2], sys.argv[3:]

lock = {"schema": "selvedge/images-lock/v1", "images": {}}
if os.path.exists(lock_path):
    with open(lock_path, "r", encoding="utf-8") as handle:
        try:
            existing = json.load(handle)
        except ValueError as exc:
            raise SystemExit(f"build-courts: existing lock is not valid JSON: {exc}")
    if existing.get("schema") != "selvedge/images-lock/v1":
        raise SystemExit("build-courts: existing lock has an unexpected schema")
    lock["images"] = existing.get("images") or {}

import datetime

for key in keys:
    def read(suffix):
        with open(os.path.join(work, f"{key}.{suffix}"), "r", encoding="utf-8") as handle:
            return handle.read().strip()

    toolchain = {}
    with open(os.path.join(work, f"{key}.toolchain"), "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if not line:
                continue
            name, _, value = line.partition("\t")
            toolchain[name] = value

    lock["images"][key] = {
        "ref": read("ref"),
        "id": read("id"),
        "repo_digest": None,
        "base_ref": read("base_ref"),
        "base_digest": read("base_digest"),
        "inputs_sha256_12": read("inputs_hash"),
        "built_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "toolchain": toolchain,
    }

lock["updated_utc"] = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

tmp_path = lock_path + ".tmp"
with open(tmp_path, "w", encoding="utf-8") as handle:
    json.dump(lock, handle, indent=2, sort_keys=True)
    handle.write("\n")
os.replace(tmp_path, lock_path)
print(f"build-courts: wrote {lock_path} for {', '.join(keys)}")
PY

printf 'build-courts: done\n' >&2
