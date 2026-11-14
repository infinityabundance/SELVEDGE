#!/usr/bin/env bash
# Emit results/hardware/probe-<utc-stamp>.json.
#
# Every hardware fact is collected inside a court container:
#   * host-visible CPU / RAM / swap / NVMe / kernel facts come from a Python
#     court run with the host root mounted read-only at /hostroot;
#   * GPU facts come from a --gpu court run that executes nvidia-smi *inside*
#     the container, so "GPU works" means "a court container can see it";
#   * toolchain versions come from the Receipts of rust/python court runs.
#
# The host itself is only used for git and Docker orchestration (uname, docker
# version). No test or build ever runs on the host.
set -Eeuo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
repo_root=$(cd -- "$script_dir/.." && pwd -P)

die() {
  printf 'probe-hardware: %s\n' "$1" >&2
  exit 1
}

stamp=$(date -u +%Y%m%dT%H%M%SZ)
out_dir="$repo_root/results/hardware"
mkdir -p "$out_dir" || die "cannot create $out_dir"
out_file="$out_dir/probe-${stamp}.json"

tmpdir=$(mktemp -d "${TMPDIR:-/tmp}/selvedge-hwprobe.XXXXXX")
cleanup() {
  local code=$?
  rm -rf "$tmpdir"
  exit "$code"
}
trap cleanup EXIT

revision=$(git -C "$repo_root" rev-parse HEAD 2>/dev/null || echo unborn)
dirty=false
if [[ -n "$(git -C "$repo_root" status --porcelain 2>/dev/null || true)" ]]; then
  dirty=true
fi

# ---------------------------------------------------------------------------
# 1. host-visible facts, inside the Python court
# ---------------------------------------------------------------------------
printf 'probe-hardware: collecting host facts in the python court\n' >&2
SELVEDGE_RECEIPT_PATHFILE="$tmpdir/host.receipt" \
  "$script_dir/docker-court.sh" python --host-root-ro -- \
  python3 scripts/courts/hardware_probe.py >"$tmpdir/host.json" ||
  die "host-facts court run failed (see the court receipt above)"
[[ -s "$tmpdir/host.json" ]] || die "host-facts court produced no JSON"
python3 -c 'import json,sys; json.load(open(sys.argv[1]))' "$tmpdir/host.json" ||
  die "host-facts court produced invalid JSON"
host_receipt=$(cat "$tmpdir/host.receipt")

# ---------------------------------------------------------------------------
# 2. GPU facts, inside a --gpu court
# ---------------------------------------------------------------------------
gpu_status="ok"
gpu_receipt=""
printf 'probe-hardware: probing the GPU in a --gpu court\n' >&2
set +e
SELVEDGE_RECEIPT_PATHFILE="$tmpdir/gpu.receipt" \
  "$script_dir/docker-court.sh" python --gpu -- \
  python3 scripts/courts/gpu_probe.py >"$tmpdir/gpu.json" 2>"$tmpdir/gpu.err"
gpu_rc=$?
set -e
if [[ $gpu_rc -ne 0 ]]; then
  gpu_status="failed (exit $gpu_rc): $(tr '\n' ' ' <"$tmpdir/gpu.err" | cut -c1-600)"
  printf '{}\n' >"$tmpdir/gpu.json"
  printf 'probe-hardware: GPU court failed; recorded as unavailable\n' >&2
fi
if [[ -f "$tmpdir/gpu.receipt" ]]; then
  gpu_receipt=$(cat "$tmpdir/gpu.receipt")
fi

# ---------------------------------------------------------------------------
# 3. toolchain facts, from real court receipts
# ---------------------------------------------------------------------------
printf 'probe-hardware: recording toolchain versions from court receipts\n' >&2
SELVEDGE_RECEIPT_PATHFILE="$tmpdir/rust.receipt" \
  "$script_dir/docker-court.sh" rust -- rustc --version >/dev/null ||
  die "rust toolchain court run failed"
SELVEDGE_RECEIPT_PATHFILE="$tmpdir/python.receipt" \
  "$script_dir/docker-court.sh" python -- python3 --version >/dev/null ||
  die "python toolchain court run failed"

docker_client=$(docker version --format '{{.Client.Version}}')
docker_server=$(docker version --format '{{.Server.Version}}')
cgroup_version=$(docker info --format '{{.CgroupVersion}}')

# ---------------------------------------------------------------------------
# 4. assemble the receipt
# ---------------------------------------------------------------------------
python3 "$script_dir/courts/assemble_hardware_probe.py" \
  --out "$out_file" \
  --stamp "$stamp" \
  --revision "$revision" \
  --dirty "$dirty" \
  --host "$tmpdir/host.json" \
  --gpu "$tmpdir/gpu.json" \
  --gpu-status "$gpu_status" \
  --gpu-court-receipt "${gpu_receipt:-none}" \
  --host-court-receipt "$host_receipt" \
  --rust-receipt "$(cat "$tmpdir/rust.receipt")" \
  --python-receipt "$(cat "$tmpdir/python.receipt")" \
  --docker-client "$docker_client" \
  --docker-server "$docker_server" \
  --cgroup-version "$cgroup_version" \
  --host-uname "$(uname -srm)" \
  --repo "$repo_root" ||
  die "cannot assemble the hardware probe receipt"

printf 'probe-hardware: %s\n' "$out_file" >&2
