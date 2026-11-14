#!/usr/bin/env bash
# SELVEDGE Docker court — the only place builds and tests are allowed to run.
#
# Usage:
#   scripts/docker-court.sh <court> [flags] [-- argv...]
#
#   courts: rust | python | io | commit_identity | perf_numbers
#   flags:  --net            enable network (default: --network none)
#           --gpu            pass the NVIDIA GPU (fails closed if unavailable)
#           --host-root-ro   expose the host root read-only at /hostroot
#           --               everything after this is the container argv
#
# Environment limits (all required to be non-empty, non-zero, well formed):
#   SELVEDGE_MEM    memory limit, swap limit is forced equal   (default 8g)
#   SELVEDGE_CPUS   CPU limit                                  (default 8)
#   SELVEDGE_PIDS   pids limit                                 (default 1024)
#   SELVEDGE_TMP    /tmp tmpfs size                            (default 2g)
#   SELVEDGE_NOFILE open-file limit                            (default 65536)
#
# Fail-closed rules enforced here:
#   * A missing/invalid/zero limit refuses to run (exit 78). There is no flag to
#     disable a limit.
#   * A missing docker binary or unreachable daemon refuses to run; there is no
#     host-execution fallback.
#   * After the container exits, the live container is inspected and the applied
#     limits are compared with the requested ones. Any mismatch writes a
#     receipt recording the mismatch and exits non-zero.
#   * Every run writes results/courts/<court>-<stamp>-<rand>.json and a log
#     under results/courts/logs/. Evidence is mandatory, not optional.
set -Eeuo pipefail
IFS=$'\n\t'

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
repo_root=$(cd -- "$script_dir/.." && pwd -P)
receipt_writer="$script_dir/courts/write-receipt.py"
toolchain_probe_script="/work/scripts/courts/probe-toolchain.sh"

die() {
  local code="${2:-78}"
  printf 'court: %s\n' "$1" >&2
  exit "$code"
}

usage() {
  cat <<'EOF'
usage: scripts/docker-court.sh <court> [flags] [-- argv...]

  courts : rust | python | io | commit_identity | perf_numbers
  flags  : --net            enable network (default: --network none)
           --gpu            pass the NVIDIA GPU (fails closed if unavailable)
           --host-root-ro   expose the host root read-only at /hostroot
           --               everything after this is the container argv

limits (required; empty, zero or malformed refuses to run):
  SELVEDGE_MEM=8g SELVEDGE_CPUS=8 SELVEDGE_PIDS=1024 SELVEDGE_TMP=2g

Every run writes results/courts/<court>-<stamp>-<rand>.json and a log under
results/courts/logs/. There is no host-execution fallback.
EOF
}

size_to_bytes() {
  local value="$1" number unit
  number="${value%[kKmMgG]}"
  unit="${value: -1}"
  unit="${unit,,}"
  case "$unit" in
    k) printf '%s' "$((number * 1024))" ;;
    m) printf '%s' "$((number * 1024 * 1024))" ;;
    g) printf '%s' "$((number * 1024 * 1024 * 1024))" ;;
    *) die "internal: bad size unit in '$value'" 70 ;;
  esac
}

validate_limit() {
  local name="$1" value="$2" pattern="$3"
  if [[ -z "$value" ]]; then
    die "$name is empty; refusing to run without a hard limit" 78
  fi
  if [[ ! "$value" =~ $pattern ]]; then
    die "$name='$value' is not a valid limit (must match $pattern); refusing to run unbounded" 78
  fi
}

# ---------------------------------------------------------------------------
# argument parsing
# ---------------------------------------------------------------------------
if (($# == 0)); then
  usage >&2
  exit 64
fi
case "$1" in
  -h | --help)
    usage
    exit 0
    ;;
esac

court="$1"
shift
case "$court" in
  rust | python | io | commit_identity | perf_numbers) ;;
  *) die "unknown court '$court' (expected rust|python|io|commit_identity|perf_numbers)" 64 ;;
esac

net=0
gpu=0
host_root_ro=0
declare -a argv=()
while (($#)); do
  case "$1" in
    --)
      shift
      argv=("$@")
      break
      ;;
    --net)
      net=1
      shift
      ;;
    --gpu)
      gpu=1
      shift
      ;;
    --host-root-ro)
      host_root_ro=1
      shift
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    *)
      die "unexpected argument '$1' (court flags must precede '--')" 64
      ;;
  esac
done

# ---------------------------------------------------------------------------
# hard limits, validated before anything runs
# ---------------------------------------------------------------------------
mem="${SELVEDGE_MEM-8g}"
cpus="${SELVEDGE_CPUS-8}"
pids="${SELVEDGE_PIDS-1024}"
tmp_size="${SELVEDGE_TMP-2g}"
nofile="${SELVEDGE_NOFILE-65536}"

validate_limit SELVEDGE_MEM "$mem" '^[1-9][0-9]*[kKmMgG]$'
validate_limit SELVEDGE_CPUS "$cpus" '^[1-9][0-9]*(\.[0-9]+)?$'
validate_limit SELVEDGE_PIDS "$pids" '^[1-9][0-9]*$'
validate_limit SELVEDGE_TMP "$tmp_size" '^[1-9][0-9]*[kKmMgG]$'
validate_limit SELVEDGE_NOFILE "$nofile" '^[1-9][0-9]*$'

mem_bytes=$(size_to_bytes "$mem")
tmp_bytes=$(size_to_bytes "$tmp_size")
nano_cpus=$(awk -v c="$cpus" 'BEGIN { printf "%.0f", c * 1000000000 }')
if [[ "$nano_cpus" -le 0 ]]; then
  die "SELVEDGE_CPUS='$cpus' resolves to a non-positive cpu quota" 78
fi

# ---------------------------------------------------------------------------
# runtime preflight: no host fallback, ever
# ---------------------------------------------------------------------------
command -v docker >/dev/null 2>&1 ||
  die "docker not found; refusing to run on the host (no fallback exists)" 69
docker info >/dev/null 2>&1 ||
  die "docker daemon unreachable; refusing to run on the host (no fallback exists)" 69

runtime_caps=$(docker info --format '{{.MemoryLimit}} {{.SwapLimit}} {{.PidsLimit}}' 2>/dev/null) ||
  die "cannot query docker runtime capabilities; refusing to run" 69
IFS=' ' read -r cap_mem cap_swap cap_pids <<<"$runtime_caps" || true
cap_mem="${cap_mem:-unknown}"
cap_swap="${cap_swap:-unknown}"
cap_pids="${cap_pids:-unknown}"
if [[ "$cap_mem" != "true" || "$cap_swap" != "true" || "$cap_pids" != "true" ]]; then
  die "docker runtime cannot enforce memory/swap/pids limits (memory=$cap_mem swap=$cap_swap pids=$cap_pids); refusing to run" 78
fi

# ---------------------------------------------------------------------------
# image resolution from the lock file (content-addressed tags)
# ---------------------------------------------------------------------------
lock_file="$repo_root/docker/IMAGES.lock"
[[ -f "$lock_file" ]] || die "docker/IMAGES.lock is missing; run scripts/build-courts.sh" 78

case "$court" in
  rust) image_key="rust" ;;
  *) image_key="python" ;;
esac

image_ref=$(python3 - "$lock_file" "$image_key" <<'PY'
import json, sys
path, key = sys.argv[1], sys.argv[2]
with open(path, "r", encoding="utf-8") as handle:
    lock = json.load(handle)
image = (lock.get("images") or {}).get(key)
if not image or not image.get("ref"):
    raise SystemExit(f"no image entry for '{key}'")
print(image["ref"])
PY
) || die "cannot resolve the '$image_key' court image from $lock_file" 78

docker image inspect "$image_ref" >/dev/null 2>&1 ||
  die "court image '$image_ref' is not present; run scripts/build-courts.sh" 78

image_id=$(docker image inspect --format '{{.Id}}' "$image_ref")
image_digest=$(docker image inspect --format '{{index .RepoDigests 0}}' "$image_ref" 2>/dev/null || true)
if [[ -n "$image_digest" ]]; then
  image_digest_source="RepoDigest"
else
  image_digest="$image_id"
  image_digest_source="image-id (locally built, never pushed to a registry)"
fi

docker_client=$(docker version --format '{{.Client.Version}}' 2>/dev/null || echo unknown)
docker_server=$(docker version --format '{{.Server.Version}}' 2>/dev/null || echo unknown)
cgroup_version=$(docker info --format '{{.CgroupVersion}}' 2>/dev/null || echo unknown)

# ---------------------------------------------------------------------------
# default argv per court
# ---------------------------------------------------------------------------
if ((${#argv[@]} == 0)); then
  case "$court" in
    rust) argv=(cargo --version) ;;
    python) argv=(python3 --version) ;;
    io) argv=(df -h /tmp) ;;
    commit_identity) argv=(scripts/check-commit-identity.sh) ;;
    perf_numbers) argv=(python3 scripts/perf_numbers.py --check) ;;
  esac
fi

# ---------------------------------------------------------------------------
# caches and output directories
# ---------------------------------------------------------------------------
[[ -n "${HOME:-}" ]] || die "HOME is unset; cannot locate the persistent caches" 78
cargo_cache="${SELVEDGE_CARGO_CACHE:-$HOME/.cache/selvedge/cargo}"
pycache="${SELVEDGE_PYCACHE:-$HOME/.cache/selvedge/pycache}"
mkdir -p "$cargo_cache/registry" "$cargo_cache/git" "$pycache/pip" "$pycache/pycache" "$repo_root/target" ||
  die "cannot create cache directories under $cargo_cache / $pycache" 73
mkdir -p "$repo_root/results/courts/logs" "$repo_root/results/courts/.tmp" ||
  die "cannot create results/courts directories" 73

stamp=$(date -u +%Y%m%dT%H%M%SZ)
rand=$(od -An -N3 -tx1 /dev/urandom | tr -d ' \n')
started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
container_name="selvedge-${court}-court-${stamp}-${rand}"
log_rel="results/courts/logs/${court}-${stamp}-${rand}.log"
receipt_rel="results/courts/${court}-${stamp}-${rand}.json"
toolchain_rel="results/courts/.tmp/${rand}.toolchain"
log_abs="$repo_root/$log_rel"
toolchain_abs="$repo_root/$toolchain_rel"

tmpdir=$(mktemp -d "${TMPDIR:-/tmp}/selvedge-court.XXXXXX")
cleanup() {
  local code=$?
  rm -rf "$tmpdir"
  exit "$code"
}
trap cleanup EXIT

if revision=$(git -C "$repo_root" rev-parse --verify --quiet HEAD 2>/dev/null); then
  :
else
  revision="unborn"
fi
worktree_dirty=false
if [[ -n "$(git -C "$repo_root" status --porcelain -- . ':(exclude)results' ':(exclude)target' 2>/dev/null || true)" ]]; then
  worktree_dirty=true
fi

# ---------------------------------------------------------------------------
# GPU selection: --gpus all first, then explicit device passthrough.
# Recorded in the receipt; a GPU court never silently runs on CPU.
# ---------------------------------------------------------------------------
declare -a gpu_args=()
gpu_mode="none"
gpu_detail=""
if ((gpu)); then
  probe_common=(
    --rm
    --memory "$mem"
    --memory-swap "$mem"
    --pids-limit "$pids"
    --cpus "$cpus"
    --tmpfs "/tmp:rw,size=${tmp_size},mode=1777,nosuid,nodev"
    --security-opt no-new-privileges
    --cap-drop ALL
    --network none
  )
  gpus_all_error=""
  if gpus_all_error=$(docker run "${probe_common[@]}" --gpus all "$image_ref" /bin/true 2>&1); then
    gpu_mode="gpus-all"
    gpu_args=(--gpus all)
  else
    gpus_all_error=$(printf '%s' "$gpus_all_error" | tr '\n' ' ' | cut -c1-400)
    command -v nvidia-smi >/dev/null 2>&1 ||
      die "GPU requested but neither '--gpus all' nor device passthrough works ($gpus_all_error)" 69
    driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | tr -d ' \r')
    [[ -n "$driver" ]] || die "GPU requested but the driver version cannot be read" 69
    libdir="${SELVEDGE_GPU_LIBDIR:-/usr/lib/x86_64-linux-gnu}"
    declare -a found_libs=()
    for base in libcuda libnvidia-ml libnvidia-nvvm libnvidia-ptxjitcompiler libnvidia-cfg; do
      for src in "/usr/lib/${base}.so.${driver}" "/usr/lib/x86_64-linux-gnu/${base}.so.${driver}"; do
        if [[ -e "$src" ]]; then
          found_libs+=(-v "${src}:${libdir}/${base}.so.1:ro")
          break
        fi
      done
    done
    ((${#found_libs[@]} > 0)) ||
      die "GPU requested: '--gpus all' failed ($gpus_all_error) and no host NVIDIA libraries were found" 69
    smi_path=$(command -v nvidia-smi)
    gpu_args=(
      --device /dev/nvidiactl
      --device /dev/nvidia-uvm
      --device /dev/nvidia-uvm-tools
      "${found_libs[@]}"
      -v "${smi_path}:/usr/bin/nvidia-smi:ro"
    )
    for dev in /dev/nvidia0 /dev/nvidia1; do
      [[ -e "$dev" ]] && gpu_args+=(--device "$dev")
    done
    gpu_mode="device-passthrough"
    gpu_detail="--gpus all unavailable: ${gpus_all_error}"
  fi
fi

# ---------------------------------------------------------------------------
# container invocation
# ---------------------------------------------------------------------------
if ((net)); then
  network_mode="bridge"
else
  network_mode="none"
fi

host_uid=$(id -u)
host_gid=$(id -g)
declare -a mounts=("$repo_root:/work" "$cargo_cache:/cargo" "$repo_root/target:/work/target")
declare -a extra_mounts=()
if [[ "$court" == "python" || "$court" == "io" || "$court" == "perf_numbers" ]]; then
  mounts+=("$pycache:/pycache")
fi
if ((host_root_ro)); then
  mounts+=("/:/hostroot:ro")
  extra_mounts+=("/:/hostroot:ro")
fi

declare -a common=(
  --name "$container_name"
  --init
  --memory "$mem"
  --memory-swap "$mem"
  --pids-limit "$pids"
  --cpus "$cpus"
  --ulimit "nofile=${nofile}:${nofile}"
  --security-opt no-new-privileges
  --cap-drop ALL
  --tmpfs "/tmp:rw,size=${tmp_size},mode=1777,nosuid,nodev"
  --network "$network_mode"
  --user "${host_uid}:${host_gid}"
  -w /work
  -e "SELVEDGE_IN_COURT=1"
  -e "SELVEDGE_COURT=${court}"
  -e "SELVEDGE_COURT_IMAGE=${image_ref}"
  -e "SELVEDGE_COURT_IMAGE_ID=${image_id}"
  -e "SELVEDGE_COURT_MEM=${mem}"
  -e "SELVEDGE_COURT_MEM_BYTES=${mem_bytes}"
  -e "SELVEDGE_COURT_TMP_BYTES=${tmp_bytes}"
  -e "SELVEDGE_COURT_CPUS=${cpus}"
  -e "SELVEDGE_COURT_PIDS=${pids}"
  -e "SELVEDGE_REVISION=${revision}"
  -e "SELVEDGE_TOOLCHAIN_OUT=/work/${toolchain_rel}"
  -e "HOME=/tmp"
  -e "TMPDIR=/tmp"
  -e "CARGO_HOME=/cargo"
  -e "CARGO_TARGET_DIR=/work/target"
  -e "RUSTUP_HOME=/usr/local/rustup"
  -e "RUST_BACKTRACE=1"
  -e "CARGO_INCREMENTAL=0"
  -e "CARGO_BUILD_JOBS=${cpus%%.*}"
  -e "PYTHONPYCACHEPREFIX=/pycache/pycache"
  -e "PIP_CACHE_DIR=/pycache/pip"
  -e "PYTHONDONTWRITEBYTECODE=1"
)
for mount in "${mounts[@]}"; do
  common+=(-v "$mount")
done
if [[ "$network_mode" == "none" ]]; then
  # Offline by construction; cargo must fail loudly instead of hanging on DNS.
  common+=(-e "CARGO_NET_OFFLINE=true")
fi

printf 'court: %s image=%s memory=%s cpus=%s pids=%s tmp=%s network=%s gpu=%s\n' \
  "$court" "$image_ref" "$mem" "$cpus" "$pids" "$tmp_size" "$network_mode" "$gpu_mode" >&2

start_ns=$(date +%s%N)
set +e
docker run "${common[@]}" "${gpu_args[@]}" "$image_ref" \
  /bin/sh -c 'if [ -n "${SELVEDGE_TOOLCHAIN_OUT:-}" ]; then /bin/sh '"$toolchain_probe_script"' > "$SELVEDGE_TOOLCHAIN_OUT" 2>&1 || true; fi; exec "$@"' \
  selvedge-court "${argv[@]}" 2>&1 | tee "$log_abs"
rc=${PIPESTATUS[0]}
set -e
end_ns=$(date +%s%N)
duration_ms=$(((end_ns - start_ns) / 1000000))
finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)

# ---------------------------------------------------------------------------
# verify the limits actually applied, then write the receipt
# ---------------------------------------------------------------------------
inspect_file="$tmpdir/inspect.json"
if ! docker inspect "$container_name" >"$inspect_file" 2>/dev/null; then
  printf '[]\n' >"$inspect_file"
  printf 'court: WARNING could not inspect container %s; enforcement cannot be proven\n' \
    "$container_name" >&2
fi
docker rm -f "$container_name" >/dev/null 2>&1 || true

toolchain_path=""
if [[ -f "$toolchain_abs" ]]; then
  toolchain_path="$toolchain_abs"
fi

set +e
python3 "$receipt_writer" \
  --receipt-out "$repo_root/$receipt_rel" \
  --court "$court" \
  --stamp "$stamp" \
  --started-utc "$started_utc" \
  --finished-utc "$finished_utc" \
  --revision "$revision" \
  --worktree-dirty "$worktree_dirty" \
  --container-name "$container_name" \
  --inspect "$inspect_file" \
  --toolchain "$toolchain_path" \
  --image-ref "$image_ref" \
  --image-id "$image_id" \
  --image-digest "$image_digest" \
  --image-digest-source "$image_digest_source" \
  --memory "$mem" \
  --cpus "$cpus" \
  --pids "$pids" \
  --tmpfs-size "$tmp_size" \
  --nofile "$nofile" \
  --network "$network_mode" \
  --gpu-mode "$gpu_mode" \
  --gpu-detail "$gpu_detail" \
  --user "${host_uid}:${host_gid}" \
  --mount "$repo_root:/work" \
  --mount "$cargo_cache:/cargo" \
  --mount "$repo_root/target:/work/target" \
  --exit-code "$rc" \
  --duration-ms "$duration_ms" \
  --host-uname "$(uname -srm)" \
  --docker-client "$docker_client" \
  --docker-server "$docker_server" \
  --cgroup-version "$cgroup_version" \
  --log "$log_rel" \
  -- "${argv[@]}"
receipt_rc=$?
set -e

rm -f "$toolchain_abs"

printf 'court: %s exit=%s enforcement=%s receipt=%s log=%s\n' \
  "$court" "$rc" "$([[ $receipt_rc -eq 0 ]] && echo ok || echo FAILED)" \
  "$receipt_rel" "$log_rel" >&2

if [[ -n "${SELVEDGE_RECEIPT_PATHFILE:-}" ]]; then
  printf '%s\n' "$repo_root/$receipt_rel" >"$SELVEDGE_RECEIPT_PATHFILE"
fi

if [[ "$receipt_rc" -ne 0 ]]; then
  exit "$receipt_rc"
fi
exit "$rc"
