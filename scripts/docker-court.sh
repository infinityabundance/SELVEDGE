#!/usr/bin/env bash
# SELVEDGE Docker court — the only place builds and tests are allowed to run.
#
# Usage:
#   scripts/docker-court.sh <court> [flags] [-- argv...]
#
#   courts: rust | python | oracle | eval | io | commit_identity | perf_numbers | status_consistency | prereg_provenance | wait
#   flags:  --net            enable network (default: --network none)
#           --gpu            pass the NVIDIA GPU (fails closed if unavailable)
#           --host-root-ro   expose the host root read-only at /hostroot
#           --detach         launch with `docker run -d` and return immediately;
#                            `scripts/court wait <state.json>` finishes the run,
#                            writes the log and the receipt, and exits with the
#                            container's exit code. Limits, mounts and the GPU
#                            are byte-identical to the foreground path.
#           --               everything after this is the container argv
#
# The `oracle` court mounts the persistent Hugging Face cache read-write at
# /hf-cache and exports HF_HOME/HF_HUB_CACHE. The mount is recorded in every
# receipt, together with the GPU mode. The court is offline unless --net is
# passed, so model fetching is always an explicit, receipted step.
#
# Environment limits (all required to be non-empty, non-zero, well formed):
#   SELVEDGE_MEM    memory limit, swap limit is forced equal   (default 8g)
#   SELVEDGE_CPUS   CPU limit                                  (default 8)
#   SELVEDGE_PIDS   pids limit                                 (default 1024)
#   SELVEDGE_TMP    /tmp tmpfs size                            (default 2g)
#   SELVEDGE_NOFILE open-file limit                              (default 65536)
#
# GPU memory policy: `PYTORCH_CUDA_ALLOC_CONF` is forwarded into every
# container (default `expandable_segments:True`) and recorded in the receipt.
# It changes no arithmetic; it stops a long-lived evaluation that holds a
# teacher and a student at once from fragmenting the allocator into an OOM.
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

  courts : rust | python | oracle | eval | io | commit_identity | perf_numbers | status_consistency | prereg_provenance | wait
           (or start with -- for a generic Python-court passthrough)
  flags  : --net            enable network (default: --network none)
           --gpu            pass the NVIDIA GPU (fails closed if unavailable)
           --host-root-ro   expose the host root read-only at /hostroot
           --detach         launch detached; finish with `court wait <state.json>`
           --               everything after this is the container argv

  detach : scripts/docker-court.sh oracle --gpu --detach -- python3 work.py
           # prints the container name and the pending-state path
           scripts/docker-court.sh wait -- results/courts/pending/<name>.json

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

court=""
net=0
gpu=0
host_root_ro=0
detach=0
declare -a argv=()

if [[ "$1" == "--" ]]; then
  # Generic passthrough: the Python court image, no court-specific argv.
  shift
  court="io"
  argv=("$@")
  set --
else
  court="$1"
  shift
fi

case "$court" in
  rust | python | oracle | eval | io | commit_identity | perf_numbers | status_consistency | prereg_provenance | wait) ;;
  *) die "unknown court '$court' (expected rust|python|oracle|io|commit_identity|perf_numbers|status_consistency|wait, or -- <argv>)" 64 ;;

esac

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
    --detach)
      detach=1
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
# wait mode: block on a detached container, capture its log, write its receipt
# ---------------------------------------------------------------------------
if [[ "$court" == "wait" ]]; then
  ((${#argv[@]} == 1)) || die "wait takes exactly one pending-state file" 64
  state_abs="${argv[0]}"
  [[ -f "$state_abs" ]] || die "pending-state file '$state_abs' does not exist" 66
  command -v docker >/dev/null 2>&1 || die "docker not found; cannot wait" 69
  docker info >/dev/null 2>&1 || die "docker daemon unreachable; cannot wait" 69

  mapfile -t state_fields < <(python3 - "$state_abs" <<'PY'
import json, sys
state = json.load(open(sys.argv[1], encoding="utf-8"))
for key in ("container_name", "log_rel", "receipt_rel", "toolchain_abs",
            "inspect_abs", "started_utc"):
    print(state.get(key, ""))
PY
  ) || die "cannot read pending state '$state_abs'" 78
  ((${#state_fields[@]} == 6)) || die "malformed pending state '$state_abs'" 78
  container_name="${state_fields[0]}"
  log_rel="${state_fields[1]}"
  receipt_rel="${state_fields[2]}"
  toolchain_abs="${state_fields[3]}"
  inspect_abs="${state_fields[4]}"
  started_utc="${state_fields[5]}"
  [[ -n "$container_name" && -n "$log_rel" && -n "$receipt_rel" ]] ||
    die "pending state '$state_abs' is missing container or receipt fields" 78

  # Fail closed if the container vanished: there is nothing to attribute.
  docker inspect --format '{{.State.Status}}' "$container_name" >/dev/null 2>&1 ||
    die "detached container '$container_name' no longer exists" 66

  start_ns=$(date +%s%N)
  set +e
  # `docker wait` prints the container's exit code on stdout; its own status is
  # the success of the wait, not of the container.
  rc=$(docker wait "$container_name" 2>/dev/null)
  wait_rc=$?
  set -e
  end_ns=$(date +%s%N)
  wait_ms=$(((end_ns - start_ns) / 1000000))
  ((wait_rc == 0)) || die "docker wait failed for '$container_name'" 69
  [[ "$rc" =~ ^[0-9]+$ ]] || die "docker wait returned '$rc' for '$container_name'" 69

  mkdir -p "$(dirname "$repo_root/$log_rel")" "$(dirname "$inspect_abs")" ||
    die "cannot create the log directory" 73
  set +e
  docker logs "$container_name" >"$repo_root/$log_rel" 2>&1
  logs_rc=$?
  set -e
  ((logs_rc == 0)) || printf 'court: WARNING docker logs failed for %s\n' "$container_name" >&2

  if ! docker inspect "$container_name" >"$inspect_abs" 2>/dev/null; then
    printf '[]\n' >"$inspect_abs"
    printf 'court: WARNING could not inspect container %s; enforcement cannot be proven\n' \
      "$container_name" >&2
  fi
  # Duration prefers the daemon's own start/finish timestamps.
  duration_ms=$wait_ms
  finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  if measured=$(python3 - "$inspect_abs" <<'PY'
import datetime, json, sys
try:
    payload = json.load(open(sys.argv[1], encoding="utf-8"))
except Exception:
    raise SystemExit(0)
if not isinstance(payload, list) or not payload:
    raise SystemExit(0)
state = payload[0].get("State") or {}
started = state.get("StartedAt") or ""
finished = state.get("FinishedAt") or ""
try:
    begin = datetime.datetime.fromisoformat(started.replace("Z", "+00:00"))
    end = datetime.datetime.fromisoformat(finished.replace("Z", "+00:00"))
except Exception:
    raise SystemExit(0)
if end <= begin:
    raise SystemExit(0)
print(int((end - begin).total_seconds() * 1000))
PY
  ); then
    if [[ "$measured" =~ ^[0-9]+$ ]] && ((measured > 0)); then
      duration_ms=$measured
    fi
  fi

  docker rm -f "$container_name" >/dev/null 2>&1 || true

  toolchain_path=""
  [[ -f "$toolchain_abs" ]] && toolchain_path="$toolchain_abs"

  set +e
  python3 - "$state_abs" "$finished_utc" "$toolchain_path" "$rc" "$duration_ms" <<'PY'
import json, subprocess, sys
state = json.load(open(sys.argv[1], encoding="utf-8"))
finished_utc, toolchain, exit_code, duration_ms = sys.argv[2:6]
argv = [state["receipt_args"][0]] + state["receipt_args"][1:]
argv += ["--finished-utc", finished_utc, "--toolchain", toolchain,
         "--exit-code", exit_code, "--duration-ms", duration_ms, "--"] + state["argv"]
raise SystemExit(subprocess.call([sys.executable] + argv))
PY
  receipt_rc=$?
  set -e

  rm -f "$toolchain_abs" "$inspect_abs"

  printf 'court: %s exit=%s enforcement=%s receipt=%s log=%s\n' \
    "${state_abs##*/}" "$rc" "$([[ $receipt_rc -eq 0 ]] && echo ok || echo FAILED)" \
    "$receipt_rel" "$log_rel" >&2
  if [[ -n "${SELVEDGE_RECEIPT_PATHFILE:-}" ]]; then
    printf '%s\n' "$repo_root/$receipt_rel" >"$SELVEDGE_RECEIPT_PATHFILE"
  fi
  if ((receipt_rc != 0)); then
    exit "$receipt_rc"
  fi
  # The pending-state file is kept: it holds the exact argv of the run and is
  # consumed by the pilot's arm receipt. A human can delete old ones by hand.
  exit "$rc"
fi

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

# GPU allocator policy: forwarded, recorded, and fail-closed on garbage.
alloc_conf="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
[[ "$alloc_conf" =~ ^[A-Za-z0-9_:,.=+-]+$ ]] ||
  die "PYTORCH_CUDA_ALLOC_CONF='$alloc_conf' is not a well formed allocator config" 78

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
  oracle) image_key="oracle" ;;
  eval) image_key="eval" ;;
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
    oracle) argv=(python3 /work/oracle/probe_oracle.py) ;;
    io) argv=(df -h /tmp) ;;
    commit_identity) argv=(scripts/check-commit-identity.sh) ;;
    perf_numbers) argv=(python3 scripts/perf_numbers.py --check) ;;
    status_consistency) argv=(python3 scripts/status_consistency.py --check) ;;
    prereg_provenance) argv=(python3 scripts/prereg_provenance.py --check) ;;
  esac
fi

# ---------------------------------------------------------------------------
# caches and output directories
# ---------------------------------------------------------------------------
[[ -n "${HOME:-}" ]] || die "HOME is unset; cannot locate the persistent caches" 78
cargo_cache="${SELVEDGE_CARGO_CACHE:-$HOME/.cache/selvedge/cargo}"
pycache="${SELVEDGE_PYCACHE:-$HOME/.cache/selvedge/pycache}"
hf_cache="${SELVEDGE_HF_CACHE:-$HOME/.cache/selvedge/hf}"
mkdir -p "$cargo_cache/registry" "$cargo_cache/git" "$pycache/pip" "$pycache/pycache" "$hf_cache" "$repo_root/target" ||
  die "cannot create cache directories under $cargo_cache / $pycache / $hf_cache" 73
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
declare -a recorded_mounts=()
if [[ "$court" == "python" || "$court" == "io" || "$court" == "perf_numbers" || "$court" == "status_consistency" || "$court" == "prereg_provenance" || "$court" == "eval" ]]; then
  mounts+=("$pycache:/pycache")
fi
if [[ "$court" == "oracle" || "$court" == "eval" ]]; then
  # Persistent model cache, read-write. Recorded so a receipt always names the
  # exact host path a teacher is read from.
  mounts+=("$hf_cache:/hf-cache")
  recorded_mounts+=("$hf_cache:/hf-cache")
fi
if [[ "$court" == "eval" ]]; then
  # Harness task data (datasets library cache). Bounded by the eval workflow:
  # only the anchored task list is fetched, and the run refuses to start if the
  # cache filesystem is below SELVEDGE_DATASETS_MIN_FREE (default 3 GiB).
  datasets_cache="${SELVEDGE_DATASETS_CACHE:-$hf_cache/datasets}"
  mkdir -p "$datasets_cache" || die "cannot create $datasets_cache" 73
  mounts+=("$datasets_cache:/datasets-cache")
  recorded_mounts+=("$datasets_cache:/datasets-cache")
fi
if [[ "$court" == "rust" ]]; then
  # Rust accounting/conversion read real checkpoints from the same cache but
  # never write to it. Read-only is enforced by the mount.
  mounts+=("$hf_cache:/hf-cache:ro")
  recorded_mounts+=("$hf_cache:/hf-cache:ro")
fi
if ((host_root_ro)); then
  mounts+=("/:/hostroot:ro")
  extra_mounts+=("/:/hostroot:ro")
fi

declare -a receipt_mount_args=(
  --mount "$repo_root:/work"
  --mount "$cargo_cache:/cargo"
  --mount "$repo_root/target:/work/target"
)
for mounted in "${recorded_mounts[@]}"; do
  receipt_mount_args+=(--mount "$mounted")
done
for extra in "${extra_mounts[@]}"; do
  receipt_mount_args+=(--mount "$extra")
done

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
  -e "PYTORCH_CUDA_ALLOC_CONF=${alloc_conf}"
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
  -e "HF_HOME=/hf-cache"
  -e "HF_HUB_CACHE=/hf-cache/hub"
  -e "TOKENIZERS_PARALLELISM=false"
  -e "HF_HUB_DISABLE_TELEMETRY=1"
  -e "HF_HUB_DISABLE_PROGRESS_BARS=1"
)
if ((net)); then
  common+=(-e "HF_HUB_OFFLINE=0" -e "TRANSFORMERS_OFFLINE=0")
else
  common+=(-e "HF_HUB_OFFLINE=1" -e "TRANSFORMERS_OFFLINE=1")
fi
if [[ "$court" == "eval" ]]; then
  common+=(-e "HF_DATASETS_CACHE=/datasets-cache")
fi

for mount in "${mounts[@]}"; do
  common+=(-v "$mount")
done
if [[ "$network_mode" == "none" ]]; then
  # Offline by construction; cargo must fail loudly instead of hanging on DNS.
  common+=(-e "CARGO_NET_OFFLINE=true")
fi

detach_note=""
if ((detach)); then
  detach_note=" detach=pending"
fi
printf 'court: %s image=%s memory=%s cpus=%s pids=%s tmp=%s network=%s gpu=%s%s\n' \
  "$court" "$image_ref" "$mem" "$cpus" "$pids" "$tmp_size" "$network_mode" "$gpu_mode" \
  "$detach_note" >&2

# The receipt arguments are built once and reused by the foreground path and by
# the detached `wait` path, so a detached run cannot drift from a supervised one.
inspect_abs="$repo_root/results/courts/.tmp/${rand}.inspect.json"
declare -a receipt_static=(
  --receipt-out "$repo_root/$receipt_rel"
  --court "$court"
  --stamp "$stamp"
  --started-utc "$started_utc"
  --revision "$revision"
  --worktree-dirty "$worktree_dirty"
  --container-name "$container_name"
  --inspect "$inspect_abs"
  --image-ref "$image_ref"
  --image-id "$image_id"
  --image-digest "$image_digest"
  --image-digest-source "$image_digest_source"
  --memory "$mem"
  --cpus "$cpus"
  --pids "$pids"
  --tmpfs-size "$tmp_size"
  --nofile "$nofile"
  --network "$network_mode"
  --gpu-mode "$gpu_mode"
  --gpu-detail "$gpu_detail"
  --env "PYTORCH_CUDA_ALLOC_CONF=${alloc_conf}"
  --user "${host_uid}:${host_gid}"
  "${receipt_mount_args[@]}"
  --host-uname "$(uname -srm)"
  --docker-client "$docker_client"
  --docker-server "$docker_server"
  --cgroup-version "$cgroup_version"
  --log "$log_rel"
)

if ((detach)); then
  # -------------------------------------------------------------------------
  # detached launch: the container is not tied to this shell, so the training
  # survives a session loss. `wait` finishes it and writes the receipt.
  # -------------------------------------------------------------------------
  pending_dir="$repo_root/results/courts/pending"
  mkdir -p "$pending_dir" || die "cannot create $pending_dir" 73
  state_abs="$pending_dir/${container_name}.json"
  printf '%s\0' "$receipt_writer" "${receipt_static[@]}" >"$tmpdir/receipt_args"
  printf '%s\0' "${argv[@]}" >"$tmpdir/argv"
  python3 - "$state_abs" "$container_name" "$log_rel" "$receipt_rel" \
    "$toolchain_abs" "$inspect_abs" "$stamp" "$started_utc" "$revision" \
    "$tmpdir/receipt_args" "$tmpdir/argv" <<'PY' || die "cannot write the pending-state file" 73
import json, os, sys
(state_path, container_name, log_rel, receipt_rel, toolchain_abs, inspect_abs,
 stamp, started_utc, revision, args_path, argv_path) = sys.argv[1:12]

def read_nul(path):
    with open(path, "rb") as handle:
        parts = handle.read().split(b"\0")
    if parts and parts[-1] == b"":
        parts.pop()
    return [item.decode() for item in parts]

state = {
    "schema": "selvedge/court-detached/v1",
    "container_name": container_name,
    "log_rel": log_rel,
    "receipt_rel": receipt_rel,
    "toolchain_abs": toolchain_abs,
    "inspect_abs": inspect_abs,
    "stamp": stamp,
    "started_utc": started_utc,
    "revision": revision,
    "receipt_args": read_nul(args_path),
    "argv": read_nul(argv_path),
}
tmp = state_path + ".tmp"
with open(tmp, "w", encoding="utf-8") as handle:
    json.dump(state, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
os.replace(tmp, state_path)
PY
  container_id=$(docker run -d "${common[@]}" "${gpu_args[@]}" "$image_ref" \
    /bin/sh -c 'if [ -n "${SELVEDGE_TOOLCHAIN_OUT:-}" ]; then /bin/sh '"$toolchain_probe_script"' > "$SELVEDGE_TOOLCHAIN_OUT" 2>&1 || true; fi; exec "$@"' \
    selvedge-court "${argv[@]}") || die "detached docker run failed for $container_name" 69
  printf 'court: %s detached container=%s state=%s log=%s\n' \
    "$court" "$container_id" "$state_abs" "$repo_root/$log_rel" >&2
  printf '%s\n' "$state_abs"
  exit 0
fi

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
inspect_file="$inspect_abs"
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
python3 "$receipt_writer" "${receipt_static[@]}" \
  --finished-utc "$finished_utc" \
  --toolchain "$toolchain_path" \
  --exit-code "$rc" \
  --duration-ms "$duration_ms" \
  -- "${argv[@]}"
receipt_rc=$?
set -e

rm -f "$toolchain_abs" "$inspect_abs"

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
