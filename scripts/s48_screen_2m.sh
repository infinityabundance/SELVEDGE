#!/usr/bin/env bash
# S48: build the corrective init or launch ONE 2M screen arm.
#
# Everything about an arm — stream, teacher, budget, optimizer, seeds, run dir —
# is read from the registration, not from this script's defaults:
#
#   docs/ws19-stage2b-screen-preregistration-v1.json
#
#   SELVEDGE_MEM=6g scripts/s48_screen_2m.sh --init            # corrective init (GPU)
#   SELVEDGE_MEM=6g scripts/s48_screen_2m.sh --arm corrective-kl-2m-math
#   SELVEDGE_MEM=6g scripts/s48_screen_2m.sh --arm adapter-2m-math-matched
#   scripts/s48_screen_2m.sh --arm gated-dense-2m-math --dry-run
#
# SELVEDGE_MEM is mandatory and must come from the measured cgroup peak (1.5x,
# never below measured + 1 GiB); the registration states the arithmetic. The
# script refuses to launch while the 20M bar is active or when host memory is
# short, because the screen is serialised behind the bar by registration.
set -Eeuo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
repo_root=$(cd -- "$script_dir/.." && pwd -P)
cd "$repo_root"

registration="docs/ws19-stage2b-screen-preregistration-v1.json"
arm=""
init=0
dry_run=0
while (($#)); do
  case "$1" in
    --arm) arm="${2-}"; shift 2 ;;
    --arm=*) arm="${1#*=}"; shift ;;
    --init) init=1; shift ;;
    --dry-run) dry_run=1; shift ;;
    -h | --help) sed -n '2,20p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) printf 's48-screen: unknown argument %s\n' "$1" >&2; exit 64 ;;
  esac
done

if [[ ! -f "$registration" ]]; then
  printf 's48-screen: missing registration %s\n' "$registration" >&2
  exit 78
fi

read_field() {
  python3 - "$registration" "$@" <<'PY'
import json, sys
path, *keys = sys.argv[1:]
value = json.load(open(path, encoding="utf-8"))
for key in keys:
    value = value[key]
print(value)
PY
}

is_known_arm() {
  python3 - "$registration" "$1" <<'PY'
import json, sys
path, name = sys.argv[1], sys.argv[2]
raise SystemExit(0 if name in json.load(open(path, encoding="utf-8"))["arms"] else 1)
PY
}

stream_manifest=$(read_field stream manifest)
stream_sha=$(read_field stream sha256)
if [[ ! -f "$stream_manifest" ]]; then
  printf 's48-screen: missing stream manifest %s\n' "$stream_manifest" >&2
  exit 78
fi
recorded=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["sha256"])' "$stream_manifest")
if [[ "$recorded" != "$stream_sha" ]]; then
  printf 's48-screen: stream %s records %s, registration says %s\n' \
    "$stream_manifest" "$recorded" "$stream_sha" >&2
  exit 78
fi

tokens=$(read_field common tokens)
seq_len=$(read_field common sequence_length)
micro_batch=$(read_field common micro_batch)
grad_accum=$(read_field common gradient_accumulation)
steps=$(read_field common steps)
lr=$(read_field common lr)
warmup=$(read_field common warmup_steps)
grad_clip=$(read_field common grad_clip)
seed=$(read_field common seed)
temperature=$(read_field teacher temperature)
kl_weight=$(read_field teacher kl_weight)
vram=$(read_field limits vram_limit_gib)

memory_init=""
while IFS= read -r candidate; do memory_init="$candidate"; done < <(
  ls -1 results/stage2/graft/corrective/*/memory-init.pt 2>/dev/null | sort)

if ((init)); then
  if ((dry_run == 0)) && [[ -n "$memory_init" ]]; then
    printf 's48-screen: corrective init already exists: %s\n' "$memory_init"
    exit 0
  fi
  cmd=(python3 oracle/frontier/graft.py --corrective
    --stream-manifest "$stream_manifest" --seed "$seed"
    --vram-limit-gib "$vram" --out-dir results/stage2/graft/corrective)
else
  if [[ -z "$arm" ]] || ! is_known_arm "$arm"; then
    printf 's48-screen: --arm must be one of the registered screen arms\n' >&2
    python3 - "$registration" <<'PY'
import json, sys
for name in json.load(open(sys.argv[1], encoding="utf-8"))["arms"]:
    print(f"  {name}")
PY
    exit 64
  fi
  run_dir=$(read_field arms "$arm" run_dir)
  family=$(read_field arms "$arm" family)
  case "$family" in
    corrective)
      weighting=$(read_field arms "$arm" weighting)
      if [[ -z "$memory_init" ]]; then
        if ((dry_run == 0)); then
          printf 's48-screen: no corrective init under results/stage2/graft/corrective/*/memory-init.pt\n' >&2
          printf 's48-screen: build it first: SELVEDGE_MEM=<smoke peak x1.5> %s --init\n' \
            "${BASH_SOURCE[0]}" >&2
          exit 78
        fi
        memory_init="<corrective-init-not-built-yet>"
      fi
      arm_args=(--arm corrective --objective correction
        --advantage-weighting "$weighting" --memory-init "$memory_init")
      ;;
    matched_control_data)
      arm_args=(--arm adapter-matched --objective ce_kl)
      ;;
    matched_control_gate)
      arm_args=(--arm gated-dense --objective ce_kl)
      ;;
    *) printf 's48-screen: unknown arm family %s\n' "$family" >&2; exit 78 ;;
  esac
  mkdir -p "$run_dir"
  resume=()
  if [[ -f "$run_dir/latest.pt" ]]; then
    resume=(--resume latest)
    printf 's48-screen: resuming from %s/latest.pt\n' "$run_dir"
  else
    printf 's48-screen: fresh run in %s\n' "$run_dir"
  fi
  cmd=(python3 oracle/frontier/train.py
    "${arm_args[@]}"
    --stream-manifest "$stream_manifest"
    --tokens "$tokens"
    --seq-len "$seq_len"
    --micro-batch "$micro_batch"
    --grad-accum "$grad_accum"
    --lr "$lr"
    --temperature "$temperature"
    --kl-weight "$kl_weight"
    --optimizer adamw
    --schedule constant
    --warmup "$warmup"
    --grad-clip "$grad_clip"
    --seed "$seed"
    --ckpt-every 488
    --keep-checkpoints 2
    --log-every 100
    --run-dir "$run_dir"
    --vram-limit-gib "$vram"
    "${resume[@]}")
fi

if ((dry_run)); then
  printf 's48-screen: arm=%s run_dir=%s\n' "${arm:-<init>}" "${run_dir:-<init>}"
  [[ -n "$memory_init" ]] && printf 's48-screen: init=%s\n' "$memory_init"
  printf 's48-screen: stream=%s sha256=%s tokens=%s steps=%s\n' \
    "$stream_manifest" "$stream_sha" "$tokens" "$steps"
  printf 's48-screen: limits: SELVEDGE_MEM=%s SELVEDGE_CPUS=8 SELVEDGE_PIDS=1024 SELVEDGE_TMP=2g VRAM=%s GiB\n' \
    "${SELVEDGE_MEM:-<unset: required at launch>}" "$vram"
  printf 's48-screen: command:\n  '
  printf '%q ' scripts/court --gpu oracle -- "${cmd[@]}"
  printf '\n'
  exit 0
fi

if [[ -z "${SELVEDGE_MEM:-}" ]]; then
  printf 's48-screen: set SELVEDGE_MEM from the measured cgroup peak (1.5x, at least +1 GiB)\n' >&2
  exit 78
fi
export SELVEDGE_MEM
available=$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo)
need=$(python3 -c 'import re,sys; v=sys.argv[1]; print(int(re.match(r"([0-9]+)", v).group(1)))' "$SELVEDGE_MEM")
if ((available < need + 4)); then
  printf 's48-screen: host MemAvailable %s GiB < limit %s GiB + 4 GiB reserve\n' \
    "$available" "$need" >&2
  exit 75
fi
if systemctl --user --quiet is-active selvedge-fulldistill-math-*.service 2>/dev/null; then
  printf 's48-screen: the 20M bar run is still active; the screen waits for it and its score\n' >&2
  exit 75
fi

SELVEDGE_CPUS=8 SELVEDGE_PIDS=1024 SELVEDGE_TMP=2g \
  exec scripts/court --gpu oracle -- "${cmd[@]}"
