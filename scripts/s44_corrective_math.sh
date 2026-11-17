#!/usr/bin/env bash
# Stage-2b: launch (or resume) ONE corrective arm on the S43 math-weighted
# 20M stream. The arm, its weighting, budget, seeds and objective weights are
# read from the registration, not from this script's defaults:
#
#   docs/ws19-stage2b-corrective-advantage-preregistration-v1.json
#
#   SELVEDGE_MEM=10g scripts/s44_corrective_math.sh --weighting kl
#   SELVEDGE_MEM=10g scripts/s44_corrective_math.sh --weighting signed
#   SELVEDGE_MEM=10g scripts/s44_corrective_math.sh --weighting signed-true
#   SELVEDGE_MEM=10g scripts/s44_corrective_math.sh --weighting both
#   scripts/s44_corrective_math.sh --weighting kl --dry-run      # print, do not run
#
# SELVEDGE_MEM is mandatory and must come from the arm's own 20-step smoke cgroup
# peak (1.5x, never below measured + 1 GiB). There is no fallback to the court
# default. The script refuses to launch while another memory-heavy job holds the
# GPU: stage-2b runs after the 20M bar has finished and been scored.
set -Eeuo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
repo_root=$(cd -- "$script_dir/.." && pwd -P)
cd "$repo_root"

registration="docs/ws19-stage2b-corrective-advantage-preregistration-v1.json"
weighting=""
dry_run=0
while (($#)); do
  case "$1" in
    --weighting) weighting="${2-}"; shift 2 ;;
    --weighting=*) weighting="${1#*=}"; shift ;;
    --dry-run) dry_run=1; shift ;;
    -h | --help) sed -n '2,20p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) printf 's44-corrective: unknown argument %s\n' "$1" >&2; exit 64 ;;
  esac
done

case "$weighting" in
  kl | signed | signed-exp | signed-true | both) ;;
  *) printf 's44-corrective: --weighting must be kl|signed|signed-exp|signed-true|both\n' >&2; exit 64 ;;
esac

if [[ ! -f "$registration" ]]; then
  printf 's44-corrective: missing registration %s\n' "$registration" >&2
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

arm="corrective-${weighting}-20m-math"
run_dir=$(read_field "arms" "$arm" "run_dir")
stream_manifest=$(read_field stream manifest)
stream_sha=$(read_field stream sha256)
tokens=$(read_field common tokens)
seq_len=$(read_field common sequence_length)
micro_batch=$(read_field common micro_batch)
grad_accum=$(read_field common gradient_accumulation)
lr=$(read_field common lr)
warmup=$(read_field common warmup_steps)
grad_clip=$(read_field common grad_clip)
seed=$(read_field common seed)
temperature=$(read_field objective temperature)
kl_weight=$(read_field objective kl_weight)
preserve_weight=$(read_field objective preserve_weight)
interference_weight=$(read_field objective interference_weight)

if [[ ! -f "$stream_manifest" ]]; then
  printf 's44-corrective: missing stream manifest %s\n' "$stream_manifest" >&2
  exit 78
fi
recorded=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["sha256"])' "$stream_manifest")
if [[ "$recorded" != "$stream_sha" ]]; then
  printf 's44-corrective: stream %s records %s, registration says %s\n' \
    "$stream_manifest" "$recorded" "$stream_sha" >&2
  exit 78
fi

# The corrective init is built once for all three weightings (see the
# registration's `init` block). Fail closed with the builder command if absent.
memory_init=""
while IFS= read -r candidate; do memory_init="$candidate"; done < <(
  ls -1 results/stage2/graft/corrective/*/memory-init.pt 2>/dev/null | sort)
if [[ -z "$memory_init" ]]; then
  if [[ "$dry_run" == "0" ]]; then
    printf 's44-corrective: no corrective init under results/stage2/graft/corrective/*/memory-init.pt\n' >&2
    printf 's44-corrective: build it first (needs the GPU, ~minutes):\n' >&2
    printf '  %s\n' "$(read_field init command)" >&2
    exit 78
  fi
  memory_init="<corrective-init-not-built-yet>"
fi

if [[ "$dry_run" == "0" ]]; then
  if [[ -z "${SELVEDGE_MEM:-}" ]]; then
    printf 's44-corrective: set SELVEDGE_MEM from the arm smoke cgroup peak (1.5x, at least +1 GiB)\n' >&2
    exit 78
  fi
  export SELVEDGE_MEM
  # Reserve memory explicitly: the bar arm holds about 10 GiB until it exits.
  available=$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo)
  need=$(python3 -c 'import re,sys; v=sys.argv[1]; print(int(re.match(r"([0-9]+)", v).group(1)))' "$SELVEDGE_MEM")
  if ((available < need + 4)); then
    printf 's44-corrective: host MemAvailable %s GiB < limit %s GiB + 4 GiB reserve\n' \
      "$available" "$need" >&2
    exit 75
  fi
  if systemctl --user --quiet is-active selvedge-fulldistill-math-*.service 2>/dev/null; then
    printf 's44-corrective: the 20M bar run is still active; stage-2b waits for it\n' >&2
    exit 75
  fi
fi

mkdir -p "$run_dir"
resume=()
if [[ -f "$run_dir/latest.pt" ]]; then
  resume=(--resume latest)
  printf 's44-corrective: resuming from %s/latest.pt\n' "$run_dir"
else
  printf 's44-corrective: fresh run in %s\n' "$run_dir"
fi

cmd=(
  scripts/court --gpu oracle -- python3 oracle/frontier/train.py
  --arm corrective
  --objective correction
  --advantage-weighting "$weighting"
  --stream-manifest "$stream_manifest"
  --memory-init "$memory_init"
  --tokens "$tokens"
  --seq-len "$seq_len"
  --micro-batch "$micro_batch"
  --grad-accum "$grad_accum"
  --lr "$lr"
  --temperature "$temperature"
  --kl-weight "$kl_weight"
  --preserve-weight "$preserve_weight"
  --interference-weight "$interference_weight"
  --optimizer adamw
  --schedule constant
  --warmup "$warmup"
  --grad-clip "$grad_clip"
  --seed "$seed"
  --ckpt-every 1000
  --keep-checkpoints 2
  --log-every 100
  --run-dir "$run_dir"
  --vram-limit-gib 12.5
)

if [[ "$dry_run" == "1" ]]; then
  printf 's44-corrective: arm=%s weighting=%s run_dir=%s\n' "$arm" "$weighting" "$run_dir"
  printf 's44-corrective: init=%s\n' "$memory_init"
  printf 's44-corrective: stream=%s sha256=%s tokens=%s steps=%s\n' \
    "$stream_manifest" "$stream_sha" "$tokens" "$((tokens / (seq_len * micro_batch * grad_accum)))"
  printf 's44-corrective: limits: SELVEDGE_MEM=%s SELVEDGE_CPUS=8 SELVEDGE_PIDS=1024 SELVEDGE_TMP=2g VRAM=12.5 GiB\n' \
    "${SELVEDGE_MEM:-<unset: required at launch>}"
  printf 's44-corrective: command:\n  '
  printf '%q ' "${cmd[@]}"
  printf '\n'
  exit 0
fi

SELVEDGE_CPUS=8 SELVEDGE_PIDS=1024 SELVEDGE_TMP=2g exec "${cmd[@]}"
