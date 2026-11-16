#!/usr/bin/env bash
# Runtime courts on the pinned Qwen3-0.6B control.
#
#   Court E — exact-mode semantic parity: the same model, the same tokens and the
#   same greedy rule under different cache capacity, cache policy, prefetch
#   setting, fetch order, injected device delay and kernel thread count. Every
#   run must produce the same sequence digest (a hash over each token's exact f32
#   logits) and the same generated tokens.
#
#   Court F — CPU latency: cold (no cache) and warm (bundle resident) prefill and
#   decode latency with p50/p95, tokens/s, bytes read per token, at one and eight
#   kernel threads. This is a CPU baseline, never a GPU claim.
#
# Every step runs inside the rust court with an explicit memory limit sized from
# the measured peak: the runtime holds the backbone in host RAM (1.33 GiB in f32)
# plus the bundle cache (1.2 GiB) plus per-group buffers, and each receipt
# records the cgroup `memory.peak` observed by that step.
#
# Usage:
#   scripts/runtime-courts.sh [court-e|court-f|both]
set -Eeuo pipefail
IFS=$'\n\t'

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
repo_root=$(cd -- "$script_dir/.." && pwd -P)
cd "$repo_root"

model_dir=${SELVEDGE_RUNTIME_MODEL:-/hf-cache/models/Qwen/Qwen3-0.6B/c1899de289a04d12100db370d81485cdf75e47ca}
bundle=${SELVEDGE_RUNTIME_BUNDLE:-results/artifacts/qwen3-0.6b-ffn-g16/swiglu-ad15450e2cc2-76dd1b90442e.slv}
prompt_ids=${SELVEDGE_RUNTIME_PROMPT_IDS:-785,6722,315,9625,374}
mem=${SELVEDGE_MEM:-10g}
cpus=${SELVEDGE_CPUS:-8}
stamp=$(date -u +%Y%m%dT%H%M%SZ)
out_dir=results/runtime
mkdir -p "$out_dir"

if [[ ! -f $bundle ]]; then
  printf 'runtime-courts: bundle %s is missing\n' "$bundle" >&2
  exit 66
fi

# $1 name, $2 generated tokens, rest: extra selvedge flags
run_selvedge() {
  local name="$1"
  local max_new="$2"
  shift 2
  local extra
  # IFS is newline/tab in this script so paths with spaces stay intact; join the
  # extra flags with plain spaces here, because the container shell parses them.
  extra=$(IFS=' '; printf '%s' "$*")
  printf 'runtime-courts: %s (max_new_tokens=%s) %s\n' "$name" "$max_new" "$extra" >&2
  SELVEDGE_MEM="$mem" SELVEDGE_CPUS="$cpus" scripts/docker-court.sh rust -- sh -c "
    set +e
    SELVEDGE_STAMP='$stamp' ./target/release/selvedge run \
      --model '$model_dir' \
      --bundle '/work/$bundle' \
      --mode exact \
      --prompt-ids '$prompt_ids' \
      --max-new-tokens $max_new \
      --accounting '/work/$out_dir/$name-$stamp.jsonl' \
      --json $extra > '/work/$out_dir/$name-$stamp.json' 2> '/work/$out_dir/$name-$stamp.err'
    status=\$?
    cat /sys/fs/cgroup/memory.peak > '/work/$out_dir/$name-$stamp.peak'
    exit \$status
  "
}

court_e() {
  run_selvedge court-e-cold 2 --threads 8
  run_selvedge court-e-warm 2 --threads 8 --cache-bytes 1200000000 --policy lru
  run_selvedge court-e-prefetch 2 --threads 4 --cache-bytes 1200000000 --policy lru --prefetch --read-order reverse
  run_selvedge court-e-delay 2 --threads 2 --cache-bytes 67108864 --policy lru --inject-delay-ns 200000
  python3 scripts/runtime_court_e.py \
    --stamp "$stamp" \
    --dir "$out_dir" \
    --names court-e-cold,court-e-warm,court-e-prefetch,court-e-delay \
    --out "$out_dir/court-e-0.6b-$stamp.json"
}

# Real-model parity against the independent NumPy reference. The reference is
# regenerated here (oracle court, offline), then the runtime must reproduce its
# prefill logits within the tolerance pre-registered in docs/RUNTIME.md
# (abs=1e-3, rel=1e-2 on the pinned 0.6B checkpoint).
parity() {
  local ref="$out_dir/qwen3-0.6b-ref-logits.bin"
  printf 'runtime-parity: regenerating the NumPy reference\n' >&2
  SELVEDGE_MEM=12g SELVEDGE_CPUS=8 scripts/docker-court.sh oracle -- \
    python3 oracle/qwen3_numpy_reference.py \
      --model "$model_dir" \
      --prompt-ids "$prompt_ids" \
      --decode 0 \
      --bin "/work/$ref" \
      --meta "/work/$out_dir/qwen3-0.6b-ref-meta-$stamp.json"
  printf 'runtime-parity: comparing the runtime against it\n' >&2
  SELVEDGE_MEM="$mem" SELVEDGE_CPUS="$cpus" scripts/docker-court.sh rust -- sh -c "
    set +e
    SELVEDGE_STAMP='$stamp' ./target/release/selvedge run \
      --model '$model_dir' \
      --bundle '/work/$bundle' \
      --mode exact \
      --prompt-ids '$prompt_ids' \
      --max-new-tokens 1 \
      --threads 8 \
      --cache-bytes 1200000000 \
      --policy lru \
      --compare-logits '/work/$ref' \
      --tolerance-abs 1e-3 \
      --tolerance-rel 1e-2 \
      --accounting '/work/$out_dir/court-e-parity-0.6b-$stamp.jsonl' \
      --json > '/work/$out_dir/court-e-parity-0.6b-$stamp.json' 2> '/work/$out_dir/court-e-parity-0.6b-$stamp.err'
    status=\$?
    cat /sys/fs/cgroup/memory.peak > '/work/$out_dir/court-e-parity-0.6b-$stamp.peak'
    exit \$status
  "
  python3 scripts/runtime_court_parity.py "$out_dir/court-e-parity-0.6b-$stamp.json"
}

court_f() {
  run_selvedge court-f-cold 16 --threads 8
  run_selvedge court-f-warm 16 --threads 8 --cache-bytes 1200000000 --policy lru
  run_selvedge court-f-warm-1t 16 --threads 1 --cache-bytes 1200000000 --policy lru
  python3 scripts/runtime_court_f.py \
    --stamp "$stamp" \
    --dir "$out_dir" \
    --names court-f-cold,court-f-warm,court-f-warm-1t \
    --out "$out_dir/court-f-cpu-$stamp.json"
}

case "${1:-both}" in
  court-e) court_e ;;
  court-f) court_f ;;
  parity) parity ;;
  both)
    court_e
    parity
    court_f
    ;;
  -h | --help)
    sed -n '2,20p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    ;;
  *)
    printf 'runtime-courts: unknown argument %q\n' "$1" >&2
    exit 64
    ;;
esac
