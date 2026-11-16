#!/usr/bin/env bash
# P14 courts on the pinned Qwen3-0.6B control (CPU only).
#
#   residency — the same warm decode with three decoded-residency budgets:
#     off (0, the pre-P14.1 path), partial (256 MB) and full (1.5 GB, the whole
#     1.06 GB payload). Reports decode passes per token, residency hit rate,
#     decode p50/p95 and tokens/s.
#
#   prefetch — the same decode with the fetcher off, synchronous and
#     asynchronous, at a 64 MB storage cache so shards are re-read every token.
#     A delayed set repeats the arms with 200 us injected per device read, so
#     the overlap is measurable and the hidden/exposed split is attributable.
#     Reports exposed stall p50/p95, hidden latency, fetched/used/wasted bytes,
#     wall clock and tokens/s.
#
#   prefill — prompt lengths 8, 32 and 128, scalar (the oracle) against batched.
#     Requires identical logits digests, which the collator asserts.
#
# This is a CPU baseline. No GPU number and no GPU overlap is claimed. Every
# step runs in the rust court with an explicit memory limit sized from the
# measured peak (the warm runtime holds ~1.33 GiB of backbone plus the cache
# plus, for the full-residency arm, the 1.06 GB decoded payload) and records
# the cgroup memory.peak in the receipts.
#
# Usage:
#   scripts/runtime-p14-courts.sh [residency|prefetch|prefill|all]
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
  printf 'runtime-p14: bundle %s is missing\n' "$bundle" >&2
  exit 66
fi

# A deterministic 128-token prompt for the prefill length sweep.
long_prompt=$(python3 - <<'PY'
ids = [(1000 + index * 37) % 150000 for index in range(128)]
print(",".join(str(value) for value in ids))
PY
)
short_prompt=$(python3 -c '
ids = [(1000 + index * 37) % 150000 for index in range(128)]
print(",".join(str(v) for v in ids[:8]))
')
medium_prompt=$(python3 -c '
ids = [(1000 + index * 37) % 150000 for index in range(128)]
print(",".join(str(v) for v in ids[:32]))
')

# $1 name, $2 prompt-ids, $3 generated tokens, rest: extra selvedge flags
run_selvedge() {
  local name="$1"
  local ids="$2"
  local max_new="$3"
  shift 3
  local extra
  extra=$(IFS=' '; printf '%s' "$*")
  printf 'runtime-p14: %s %s\n' "$name" "$extra" >&2
  SELVEDGE_MEM="$mem" SELVEDGE_CPUS="$cpus" scripts/docker-court.sh rust -- sh -c "
    set +e
    SELVEDGE_STAMP='$stamp' ./target/release/selvedge run \
      --model '$model_dir' \
      --bundle '/work/$bundle' \
      --mode exact \
      --prompt-ids '$ids' \
      --max-new-tokens $max_new \
      --accounting '/work/$out_dir/$name-$stamp.jsonl' \
      --json $extra > '/work/$out_dir/$name-$stamp.json' 2> '/work/$out_dir/$name-$stamp.err'
    status=\$?
    cat /sys/fs/cgroup/memory.peak > '/work/$out_dir/$name-$stamp.peak'
    exit \$status
  "
}

residency() {
  run_selvedge p14-residency-off "$prompt_ids" 8 \
    --threads 1 --cache-bytes 1200000000 --policy lru --decode-cache-bytes 0
  run_selvedge p14-residency-partial "$prompt_ids" 8 \
    --threads 1 --cache-bytes 1200000000 --policy lru --decode-cache-bytes 268435456
  run_selvedge p14-residency-full "$prompt_ids" 8 \
    --threads 1 --cache-bytes 1200000000 --policy lru --decode-cache-bytes 1500000000
  run_selvedge p14-residency-full-cold "$prompt_ids" 8 \
    --threads 1 --cache-bytes 0 --policy none --decode-cache-bytes 1500000000
  python3 scripts/runtime_p14.py --stamp "$stamp" --dir "$out_dir" \
    --residency p14-residency-off,p14-residency-partial,p14-residency-full,p14-residency-full-cold \
    --out "$out_dir/p14-residency-$stamp.json"
}

prefetch() {
  run_selvedge p14-prefetch-off "$prompt_ids" 8 \
    --threads 1 --cache-bytes 67108864 --policy lru --decode-cache-bytes 0
  run_selvedge p14-prefetch-sync "$prompt_ids" 8 \
    --threads 1 --cache-bytes 67108864 --policy lru --decode-cache-bytes 0 \
    --prefetch --read-order plan
  run_selvedge p14-prefetch-async "$prompt_ids" 8 \
    --threads 1 --cache-bytes 67108864 --policy lru --decode-cache-bytes 0 \
    --async-prefetch --prefetch-depth 8
  run_selvedge p14-prefetch-off-delay "$prompt_ids" 8 \
    --threads 1 --cache-bytes 67108864 --policy lru --decode-cache-bytes 0 \
    --inject-delay-ns 200000
  run_selvedge p14-prefetch-sync-delay "$prompt_ids" 8 \
    --threads 1 --cache-bytes 67108864 --policy lru --decode-cache-bytes 0 \
    --prefetch --read-order plan --inject-delay-ns 200000
  run_selvedge p14-prefetch-async-delay "$prompt_ids" 8 \
    --threads 1 --cache-bytes 67108864 --policy lru --decode-cache-bytes 0 \
    --async-prefetch --prefetch-depth 8 --inject-delay-ns 200000
  python3 scripts/runtime_p14.py --stamp "$stamp" --dir "$out_dir" \
    --prefetch p14-prefetch-off,p14-prefetch-sync,p14-prefetch-async,p14-prefetch-off-delay,p14-prefetch-sync-delay,p14-prefetch-async-delay \
    --out "$out_dir/p14-prefetch-$stamp.json"
}

prefill() {
  for pair in "8:$short_prompt" "32:$medium_prompt" "128:$long_prompt"; do
    length=${pair%%:*}
    ids=${pair#*:}
    run_selvedge "p14-prefill-${length}-scalar" "$ids" 0 \
      --threads 1 --cache-bytes 1500000000 --policy lru --decode-cache-bytes 1500000000 \
      --prefill scalar
    run_selvedge "p14-prefill-${length}-batched" "$ids" 0 \
      --threads 1 --cache-bytes 1500000000 --policy lru --decode-cache-bytes 1500000000 \
      --prefill batched --prefill-batch "$length"
  done
  python3 scripts/runtime_p14.py --stamp "$stamp" --dir "$out_dir" \
    --prefill p14-prefill-8-scalar,p14-prefill-8-batched,p14-prefill-32-scalar,p14-prefill-32-batched,p14-prefill-128-scalar,p14-prefill-128-batched \
    --out "$out_dir/p14-prefill-$stamp.json"
}

case "${1:-all}" in
  residency) residency ;;
  prefetch) prefetch ;;
  prefill) prefill ;;
  all)
    residency
    prefetch
    prefill
    ;;
  -h | --help)
    sed -n '2,28p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    ;;
  *)
    printf 'runtime-p14: unknown argument %q\n' "$1" >&2
    exit 64
    ;;
esac
