#!/usr/bin/env bash
# S50: rerun the 20M full-backbone distillation bar with the corrected optimiser.
#
#   SELVEDGE_MEM=20g scripts/s50_bar_20m.sh [--detach] [--dry-run]
#   SELVEDGE_MEM=20g scripts/s50_bar_20m.sh --lr 3e-5 \
#     --run-dir results/stage2/runs/s50-bar-20m-lr3e-5 \
#     --kl-receipt results/stage2/s50-fixed-batch-kl-math-lr3e-5-<stamp>.json
#
# The recipe is the registered 20M bar (math stream, full text tower with the
# tied embedding frozen, CE + 0.5*T^2*KL, Adam8bit block 256, stochastic weight
# rounding, cosine to 10%, warmup 200, grad-clip 1.0, seed 0, 19,998,720 tokens)
# with one replacement: the corrected bf16 second moment. The default lr = 1e-5
# is the rate whose 200-step revalidation shows a falling fixed-batch teacher KL
# (0.2529 -> 0.2145); 1e-4 did not (0.2529 -> 0.5473), and neither run tripped the
# update tripwire. A non-default --lr is refused unless --kl-receipt names the
# 200-step fixed-batch KL gate that passed at that rate.
#
# SELVEDGE_MEM sizing: the previous 20M bar's cgroup peak was 8.76 GB, so the
# floor is max(1.5 x 8.76, 8.76 + 1) = 13.1 GB; 20g was the registered S49 value
# and is used unchanged. The corrected second moment is GPU-only (+0.456 GiB
# allocated) and does not change host memory.
set -Eeuo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
repo_root=$(cd -- "$script_dir/.." && pwd -P)
cd "$repo_root"

run_dir="results/stage2/runs/s50-bar-20m-lr1e-5"
stream="results/stage2/stage2-math-stream-20261117T071120Z.json"
stream_sha="9c5fb53b20a8f9165566b8c4e237103eb2e4c658f944bb221b3909d1dc8b6d56"
lr="1e-5"
run_dir_set=0
kl_receipt=""
detach=0
dry_run=0
while (($#)); do
  case "$1" in
    --lr) lr="${2-}"; shift 2 ;;
    --lr=*) lr="${1#*=}"; shift ;;
    --run-dir) run_dir="${2-}"; run_dir_set=1; shift 2 ;;
    --run-dir=*) run_dir="${1#*=}"; run_dir_set=1; shift ;;
    --kl-receipt) kl_receipt="${2-}"; shift 2 ;;
    --kl-receipt=*) kl_receipt="${1#*=}"; shift ;;
    --detach) detach=1; shift ;;
    --dry-run) dry_run=1; shift ;;
    -h | --help) sed -n '2,21p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) printf 's50-bar: unknown argument %s\n' "$1" >&2; exit 64 ;;
  esac
done
if ((run_dir_set == 0)); then
  run_dir="results/stage2/runs/s50-bar-20m-lr${lr}"
fi
[[ "$lr" =~ ^[0-9]+e-[0-9]+$ ]] || {
  printf 's50-bar: --lr must look like 1e-5 or 3e-5, got %s\n' "$lr" >&2
  exit 64
}
if [[ "$lr" != "1e-5" ]]; then
  [[ -f "$kl_receipt" ]] || {
    printf 's50-bar: a non-default rate requires --kl-receipt naming the 200-step fixed-batch KL gate receipt; %s is missing\n' "$kl_receipt" >&2
    exit 78
  }
fi

[[ -f "$stream" ]] || { printf 's50-bar: missing stream manifest %s\n' "$stream" >&2; exit 78; }
recorded=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["sha256"])' "$stream")
[[ "$recorded" == "$stream_sha" ]] || {
  printf 's50-bar: stream %s records %s, expected %s\n' "$stream" "$recorded" "$stream_sha" >&2
  exit 78
}

cmd=(python3 oracle/frontier/train.py
  --arm full
  --stream-manifest "$stream"
  --tokens 20000000
  --seq-len 512
  --micro-batch 1
  --grad-accum 4
  --lr "$lr"
  --temperature 2.0
  --kl-weight 0.5
  --optimizer adam8bit
  --optim-block 256
  --optim-numerics stochastic
  --schedule cosine
  --min-lr-ratio 0.1
  --warmup 200
  --grad-clip 1.0
  --seed 0
  --ckpt-every 2441
  --keep-checkpoints 8
  --log-every 200
  --run-dir "$run_dir"
  --vram-limit-gib 12.5
  --abort-update-factor 10.0
  --abort-ce-factor 1.5
  --abort-ce-baseline-steps 100
  --abort-ce-window 50)

if ((dry_run)); then
  printf 's50-bar: run_dir=%s lr=%s tokens=20000000 steps=9765\n' "$run_dir" "$lr"
  printf 's50-bar: milestones (ckpt-every 2441): 2441 (~5.00M), 4882 (~10.00M), 9765 (19.99872M)\n'
  printf 's50-bar: limits: SELVEDGE_MEM=%s SELVEDGE_CPUS=8 SELVEDGE_PIDS=1024 SELVEDGE_TMP=2g VRAM=12.5GiB\n' \
    "${SELVEDGE_MEM:-<unset: required at launch>}"
  printf 's50-bar: command:\n  '
  printf '%q ' scripts/court --gpu oracle -- "${cmd[@]}"
  printf '\n'
  exit 0
fi

[[ -n "${SELVEDGE_MEM:-}" ]] || {
  printf 's50-bar: set SELVEDGE_MEM from the measured long-run peak (floor 14g)\n' >&2
  exit 78
}
export SELVEDGE_MEM
mem_bytes=$(python3 -c 'import re,sys; v=sys.argv[1]; n=int(re.match(r"([0-9]+)", v).group(1)); u=v[-1].lower(); print(n*{"k":1024,"m":1048576,"g":1073741824}[u])' "$SELVEDGE_MEM")
available=$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo)
need=$((mem_bytes / 1073741824))
if ((available < need + 4)); then
  printf 's50-bar: host MemAvailable %s GiB < limit %s GiB + 4 GiB reserve\n' "$available" "$need" >&2
  exit 75
fi
if pgrep -f 'oracle/frontier/train.py' >/dev/null 2>&1; then
  printf 's50-bar: a training process is already running; refusing to compete\n' >&2
  exit 75
fi
if pgrep -f 'scripts/s43_score_total_cap.py' >/dev/null 2>&1; then
  printf 's50-bar: an anchored scoring run is active; refusing to share the GPU\n' >&2
  exit 75
fi

python3 - "$run_dir" "$lr" "$SELVEDGE_MEM" "$stream" "$stream_sha" "$kl_receipt" <<'PY'
import json, os, re, sys
from datetime import datetime, timezone
(run_dir, lr, mem, stream, sha, kl_receipt) = sys.argv[1:7]
n = int(re.match(r"([0-9]+)", mem).group(1))
unit = mem[-1].lower()
mem_bytes = n * {"k": 1024, "m": 1048576, "g": 1073741824}[unit]
if float(lr) == 1e-5:
    rate_selection = (
        "200-step revalidation: at 1e-4 the fixed-batch training KL rose "
        "0.2529 -> 0.5473 (results/stage2/s50-fixed-batch-kl-math-"
        "20261117T205500Z.json); at 1e-5 it fell 0.2529 -> 0.2145 "
        "(results/stage2/s50-fixed-batch-kl-math-lr1e-5-"
        "20261117T205600Z.json). Neither run tripped the update tripwire.")
else:
    import hashlib
    with open(kl_receipt, "rb") as handle:
        digest = hashlib.sha256(handle.read()).hexdigest()
    rate_selection = (
        f"the 200-step fixed-batch KL gate at lr {float(lr):g} must fall "
        f"(receipt {kl_receipt}, sha256 {digest}); this rate is 3x the "
        "revalidated 1e-5 rate, adopted only if the gate passes")
os.makedirs(run_dir, exist_ok=True)
record = {
    "schema": "selvedge/s50-bar-launch/v1",
    "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "run_dir": run_dir,
    "recipe": {
        "lr": float(lr),
        "schedule": "cosine to 10%",
        "warmup_steps": 200,
        "tokens": 20000000,
        "steps": 9765,
        "optimizer": "Adam8bit block 256, stochastic weight rounding, bf16 second moment (S50 fix)",
        "second_moment": "bfloat16, unquantised",
        "first_moment": "int8, block-wise absmax",
        "rate_selection": rate_selection,
    },
    "limits": {
        "container_memory_bytes": mem_bytes,
        "container_memory_swap_bytes": mem_bytes,
        "container_cpus": 8,
        "container_pids": 1024,
        "container_tmp": "2g",
        "vram_limit_gib": 12.5,
        "sizing": "20g = the registered S49 value; floor from the 8.76 GB previous-bar cgroup peak is max(1.5x, +1 GiB) = 13.1 GB",
    },
    "stream": {"manifest": stream, "sha256": sha},
    "checkpoints": {"every_steps": 2441, "keep": 8,
                    "milestones": [2441, 4882, 9765]},
    "tripwire": {"update_factor": 10.0, "ce_factor": 1.5,
                 "ce_baseline_steps": 100, "ce_window_steps": 50},
    "launcher": "scripts/s50_bar_20m.sh",
}
with open(os.path.join(run_dir, "launch.json"), "w", encoding="utf-8") as handle:
    json.dump(record, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY

extra=()
if ((detach)); then
  extra=(--detach)
fi
SELVEDGE_CPUS=8 SELVEDGE_PIDS=1024 SELVEDGE_TMP=2g \
  exec scripts/court "${extra[@]}" --gpu oracle -- "${cmd[@]}"
