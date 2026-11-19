#!/usr/bin/env bash
# S66: exact S58 stage-2a replication, changing only the random seed.
#
# This launcher deliberately reproduces the S58 recipe as it ran at revision
# e5381f5, including the two later-identified defects:
#
#   * loss unit: example (microbatch mean / grad_accum), *not* the S60 token
#     unit.  Current train.py resolves `auto` to `token` for SFT, so this
#     launcher passes `--loss-unit example` explicitly.
#   * row window: seq_len 1250, micro-batch 1, grad-accum 2.  S60 moved this to
#     2500 with length-aware batching; neither is used here.
#
# Everything else is the S58 launch: the same examples file and hash, the same
# deterministic row order (no shuffle), no length-aware batching, no gradient
# checkpointing, no SFT logits gather, Adam8bit block 256 stochastic rounding,
# lr 1e-5, cosine to 10%, warmup 200, grad-clip 1.0, 10,940,000 tokens,
# checkpoint every 400 steps, and the same abort tripwires.  Only `--seed` and
# `--run-dir` differ.
#
# The corrected recipe is a separate future experiment and is not mixed in.
#
# Usage:
#   SELVEDGE_MEM=14g scripts/s66_replicate_stage2a.sh --seed 1 --dry-run
#   SELVEDGE_MEM=14g scripts/s66_replicate_stage2a.sh --seed 1 --detach
set -Eeuo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
repo_root=$(cd -- "$script_dir/.." && pwd -P)
cd "$repo_root"

examples="results/artifacts/stage2/s58-combined/stage2a-combined-examples.jsonl"
examples_sha="9426e36d92f8d09ed62289b8b97d79ce4971b37090754eeb6eed038330d275b7"
student_dir="/hf-cache/models/Qwen/Qwen3.5-0.8B/2fc06364715b967f1860aea9cf38778875588b17"
seed=""
run_dir=""
detach=0
dry_run=0
vram_limit="13.5"

while (($#)); do
  case "$1" in
    --seed) seed="${2-}"; shift 2 ;;
    --run-dir) run_dir="${2-}"; shift 2 ;;
    --vram-limit-gib) vram_limit="${2-}"; shift 2 ;;
    --detach) detach=1; shift ;;
    --dry-run) dry_run=1; shift ;;
    -h | --help) sed -n '2,46p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) printf 's66-replicate: unknown argument %s\n' "$1" >&2; exit 64 ;;
  esac
done

[[ "$seed" =~ ^[0-9]+$ ]] || { printf 's66-replicate: --seed must be a non-negative integer\n' >&2; exit 64; }
[[ "$seed" != "0" ]] || { printf 's66-replicate: seed 0 is the S58 run; use 1 or 2\n' >&2; exit 64; }
[[ -n "$run_dir" ]] || run_dir="results/stage2/runs/s66-stage2a-seed${seed}"
[[ -f "$examples" ]] || { printf 's66-replicate: missing examples %s\n' "$examples" >&2; exit 78; }
actual_sha=$(sha256sum "$examples" | awk '{print $1}')
[[ "$actual_sha" == "$examples_sha" ]] || {
  printf 's66-replicate: examples hash %s != pinned %s\n' "$actual_sha" "$examples_sha" >&2
  exit 78
}
if [[ -d /hf-cache ]]; then
  [[ -d "$student_dir" ]] || {
    printf 's66-replicate: student dir %s is absent on the host\n' "$student_dir" >&2
    exit 78
  }
else
  printf 's66-replicate: /hf-cache is not mounted on the host; the student path is verified inside the court\n' >&2
fi
if [[ -e "$run_dir" && -n "$(ls -A "$run_dir" 2>/dev/null || true)" ]]; then
  printf 's66-replicate: run dir %s is not empty; refusing to clobber evidence\n' "$run_dir" >&2
  exit 73
fi
if pgrep -f 'oracle/frontier/train.py' >/dev/null 2>&1; then
  printf 's66-replicate: a training process is already running; refusing to compete\n' >&2
  exit 75
fi
if pgrep -f 'scripts/s54_post_trained_eval.py' >/dev/null 2>&1; then
  printf 's66-replicate: a stage-1 scoring run is active; refusing to share the GPU\n' >&2
  exit 75
fi

cmd=(python3 oracle/frontier/train.py
  --arm full
  --student-dir "$student_dir"
  --sft-examples "$examples"
  --sft-examples-per-row 1
  --sft-overflow drop
  --objective sft
  --loss-unit example
  --kl-weight 0
  --tokens 10940000
  --seq-len 1250
  --micro-batch 1
  --grad-accum 2
  --lr 1e-5
  --optimizer adam8bit
  --optim-block 256
  --optim-numerics stochastic
  --schedule cosine
  --min-lr-ratio 0.1
  --warmup 200
  --grad-clip 1.0
  --seed "$seed"
  --ckpt-every 400
  --keep-checkpoints 10
  --log-every 100
  --run-dir "$run_dir"
  --vram-limit-gib "$vram_limit"
  --abort-update-factor 10.0
  --abort-ce-factor 1.5
  --abort-ce-baseline-steps 400
  --abort-ce-window 100)

if ((dry_run)); then
  printf 's66-replicate: seed=%s run_dir=%s\n' "$seed" "$run_dir"
  printf 's66-replicate: limits SELVEDGE_MEM=%s SELVEDGE_CPUS=8 SELVEDGE_PIDS=1024 SELVEDGE_TMP=2g VRAM=%sGiB\n' \
    "${SELVEDGE_MEM:-<unset: required at launch>}" "$vram_limit"
  printf 's66-replicate: command:\n  '
  printf '%q ' scripts/court --gpu oracle -- "${cmd[@]}"
  printf '\n'
  exit 0
fi

[[ -n "${SELVEDGE_MEM:-}" ]] || {
  printf 's66-replicate: set SELVEDGE_MEM from the measured S58 peak (14g)\n' >&2
  exit 78
}
mem_bytes=$(python3 -c 'import re,sys; v=sys.argv[1]; n=int(re.match(r"([0-9]+)", v).group(1)); u=v[-1].lower(); print(n*{"k":1024,"m":1048576,"g":1073741824}[u])' "$SELVEDGE_MEM")
available=$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo)
need=$((mem_bytes / 1073741824))
if ((available < need + 4)); then
  printf 's66-replicate: host MemAvailable %s GiB < limit %s GiB + 4 GiB reserve\n' \
    "$available" "$need" >&2
  exit 75
fi

mkdir -p "$run_dir"
python3 - "$run_dir" "$seed" "$examples" "$examples_sha" "$SELVEDGE_MEM" \
  "$vram_limit" "$student_dir" <<'PY'
import json, os, sys, time
from datetime import datetime, timezone

run_dir, seed, examples, examples_sha, mem, vram, student = sys.argv[1:8]
record = {
    "schema": "selvedge/s66-stage2a-replication-launch/v1",
    "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "run_dir": run_dir,
    "purpose": ("seed replication of the exact S58 stage-2a recipe; only the "
                "random seed changes"),
    "source_run": {
        "run_dir": "results/stage2/runs/s58-stage2a",
        "receipt": "results/stage2/runs/s58-stage2a/receipt.json",
        "receipt_sha256": None,
        "selected_checkpoint": "results/stage2/runs/s58-stage2a/ckpt-004000.pt",
        "selected_checkpoint_sha256": "33ee31ae5586799b2544ce4b888ae09c4f2654ab11b03301d05720ddb0bb7fc6",
    },
    "data": {
        "examples": {"path": examples, "sha256": examples_sha},
        "row_order": "deterministic row order, cycled; no shuffling",
        "loss_unit": "example",
        "seq_len": 1250,
        "micro_batch": 1,
        "grad_accum": 2,
        "tokens_per_step": 2500,
    },
    "recipe": {
        "tokens": 10940000,
        "steps": 4376,
        "checkpoint_every_steps": 400,
        "optimizer": ("Adam8bit block 256, int8 first moment, bf16 second "
                      "moment, stochastic weight rounding"),
        "lr": 1e-5,
        "schedule": "cosine to 10%, warmup 200, grad-clip 1.0",
        "seed": int(seed),
        "tied_embedding": "frozen",
        "gradient_checkpointing": False,
        "length_aware_batching": False,
        "sft_logits_gather": False,
        "kl_weight": 0.0,
        "abort_tripwires": {
            "update_factor": 10.0,
            "ce_factor": 1.5,
            "ce_baseline_steps": 400,
            "ce_window": 100,
        },
    },
    "known_defects_held_constant": {
        "example_weighted_loss": True,
        "seq_len_1250_window": True,
        "why": ("the replication must change only the seed; the S60 token-unit "
                "and length-aware-window corrections belong to a separate "
                "future experiment and are not mixed in"),
    },
    "student": {
        "dir": student,
        "id": "Qwen/Qwen3.5-0.8B",
        "revision": "2fc06364715b967f1860aea9cf38778875588b17",
    },
    "limits": {
        "container_memory": mem,
        "container_memory_swap": mem,
        "container_cpus": 8,
        "container_pids": 1024,
        "container_tmp": "2g",
        "vram_limit_gib": float(vram),
        "sizing": "S58 measured cgroup memory.max 15,032,385,536 and peak 12.73 GiB allocated / 13.09 GiB reserved; 14g is the registered arm limit",
    },
    "launcher": "scripts/s66_replicate_stage2a.sh",
}
source_receipt = "results/stage2/runs/s58-stage2a/receipt.json"
if os.path.exists(source_receipt):
    import hashlib
    digest = hashlib.sha256()
    with open(source_receipt, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    record["source_run"]["receipt_sha256"] = digest.hexdigest()
with open(os.path.join(run_dir, "launch.json"), "w", encoding="utf-8") as handle:
    json.dump(record, handle, indent=2, sort_keys=True)
    handle.write("\n")
print(f"s66-replicate: launch record -> {run_dir}/launch.json")
PY

extra=()
if ((detach)); then extra=(--detach); fi
SELVEDGE_CPUS=8 SELVEDGE_PIDS=1024 SELVEDGE_TMP=2g \
  exec scripts/court "${extra[@]}" --gpu oracle -- "${cmd[@]}"
