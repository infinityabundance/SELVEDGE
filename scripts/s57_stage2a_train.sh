#!/usr/bin/env bash
# S57: stage-2a verified-teacher-solution SFT of the post-trained 0.8B.
#
# The recipe is the S55 stage-2a amendment with the S56 corrections:
#
#   student  post-trained Qwen3.5-0.8B (2fc06364...) -- full text tower,
#            tied embedding frozen (498,113,344 of 752,393,024 trained);
#   data     accepted teacher solutions + ~13% general replay, one COMPLETE
#            example per row (no boundary crossing), prompt masked, solution+EOS
#            supervised (`--objective sft`, kl_weight 0);
#   optim    Adam8bit block 256, int8 first moment, bf16 second moment,
#            stochastic weight rounding, cosine to 10%, warmup 200,
#            grad-clip 1.0, lr 1e-5, seed 0;
#   schedule at most 2 passes over the accepted corpus; the deployed checkpoint
#            is the minimum of the measured held-out-solution CE curve.
#
# Cadence geometry. The S56 amendment scores at 2M and 5M processed tokens. With
# one row of `seq_len` tokens per example, processed tokens advance in whole
# rows, so the launcher refuses any (seq_len, micro_batch, grad_accum) whose
# per-step token count does not divide 1,000,000. The default below divides it
# exactly:
#
#   seq_len 1000, micro-batch 2, grad-accum 1 -> 2,000 tokens/step
#   1,000,000 / 2,000 = 500 steps -> a checkpoint every 1.000M tokens
#   -> the 2M (step 1000) and 5M (step 2500) checkpoints are exact.
#
# 1000 tokens covers the measured teacher-solution length distribution at the
# 1024-token cap (p99 below the cap, max at the cap); a longer example is dropped
# whole and counted, never split across rows.
#
# Usage:
#   SELVEDGE_MEM=14g scripts/s57_stage2a_train.sh --dry-run
#   SELVEDGE_MEM=14g scripts/s57_stage2a_train.sh --detach
#
# SELVEDGE_MEM sizing: the previous full-arm run (the 20M distillation bar',
# seq_len 512 + online 2B teacher) measured a cgroup peak of 8.76 GB. This run
# loads no teacher and stages the same 0.8B student, so the floor is
# max(1.5 x 8.76, 8.76 + 1) = 13.1 GiB; 14g is the next whole value and is what
# the launch record states.
set -Eeuo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
repo_root=$(cd -- "$script_dir/.." && pwd -P)
cd "$repo_root"

examples="results/artifacts/stage2/s55-stage2a/stage2a-examples.jsonl"
examples_receipt="results/artifacts/stage2/s55-stage2a/stage2a-examples-receipt.json"
run_dir="results/stage2/runs/s57-stage2a"
epochs=2
seq_len=1000
micro_batch=2
grad_accum=1
lr="1e-5"
keep_checkpoints=10
vram_limit="12.5"
detach=0
dry_run=0
while (($#)); do
  case "$1" in
    --examples) examples="${2-}"; shift 2 ;;
    --examples-receipt) examples_receipt="${2-}"; shift 2 ;;
    --run-dir) run_dir="${2-}"; shift 2 ;;
    --epochs) epochs="${2-}"; shift 2 ;;
    --seq-len) seq_len="${2-}"; shift 2 ;;
    --micro-batch) micro_batch="${2-}"; shift 2 ;;
    --grad-accum) grad_accum="${2-}"; shift 2 ;;
    --lr) lr="${2-}"; shift 2 ;;
    --vram-limit-gib) vram_limit="${2-}"; shift 2 ;;
    --detach) detach=1; shift ;;
    --dry-run) dry_run=1; shift ;;
    -h | --help) sed -n '2,44p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) printf 's57-train: unknown argument %s\n' "$1" >&2; exit 64 ;;
  esac
done

[[ -f "$examples" ]] || { printf 's57-train: missing examples %s\n' "$examples" >&2; exit 78; }
[[ -f "$examples_receipt" ]] || { printf 's57-train: missing examples receipt %s\n' "$examples_receipt" >&2; exit 78; }
[[ "$lr" == "1e-5" ]] || {
  printf 's57-train: a non-default lr requires its own registered amendment and a fixed-batch KL gate receipt\n' >&2
  exit 78
}

# Row geometry, computed from the examples file the trainer will read: with
# `--sft-examples-per-row 1` every example that fits in `seq_len` is exactly one
# row, and a longer example is dropped and counted, never split.
read -r rows too_long supervised_tokens < <(python3 - "$examples" "$seq_len" <<'PY'
import json, sys
path, seq_len = sys.argv[1], int(sys.argv[2])
rows = too_long = supervised = 0
with open(path, "r", encoding="utf-8") as handle:
    for line in handle:
        if not line.strip():
            continue
        row = json.loads(line)
        length = len(row["prompt_ids"]) + len(row["solution_ids"]) + 1
        if length > seq_len:
            too_long += 1
            continue
        rows += 1
        supervised += len(row["solution_ids"]) + 1
print(rows, too_long, supervised)
PY
)
printf 's57-train: %s rows fit in seq_len %s; %s examples longer and dropped; %s supervised positions\n' \
  "$rows" "$seq_len" "$too_long" "$supervised_tokens"
((rows > 0)) || { printf 's57-train: no example fits\n' >&2; exit 78; }

tokens_per_step=$((micro_batch * seq_len * grad_accum))
if ((1000000 % tokens_per_step != 0)); then
  printf 's57-train: %s tokens/step does not divide 1,000,000; the 2M/5M cadence would not land on a checkpoint\n' \
    "$tokens_per_step" >&2
  exit 78
fi
ckpt_every=$((1000000 / tokens_per_step))
rows_per_step=$((micro_batch * grad_accum))
if ((rows % rows_per_step != 0)); then
  printf 's57-train: %s rows is not a multiple of %s rows/step; an epoch would end mid-step\n' \
    "$rows" "$rows_per_step" >&2
  exit 78
fi
steps_per_epoch=$((rows / rows_per_step))
steps=$((epochs * steps_per_epoch))
tokens=$((steps * tokens_per_step))
milestone_2m=$((2000000 / tokens_per_step))
milestone_5m=$((5000000 / tokens_per_step))
if ((milestone_2m % ckpt_every != 0 || milestone_5m % ckpt_every != 0)); then
  printf 's57-train: cadence checkpoints are not on the ckpt-every grid\n' >&2
  exit 78
fi

cmd=(python3 oracle/frontier/train.py
  --arm full
  --sft-examples "$examples"
  --sft-examples-per-row 1
  --sft-overflow drop
  --objective sft
  --kl-weight 0
  --tokens "$tokens"
  --seq-len "$seq_len"
  --micro-batch "$micro_batch"
  --grad-accum "$grad_accum"
  --lr "$lr"
  --optimizer adam8bit
  --optim-block 256
  --optim-numerics stochastic
  --schedule cosine
  --min-lr-ratio 0.1
  --warmup 200
  --grad-clip 1.0
  --seed 0
  --ckpt-every "$ckpt_every"
  --keep-checkpoints "$keep_checkpoints"
  --log-every 100
  --run-dir "$run_dir"
  --vram-limit-gib "$vram_limit"
  --abort-update-factor 10.0
  --abort-ce-factor 1.5
  --abort-ce-baseline-steps 100
  --abort-ce-window 50)

if ((dry_run)); then
  printf 's57-train: run_dir=%s epochs=%s rows=%s steps/epoch=%s steps=%s tokens=%s\n' \
    "$run_dir" "$epochs" "$rows" "$steps_per_epoch" "$steps" "$tokens"
  printf 's57-train: seq_len=%s micro=%s accum=%s tokens/step=%s ckpt-every=%s\n' \
    "$seq_len" "$micro_batch" "$grad_accum" "$tokens_per_step" "$ckpt_every"
  printf 's57-train: milestones: step %s (~2.00M), step %s (~5.00M), step %s (final, %s tokens)\n' \
    "$milestone_2m" "$milestone_5m" "$steps" "$tokens"
  printf 's57-train: limits: SELVEDGE_MEM=%s SELVEDGE_CPUS=8 SELVEDGE_PIDS=1024 SELVEDGE_TMP=2g VRAM=%sGiB\n' \
    "${SELVEDGE_MEM:-<unset: required at launch>}" "$vram_limit"
  printf 's57-train: command:\n  '
  printf '%q ' scripts/court --gpu oracle -- "${cmd[@]}"
  printf '\n'
  exit 0
fi

[[ -n "${SELVEDGE_MEM:-}" ]] || {
  printf 's57-train: set SELVEDGE_MEM from the measured peak (floor 13.1g; 14g registered)\n' >&2
  exit 78
}
export SELVEDGE_MEM
mem_bytes=$(python3 -c 'import re,sys; v=sys.argv[1]; n=int(re.match(r"([0-9]+)", v).group(1)); u=v[-1].lower(); print(n*{"k":1024,"m":1048576,"g":1073741824}[u])' "$SELVEDGE_MEM")
available=$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo)
need=$((mem_bytes / 1073741824))
if ((available < need + 4)); then
  printf 's57-train: host MemAvailable %s GiB < limit %s GiB + 4 GiB reserve\n' "$available" "$need" >&2
  exit 75
fi

if pgrep -f 'oracle/frontier/train.py' >/dev/null 2>&1; then
  printf 's57-train: a training process is already running; refusing to compete\n' >&2
  exit 75
fi
if pgrep -f 'scripts/s54_post_trained_eval.py' >/dev/null 2>&1; then
  printf 's57-train: a stage-1 scoring run is active; refusing to share the GPU\n' >&2
  exit 75
fi
if pgrep -f 'scripts/s55_stage2_data.py' >/dev/null 2>&1; then
  printf 's57-train: the teacher-data generation is still running; refusing to share the GPU\n' >&2
  exit 75
fi

python3 - "$run_dir" "$examples" "$examples_receipt" "$epochs" "$tokens" "$steps" \
  "$ckpt_every" "$milestone_2m" "$milestone_5m" "$SELVEDGE_MEM" "$vram_limit" "$rows" \
  "$too_long" "$supervised_tokens" "$tokens_per_step" <<'PY'
import hashlib, json, os, re, sys
from datetime import datetime, timezone
(run_dir, examples, receipt, epochs, tokens, steps, ckpt_every, m2m, m5m, mem,
 vram, rows, too_long, supervised, tokens_per_step) = sys.argv[1:16]


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


n = int(re.match(r"([0-9]+)", mem).group(1))
unit = mem[-1].lower()
mem_bytes = n * {"k": 1024, "m": 1048576, "g": 1073741824}[unit]
os.makedirs(run_dir, exist_ok=True)
record = {
    "schema": "selvedge/s57-stage2a-launch/v1",
    "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "run_dir": run_dir,
    "registration": [
        "docs/ws19-post-trained-route-stage2a-amendment-s55-v1.json",
        "docs/ws19-post-trained-route-stage2a-amendment-s56-v1.json",
    ],
    "data": {
        "examples": {"path": examples, "sha256": sha256(examples)},
        "examples_receipt": {"path": receipt, "sha256": sha256(receipt)},
        "rows_fit": int(rows),
        "examples_longer_than_seq_len_dropped": int(too_long),
        "supervised_positions_in_rows": int(supervised),
        "packing": "one complete example per row; no boundary crossing",
        "completion_only_loss": True,
    },
    "recipe": {
        "epochs": int(epochs),
        "tokens_processed": int(tokens),
        "steps": int(steps),
        "tokens_per_step": int(tokens_per_step),
        "optimizer": "Adam8bit block 256, int8 first moment, bf16 second moment, "
                     "stochastic weight rounding",
        "lr": 1e-5,
        "lr_basis": "the S50 fixed-batch KL gate at 1e-5 (0.2529 -> 0.2145); "
                    "1e-4 rose (0.2529 -> 0.5473)",
        "schedule": "cosine to 10%, warmup 200, grad-clip 1.0",
        "seed": 0,
        "tied_embedding": "frozen",
        "early_stop": "at most two passes; the deployed checkpoint is the "
                      "minimum of the held-out-solution-CE curve measured on "
                      "every saved checkpoint",
    },
    "cadence": {
        "checkpoint_every_steps": int(ckpt_every),
        "points": [int(m2m), int(m5m), int(steps)],
        "tokens": [int(m2m) * int(tokens_per_step),
                   int(m5m) * int(tokens_per_step),
                   int(steps) * int(tokens_per_step)],
        "rule": "a checkpoint every 1,000,000 processed tokens; 2M and 5M land "
                "on exact checkpoints",
    },
    "limits": {
        "container_memory_bytes": mem_bytes,
        "container_memory_swap_bytes": mem_bytes,
        "container_cpus": 8,
        "container_pids": 1024,
        "container_tmp": "2g",
        "vram_limit_gib": float(vram),
        "sizing": "floor max(1.5 x 8.76 GB previous full-arm peak, +1 GiB) = "
                  "13.1 GiB; 14g used. No teacher is loaded in this run, so the "
                  "peak is expected below the bar's.",
    },
    "launcher": "scripts/s57_stage2a_train.sh",
}
with open(os.path.join(run_dir, "launch.json"), "w", encoding="utf-8") as handle:
    json.dump(record, handle, indent=2, sort_keys=True)
    handle.write("\n")
print(f"s57-train: launch record -> {run_dir}/launch.json")
PY

extra=()
if ((detach)); then
  extra=(--detach)
fi
SELVEDGE_CPUS=8 SELVEDGE_PIDS=1024 SELVEDGE_TMP=2g \
  exec scripts/court "${extra[@]}" --gpu oracle -- "${cmd[@]}"
