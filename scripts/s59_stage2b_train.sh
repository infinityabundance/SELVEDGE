#!/usr/bin/env bash
# S59: stage-2b training launcher (student-policy RFT and its attribution arms).
#
# The recipe is the stage-2b registration
# (`docs/ws19-post-trained-route-stage2b-preregistration-v1.json`), which reuses
# stage 2a's optimiser and geometry with a screened learning rate:
#
#   student  post-trained Qwen3.5-0.8B (2fc06364...) -- full text tower,
#            tied embedding frozen (498,113,344 of 752,393,024 trained);
#   data     an arm's examples file (completion-only, prompt masked) + 20%
#            general replay, one COMPLETE example per row;
#   optim    Adam8bit block 256, int8 first moment, bf16 second moment,
#            stochastic weight rounding, cosine to 10%, warmup 200,
#            grad-clip 1.0, seed 0;
#   lr       the registered screen set {3e-6, 1e-5}; anything else is refused.
#
# Cadence geometry (unchanged from S58): seq_len 1250, micro-batch 1, grad-accum
# 2 -> 2,500 processed tokens/step; a checkpoint every 400 steps = 1,000,000
# tokens, so the 2M (step 800) and 5M (step 2000) milestones land on exact
# checkpoints. The launcher refuses a geometry whose 1M grid does not divide.
#
# Memory (AGENTS.md directive 2): the stage-2a run of this exact shape measured
# a cgroup peak of 15,032,385,536 B at a 14g limit and REACHED that limit, so the
# registration sets the floor at peak + 1 GiB = 15.0 GiB and requires a whole-GiB
# limit of at least 16g. A smaller SELVEDGE_MEM is refused, not clamped.
#
# Usage:
#   SELVEDGE_MEM=16g scripts/s59_stage2b_train.sh --dry-run \
#     --arm-label rft-only --examples <examples.jsonl> \
#     --examples-receipt <receipt.json> --run-dir <dir>
#   SELVEDGE_MEM=16g scripts/s59_stage2b_train.sh \
#     --arm-label screen-lr3e-6 --tokens 2000000 --lr 3e-6 --keep-checkpoints 2
set -Eeuo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
repo_root=$(cd -- "$script_dir/.." && pwd -P)
cd "$repo_root"

registration="docs/ws19-post-trained-route-stage2b-preregistration-v1.json"
amendment="docs/ws19-post-trained-route-stage2b-amendment-s59-v2.json"
student_id="Qwen/Qwen3.5-0.8B"
student_revision="2fc06364715b967f1860aea9cf38778875588b17"
student_lock="models/FRONTIER-POST-TRAINED.lock.json"
student_dir="/hf-cache/models/Qwen/Qwen3.5-0.8B/$student_revision"
arm_label=""
examples=""
examples_receipt=""
run_dir=""
tokens=""
lr=""
epochs=1
seq_len=1250
micro_batch=1
grad_accum=2
keep_checkpoints=4
vram_limit="13.0"
gradient_checkpointing=1
abort_update_factor="10.0"
logits_gather=1
detach=0
dry_run=0
while (($#)); do
  case "$1" in
    --arm-label) arm_label="${2-}"; shift 2 ;;
    --examples) examples="${2-}"; shift 2 ;;
    --examples-receipt) examples_receipt="${2-}"; shift 2 ;;
    --student-dir) student_dir="${2-}"; shift 2 ;;
    --run-dir) run_dir="${2-}"; shift 2 ;;
    --tokens) tokens="${2-}"; shift 2 ;;
    --epochs) epochs="${2-}"; shift 2 ;;
    --seq-len) seq_len="${2-}"; shift 2 ;;
    --micro-batch) micro_batch="${2-}"; shift 2 ;;
    --grad-accum) grad_accum="${2-}"; shift 2 ;;
    --lr) lr="${2-}"; shift 2 ;;
    --keep-checkpoints) keep_checkpoints="${2-}"; shift 2 ;;
    --vram-limit-gib) vram_limit="${2-}"; shift 2 ;;
    --gradient-checkpointing) gradient_checkpointing=1; shift ;;
    --no-gradient-checkpointing) gradient_checkpointing=0; shift ;;
    --abort-update-factor) abort_update_factor="${2-}"; shift 2 ;;
    --sft-logits-gather) logits_gather=1; shift ;;
    --no-sft-logits-gather) logits_gather=0; shift ;;
    --detach) detach=1; shift ;;
    --dry-run) dry_run=1; shift ;;
    -h | --help) sed -n '2,30p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) printf 's59-train: unknown argument %s\n' "$1" >&2; exit 64 ;;
  esac
done

[[ -f "$registration" ]] || { printf 's59-train: missing registration %s\n' "$registration" >&2; exit 78; }
[[ -f "$amendment" ]] || { printf 's59-train: missing amendment %s\n' "$amendment" >&2; exit 78; }
[[ -n "$arm_label" ]] || { printf 's59-train: --arm-label is required\n' >&2; exit 64; }
[[ -n "$examples" && -f "$examples" ]] || { printf 's59-train: missing examples %s\n' "$examples" >&2; exit 78; }
[[ -n "$examples_receipt" && -f "$examples_receipt" ]] || { printf 's59-train: missing examples receipt %s\n' "$examples_receipt" >&2; exit 78; }
[[ -n "$run_dir" ]] || { printf 's59-train: --run-dir is required\n' >&2; exit 64; }
[[ -n "$lr" ]] || { printf 's59-train: --lr is required (registered set: 3e-6, 1e-5)\n' >&2; exit 64; }

case "$lr" in
  3e-6 | 3e-06 | 1e-5 | 1e-05) ;;
  *)
    printf 's59-train: lr=%s is not in the registered screen set {3e-6, 1e-5}\n' "$lr" >&2
    exit 78
    ;;
esac

case "$student_dir" in
  *Qwen3.5-0.8B-Base*)
    printf 's59-train: student %s is the BASE checkpoint; stage 2b trains the post-trained 0.8B\n' "$student_dir" >&2
    exit 78
    ;;
esac
case "$student_dir" in
  */"$student_revision") ;;
  *)
    printf 's59-train: student %s does not end in the registered revision %s; identity is not provable\n' \
      "$student_dir" "$student_revision" >&2
    exit 78
    ;;
esac
[[ -f "$student_lock" ]] || { printf 's59-train: missing identity lock %s\n' "$student_lock" >&2; exit 78; }
if [[ -d /hf-cache ]]; then
  [[ -f "$student_dir/config.json" ]] || {
    printf 's59-train: student dir %s has no config.json\n' "$student_dir" >&2
    exit 78
  }
fi

# Row geometry, computed from the examples file the trainer will read.
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
printf 's59-train: %s rows fit in seq_len %s; %s examples longer and dropped; %s supervised positions\n' \
  "$rows" "$seq_len" "$too_long" "$supervised_tokens"
((rows > 0)) || { printf 's59-train: no example fits\n' >&2; exit 78; }

tokens_per_step=$((micro_batch * seq_len * grad_accum))
if ((1000000 % tokens_per_step != 0)); then
  printf 's59-train: %s tokens/step does not divide 1,000,000; the cadence would not land on a checkpoint\n' \
    "$tokens_per_step" >&2
  exit 78
fi
ckpt_every=$((1000000 / tokens_per_step))
rows_per_step=$((micro_batch * grad_accum))
if ((rows % rows_per_step != 0)); then
  printf 's59-train: %s rows is not a multiple of %s rows/step\n' "$rows" "$rows_per_step" >&2
  exit 78
fi

if [[ -z "$tokens" ]]; then
  # One epoch: every fitting row exactly once, padded to seq_len.
  tokens=$((epochs * rows * seq_len))
  printf 's59-train: %s epoch(s) over %s rows -> %s processed tokens\n' \
    "$epochs" "$rows" "$tokens"
else
  # An explicit budget (the lr screen) must be a whole number of steps.
  if ((tokens % tokens_per_step != 0)); then
    printf 's59-train: --tokens %s is not a multiple of %s\n' "$tokens" "$tokens_per_step" >&2
    exit 78
  fi
  if ((tokens > rows * seq_len)); then
    printf 's59-train: --tokens %s exceeds one epoch (%s tokens); a second pass is not registered\n' \
      "$tokens" "$((rows * seq_len))" >&2
    exit 78
  fi
fi
steps=$((tokens / tokens_per_step))
milestone_2m=$((2000000 / tokens_per_step))
milestone_5m=$((5000000 / tokens_per_step))
final_policy="epoch end"
if ((steps < milestone_5m)); then
  final_policy="epoch ends before the 5M cadence point; the final checkpoint replaces it and the receipt says so"
  printf 's59-train: NOTE %s\n' "$final_policy" >&2
fi

cmd=(python3 oracle/frontier/train.py
  --arm full
  --student-dir "$student_dir"
  --student-lock "$student_lock"
  --student-id "$student_id"
  --student-revision "$student_revision"
  --sft-examples "$examples"
  --sft-examples-per-row 1
  --sft-overflow drop
  --objective sft
  --loss-unit token
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
  --abort-update-factor "$abort_update_factor"
  --abort-ce-factor 1.5
  --abort-ce-baseline-steps 400
  --abort-ce-window 100)
if ((gradient_checkpointing)); then
  cmd+=(--gradient-checkpointing)
fi
if ((logits_gather)); then
  cmd+=(--sft-logits-gather)
fi

if ((dry_run)); then
  printf 's59-train: arm=%s run_dir=%s lr=%s rows=%s steps=%s tokens=%s\n' \
    "$arm_label" "$run_dir" "$lr" "$rows" "$steps" "$tokens"
  printf 's59-train: seq_len=%s micro=%s accum=%s ckpt-every=%s milestones: 2M=step %s, 5M=step %s, final=step %s\n' \
    "$seq_len" "$micro_batch" "$grad_accum" "$ckpt_every" "$milestone_2m" "$milestone_5m" "$steps"
  printf 's59-train: limits: SELVEDGE_MEM=%s (floor 16g) SELVEDGE_CPUS=8 SELVEDGE_PIDS=1024 SELVEDGE_TMP=2g VRAM=%sGiB\n' \
    "${SELVEDGE_MEM:-<unset: required at launch>}" "$vram_limit"
  printf 's59-train: command:\n  '
  printf '%q ' scripts/court --gpu oracle -- "${cmd[@]}"
  printf '\n'
  exit 0
fi

[[ -n "${SELVEDGE_MEM:-}" ]] || {
  printf 's59-train: set SELVEDGE_MEM >= 16g (the stage-2a peak reached its 14g limit)\n' >&2
  exit 78
}
mem_gib=$(python3 -c 'import re,sys; v=sys.argv[1]; print(int(re.match(r"([0-9]+)", v).group(1)) if v[-1].lower()=="g" else 0)' "$SELVEDGE_MEM")
if ((mem_gib < 16)); then
  printf 's59-train: SELVEDGE_MEM=%s is below the registered 16g floor for this shape\n' "$SELVEDGE_MEM" >&2
  exit 78
fi
export SELVEDGE_MEM
available=$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo)
if ((available < mem_gib + 4)); then
  printf 's59-train: host MemAvailable %s GiB < limit %s GiB + 4 GiB reserve\n' "$available" "$mem_gib" >&2
  exit 75
fi
if pgrep -f 'oracle/frontier/train.py' >/dev/null 2>&1; then
  printf 's59-train: a training process is already running; refusing to compete\n' >&2
  exit 75
fi
if pgrep -f 'scripts/s59_student_rollouts.py' >/dev/null 2>&1; then
  printf 's59-train: the student-policy generation is still running; refusing to share the GPU\n' >&2
  exit 75
fi
if pgrep -f 'scripts/s54_post_trained_eval.py' >/dev/null 2>&1; then
  printf 's59-train: a stage-1 scoring run is active; refusing to share the GPU\n' >&2
  exit 75
fi
if [[ -f "$run_dir/latest.pt" ]]; then
  printf 's59-train: %s already has a checkpoint; refusing to overwrite a run\n' "$run_dir" >&2
  exit 78
fi

python3 - "$run_dir" "$examples" "$examples_receipt" "$epochs" "$tokens" "$steps" \
  "$ckpt_every" "$milestone_2m" "$milestone_5m" "$SELVEDGE_MEM" "$vram_limit" "$rows" \
  "$too_long" "$supervised_tokens" "$tokens_per_step" "$student_dir" "$lr" \
  "$arm_label" "$registration" "$keep_checkpoints" "$final_policy" \
  "$gradient_checkpointing" "$logits_gather" "$abort_update_factor" \
  "$amendment" <<'PY'
import hashlib, json, os, re, sys
from datetime import datetime, timezone
(run_dir, examples, receipt, epochs, tokens, steps, ckpt_every, m2m, m5m, mem,
 vram, rows, too_long, supervised, tokens_per_step, student_dir, lr, arm_label,
 registration, keep, final_policy, grad_ckpt, logits_gather,
 abort_update_factor, amendment) = sys.argv[1:26]


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


mem_bytes = int(re.match(r"([0-9]+)", mem).group(1)) * 1073741824
os.makedirs(run_dir, exist_ok=True)
record = {
    "schema": "selvedge/s59-stage2b-launch/v1",
    "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "run_dir": run_dir,
    "arm_label": arm_label,
    "registration": registration,
    "amendments": [amendment],
    "data": {
        "examples": {"path": examples, "sha256": sha256(examples)},
        "examples_receipt": {"path": receipt, "sha256": sha256(receipt)},
        "rows_fit": int(rows),
        "examples_longer_than_seq_len_dropped": int(too_long),
        "supervised_positions_in_rows": int(supervised),
        "packing": "one complete example per row; no boundary crossing",
        "completion_only_loss": True,
        "replay_fraction": 0.20,
    },
    "student": {
        "dir": student_dir,
        "id": "Qwen/Qwen3.5-0.8B",
        "revision": "2fc06364715b967f1860aea9cf38778875588b17",
        "note": "post-trained chat checkpoint, not the Base model",
    },
    "recipe": {
        "epochs": int(epochs),
        "tokens_processed": int(tokens),
        "steps": int(steps),
        "tokens_per_step": int(tokens_per_step),
        "optimizer": "Adam8bit block 256, int8 first moment, bf16 second "
                     "moment, stochastic weight rounding",
        "lr": float(lr),
        "lr_basis": "registered stage-2b lr screen set {3e-6, 1e-5}; the full "
                     "run uses the rate with the lower held-out solution CE",
        "abort_update_factor": float(abort_update_factor),
        "schedule": "cosine to 10%, warmup 200, grad-clip 1.0",
        "seed": 0,
        "tied_embedding": "frozen",
        "kept_checkpoints": int(keep),
        "gradient_checkpointing": bool(int(grad_ckpt)),
        "gradient_checkpointing_note": (
            "exact recomputation of each block's forward; the S59 smoke measured "
            "step-1 loss 0.228619 against 0.228619 ungathered, i.e. equal "
            "within 1e-8, not bit-identical (different GEMM shapes)"
            if int(grad_ckpt) else "off"),
        "sft_logits_gather": bool(int(logits_gather)),
        "sft_logits_gather_note": (
            "the LM head emits logits only at supervised positions; masked "
            "positions contribute no loss or gradient. The S59 smoke measured "
            "step-1 loss 0.22861921042 (gathered) against 0.22861921787 "
            "(ungathered), a 7.5e-9 absolute difference from the GEMM shape, "
            "not bit-identity"
            if int(logits_gather) else "off"),
    },
    "cadence": {
        "checkpoint_every_steps": int(ckpt_every),
        "points": [int(m2m), int(m5m), int(steps)],
        "tokens": [int(m2m) * int(tokens_per_step),
                   int(m5m) * int(tokens_per_step),
                   int(steps) * int(tokens_per_step)],
        "final_policy": final_policy,
    },
    "limits": {
        "container_memory_bytes": mem_bytes,
        "container_memory_swap_bytes": mem_bytes,
        "container_cpus": 8,
        "container_pids": 1024,
        "container_tmp": "2g",
        "vram_limit_gib": float(vram),
        "sizing": "floor peak + 1 GiB from the stage-2a cgroup peak "
                  "15,032,385,536 B; 16g minimum enforced by the launcher",
    },
    "launcher": "scripts/s59_stage2b_train.sh",
}
with open(os.path.join(run_dir, "launch.json"), "w", encoding="utf-8") as handle:
    json.dump(record, handle, indent=2, sort_keys=True)
    handle.write("\n")
print(f"s59-train: launch record -> {run_dir}/launch.json")
PY

extra=()
if ((detach)); then
  extra=(--detach)
fi
SELVEDGE_CPUS=8 SELVEDGE_PIDS=1024 SELVEDGE_TMP=2g \
  exec scripts/court "${extra[@]}" --gpu oracle -- "${cmd[@]}"
