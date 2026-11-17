#!/usr/bin/env bash
# Stage-2b sequencing: score the 20M bar, measure the corrective arm's own
# limits with a 20-step smoke, then run the three corrective arms one at a time
# under systemd. It does nothing unless --execute is passed:
#
#   scripts/s44_stage2b_queue.sh --plan        # print the exact sequence
#   scripts/s44_stage2b_queue.sh --execute     # require the bar to be finished
#
# Order and arguments come from
# docs/ws19-stage2b-corrective-advantage-preregistration-v1.json. The bar is
# scored BEFORE any corrective arm starts, so the win bar is a measured number.
#
# Memory policy (AGENTS.md directive 2): the corrective arms' container limit is
# derived from their own smoke receipt, not from the court default and not from
# the bar's limit. The bar's 11.43 GiB VRAM peak is a property of the full
# backbone arm (gradients + 8-bit Adam on 498M elements); the corrective arm has
# the memory arm's shape (frozen backbone, 539,649 trainable elements), measured
# at 9.211 GiB allocated / 9.363 GiB reserved. The gate adds 2,049 parameters
# (16 KiB) and a (B, T) float per site, i.e. kilobytes at this batch, so the
# 12.5 GiB VRAM limit is not expected to bind. If the smoke measures otherwise,
# the ladder in the registration applies (seq 512->384 with accum 4->5).
set -Eeuo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
repo_root=$(cd -- "$script_dir/.." && pwd -P)
cd "$repo_root"

registration="docs/ws19-stage2b-corrective-advantage-preregistration-v1.json"
bar_run="results/stage2/runs/full-20m-math"
bar_eval="results/stage2/eval/full-20m-math/eval.json"
stream_manifest="results/stage2/stage2-math-stream-20261117T071120Z.json"
smoke_dir="results/stage2/runs/smoke-corrective-math"
mode="plan"
while (($#)); do
  case "$1" in
    --plan) mode="plan"; shift ;;
    --execute) mode="execute"; shift ;;
    -h | --help) sed -n '2,22p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) printf 's44-queue: unknown argument %s\n' "$1" >&2; exit 64 ;;
  esac
done

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

weightings=(kl signed both)
init_command=$(read_field init command)

bar_checkpoint="$bar_run/latest.pt"
if [[ ! -f "$bar_checkpoint" ]]; then
  bar_checkpoint="$bar_run/ckpt-009765.pt"
fi

score_bar=(
  scripts/court --gpu eval -- python3 scripts/s40_score.py
  --arm full-distill
  --registration docs/ws19-anchored-subset-preregistration-v1.json
  --tasks mmlu_pro,mmlu,gsm8k,minerva_math500
  --vram-limit-gib 12.5
  --full-checkpoint "$bar_checkpoint"
  --out "$bar_eval"
)
# No --limit-map: the effective basis of every stage-2 receipt is the full
# 14,042-item MMLU (amendment v2), so a cap would change the comparison rather
# than reproduce it. A future subset must go through s43_score_total_cap.py.

smoke() {
  local init="$1"
  SELVEDGE_CPUS=8 SELVEDGE_PIDS=1024 SELVEDGE_TMP=2g \
    scripts/court --gpu oracle -- python3 oracle/frontier/train.py \
    --arm corrective --objective correction --advantage-weighting kl \
    --stream-manifest "$stream_manifest" \
    --memory-init "$init" \
    --tokens 40960 --seq-len 512 --micro-batch 1 --grad-accum 4 \
    --lr 0.001 --temperature 2.0 --kl-weight 0.5 \
    --preserve-weight 0.5 --interference-weight 0.05 \
    --optimizer adamw --schedule constant --warmup 2 --grad-clip 1.0 --seed 0 \
    --ckpt-every 1000 --keep-checkpoints 1 --log-every 5 \
    --run-dir "$smoke_dir" --vram-limit-gib 12.5
}

smoke_label=(
  "<corrective init>"
)

plan() {
  printf 'stage-2b plan (nothing has been launched)\n'
  printf '  1. build the corrective init if absent:\n     %s\n' "$init_command"
  printf '  2. wait for the bar unit to go inactive, then score it:\n'
  printf '       scripts/court --gpu eval -- python3 scripts/s40_score.py --arm full-distill \\\n'
  printf '         --registration docs/ws19-anchored-subset-preregistration-v1.json \\\n'
  printf '         --tasks mmlu_pro,mmlu,gsm8k,minerva_math500 --vram-limit-gib 12.5 \\\n'
  printf '         --full-checkpoint %s --out %s\n' "$bar_checkpoint" "$bar_eval"
  printf '  3. 20-step corrective smoke to size the container:\n'
  printf '       scripts/court --gpu oracle -- python3 oracle/frontier/train.py --arm corrective \\\n'
  printf '         --objective correction --advantage-weighting kl --stream-manifest %s \\\n' "$stream_manifest"
  printf '         --memory-init <init> --tokens 40960 --seq-len 512 --micro-batch 1 --grad-accum 4 \\\n'
  printf '         --lr 0.001 --schedule constant --warmup 2 --run-dir %s --vram-limit-gib 12.5\n' "$smoke_dir"
  printf '     SELVEDGE_MEM = max(1.5 x cgroup peak, peak + 1 GiB), rounded up to a whole GiB\n'
  for weighting in "${weightings[@]}"; do
    printf '  4.%s systemd-run one %s arm, wait for it, then continue:\n' "$weighting" "$weighting"
    printf '       systemd-run --user --collect --unit=selvedge-corrective-%s-20m-math-<stamp> \\\n' "$weighting"
    printf '         --property=MemoryMax="${SELVEDGE_MEM}" --property=MemorySwapMax="${SELVEDGE_MEM}" \\\n'
    printf '         --property=TasksMax=1024 --property=CPUQuota=800%% \\\n'
    printf '         scripts/s44_corrective_math.sh --weighting %s\n' "$weighting"
  done
  printf '  5. score each finished arm and re-render the table:\n'
  printf '       scripts/court --gpu eval -- python3 scripts/s40_score.py --arm corrective-<w>-20m-math \\\n'
  printf '         --registration %s --tasks mmlu_pro,mmlu,gsm8k,minerva_math500 \\\n' "$registration"
  printf '         --memory-init <init> --memory-checkpoint results/stage2/runs/corrective-<w>-20m-math/latest.pt \\\n'
  printf '         --vram-limit-gib 12.5 --out results/stage2/eval/corrective-<w>-20m-math/eval.json\n'
  printf '       scripts/court python scripts/s44_arm_table.py --write\n'
}

if [[ "$mode" == "plan" ]]; then
  plan
  exit 0
fi

# --execute ----------------------------------------------------------------
if systemctl --user --quiet is-active "selvedge-fulldistill-math-"*.service 2>/dev/null; then
  printf 's44-queue: the 20M bar run is still active; refusing to start stage-2b\n' >&2
  exit 75
fi

if [[ -z "${SELVEDGE_MEM:-}" ]]; then
  printf 's44-queue: run the smoke first and pass its derived limit in SELVEDGE_MEM\n' >&2
  exit 78
fi
export SELVEDGE_MEM

if [[ ! -f "$bar_eval" ]]; then
  printf 's44-queue: scoring the bar\n'
  "${score_bar[@]}"
fi

if [[ ! -f "$smoke_dir/receipt.json" ]]; then
  init=$(ls -1 results/stage2/graft/corrective/*/memory-init.pt 2>/dev/null | sort | tail -1)
  if [[ -z "$init" ]]; then
    printf 's44-queue: no corrective init; build it first:\n  %s\n' "$init_command" >&2
    exit 78
  fi
  printf 's44-queue: running the 20-step smoke\n'
  smoke "$init"
  peak=$(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(int(d["container_memory"]["memory_peak"]))' "$smoke_dir/receipt.json")
  derived=$(python3 -c 'import sys; p=int(sys.argv[1]); print(int(max(p*1.5, p+1073741824)//1073741824)+1)' "$peak")
  printf 's44-queue: smoke cgroup peak %s bytes -> SELVEDGE_MEM=%sg\n' "$peak" "$derived"
  SELVEDGE_MEM="${derived}g"
  export SELVEDGE_MEM
fi

for weighting in "${weightings[@]}"; do
  unit="selvedge-corrective-${weighting}-20m-math-$(date -u +%Y%m%dT%H%M%SZ)"
  printf 's44-queue: launching %s in %s\n' "$weighting" "$unit"
  systemd-run --user --collect --unit="$unit" \
    --property=MemoryMax="$SELVEDGE_MEM" --property=MemorySwapMax="$SELVEDGE_MEM" \
    --property=TasksMax=1024 --property=CPUQuota=800% \
    scripts/s44_corrective_math.sh --weighting "$weighting"
  while systemctl --user --quiet is-active "$unit.service"; do sleep 30; done
  result=$(systemctl --user show "$unit.service" --property=Result --value)
  printf 's44-queue: %s finished with Result=%s\n' "$unit" "$result"
  if [[ "$result" != "success" ]]; then
    printf 's44-queue: stopping; %s did not succeed\n' "$unit" >&2
    exit 1
  fi
  run_dir=$(read_field arms "corrective-${weighting}-20m-math" run_dir)
  scripts/court --gpu eval -- python3 scripts/s40_score.py \
    --arm "corrective-${weighting}-20m-math" \
    --registration "$registration" \
    --tasks mmlu_pro,mmlu,gsm8k,minerva_math500 \
    --memory-init "$(ls -1 results/stage2/graft/corrective/*/memory-init.pt | sort | tail -1)" \
    --memory-checkpoint "$run_dir/latest.pt" \
    --vram-limit-gib 12.5 \
    --out "results/stage2/eval/corrective-${weighting}-20m-math/eval.json"
  scripts/court python scripts/s44_arm_table.py --write
done

printf 's44-queue: stage-2b complete; the table is re-rendered\n'
