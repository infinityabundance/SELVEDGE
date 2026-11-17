#!/usr/bin/env bash
# S41 anchored-harness pass: score the six available arms with one instrument.
#
# The subset and the reasons for it are registered in
# docs/ws19-anchored-subset-preregistration-v1.json before this script runs.
# Every arm uses the same task list, few-shot counts, batch sizes, greedy
# decoding, seeds, bootstrap iterations and extraction filters as the S36
# anchor; only MMLU is capped at 2,000 items (the registration says so).
#
# This script is launched once, under systemd with an explicit memory limit.
# It runs the arms sequentially: the GPU is not shared with a second job.
set -Eeuo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
cd "$repo_root"

registration="docs/ws19-anchored-subset-preregistration-v1.json"
tasks="mmlu_pro,mmlu,gsm8k,minerva_math500"
limit_map="mmlu=2000"
vram_limit="12.5"
memory_init="results/stage2/graft/20261116T231451Z/memory-init.pt"

run_arm() {
  local arm="$1"
  shift
  printf 's41: anchored harness arm %s\n' "$arm"
  scripts/court --gpu eval -- python3 scripts/s40_score.py \
    --arm "$arm" \
    --registration "$registration" \
    --tasks "$tasks" \
    --limit-map "$limit_map" \
    --vram-limit-gib "$vram_limit" \
    --out "results/stage2/eval/$arm/eval.json" \
    "$@"
  printf 's41: anchored harness arm %s done\n' "$arm"
}

run_arm base
run_arm adapter-2m \
  --adapter-checkpoint results/stage2/runs/adapter-2m/ckpt-000976.pt
run_arm memory-2m \
  --memory-init "$memory_init" \
  --memory-checkpoint results/stage2/runs/memory-2m/ckpt-000976.pt
run_arm memory-ungrafted \
  --memory-init "$memory_init"
run_arm memory-grafted \
  --memory-init "$memory_init"
run_arm adapter-20m \
  --control-checkpoint results/stage2/runs/control-20m/ckpt-009765.pt

printf 's41: anchored harness pass complete\n'
