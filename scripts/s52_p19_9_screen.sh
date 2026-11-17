#!/usr/bin/env bash
# S52: P19.9 counterfactual pre-read router screen.
#
#   SELVEDGE_MEM=2g scripts/s52_p19_9_screen.sh frame
#   SELVEDGE_MEM=12g scripts/s52_p19_9_screen.sh rollouts
#   SELVEDGE_MEM=4g scripts/s52_p19_9_screen.sh train
#   SELVEDGE_MEM=12g scripts/s52_p19_9_screen.sh eval
#   SELVEDGE_MEM=2g scripts/s52_p19_9_screen.sh screen
#   scripts/s52_p19_9_screen.sh --dry-run rollouts
#
# The rollouts and eval stages are GPU jobs and refuse to start unless the
# corrected 20M bar has a receipt and a real-gradient first-moment validation
# receipt exists. The Python driver enforces that ordering and records both
# receipt hashes in its own receipt.
#
# The limit is mandatory and must be derived from this workload's own smoke
# peak: the 2M memory arm's peak is not measured yet, so a first smoke run is
# required and its cgroup memory.peak is the sizing precedent. The script
# refuses to start if the host does not have limit + 4 GiB available.
set -Eeuo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
repo_root=$(cd -- "$script_dir/.." && pwd -P)
cd "$repo_root"

stage="${1-}"
dry_run=0
if [[ "${1-}" == "--dry-run" ]]; then
  dry_run=1
  stage="${2-}"
fi
case "$stage" in
  frame | rollouts | train | eval | screen) ;;
  -h | --help)
    sed -n '2,20p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    exit 0
    ;;
  *)
    printf 's52-p19.9: stage must be frame, rollouts, train, eval or screen\n' >&2
    exit 64
    ;;
esac

[[ -n "${SELVEDGE_MEM:-}" ]] || {
  if ((dry_run == 0)); then
    printf 's52-p19.9: set SELVEDGE_MEM from this workload smoke peak (limit + 4 GiB must be available)\n' >&2
    exit 78
  fi
}

rollout_dir="results/stage2/p19.9/rollouts-$(date -u +%Y%m%dT%H%M%SZ)"
train_dir="results/stage2/p19.9/train-$(date -u +%Y%m%dT%H%M%SZ)"
eval_dir="results/stage2/p19.9/eval-$(date -u +%Y%m%dT%H%M%SZ)"

case "$stage" in
  frame)
    cmd=(python3 scripts/s52_p19_9_router.py frame
      --out "results/stage2/s52-p19.9-frame-$(date -u +%Y%m%dT%H%M%SZ).json")
    court=(scripts/court oracle)
    cpus=2; pids=512; tmp=512m
    ;;
  rollouts)
    latest_rollouts=$(ls -1d results/stage2/p19.9/rollouts-* 2>/dev/null | tail -1 || true)
    rollout_dir="${latest_rollouts:-$rollout_dir}"
    cmd=(python3 scripts/s52_p19_9_router.py rollouts --out-dir "$rollout_dir")
    court=(scripts/court --gpu oracle)
    cpus=8; pids=1024; tmp=2g
    ;;
  train)
    latest_rollouts=$(ls -1d results/stage2/p19.9/rollouts-* 2>/dev/null | tail -1 || true)
    [[ -n "$latest_rollouts" ]] || {
      printf 's52-p19.9: no rollouts directory; run the rollouts stage first\n' >&2
      exit 78
    }
    latest_train=$(ls -1d results/stage2/p19.9/train-* 2>/dev/null | tail -1 || true)
    train_dir="${latest_train:-$train_dir}"
    cmd=(python3 scripts/s52_p19_9_router.py train
      --rollouts "$latest_rollouts/rollouts.json" --out-dir "$train_dir")
    court=(scripts/court oracle)
    cpus=4; pids=512; tmp=1g
    ;;
  eval)
    latest_rollouts=$(ls -1d results/stage2/p19.9/rollouts-* 2>/dev/null | tail -1 || true)
    latest_train=$(ls -1d results/stage2/p19.9/train-* 2>/dev/null | tail -1 || true)
    [[ -n "$latest_rollouts" && -n "$latest_train" ]] || {
      printf 's52-p19.9: eval needs rollouts and a trained router\n' >&2
      exit 78
    }
    first_moment=int8
    validation=$(ls -1 results/stage2/s50-real-gradient-validation-*.json 2>/dev/null | tail -1 || true)
    if [[ -n "$validation" ]] && grep -q 'switch-first-moment-to-bfloat16' "$validation"; then
      first_moment=bfloat16
    fi
    cmd=(python3 scripts/s52_p19_9_router.py eval
      --rollouts "$latest_rollouts/rollouts.json"
      --router-outcome "$latest_train/router-outcome.pt"
      --router-ce "$latest_train/router-ce.pt"
      --first-moment "$first_moment"
      --out "results/stage2/p19.9/eval-$(date -u +%Y%m%dT%H%M%SZ)/eval.json")
    court=(scripts/court --gpu oracle)
    cpus=8; pids=1024; tmp=2g
    ;;
  screen)
    latest_train=$(ls -1d results/stage2/p19.9/train-* 2>/dev/null | tail -1 || true)
    latest_eval=$(ls -1d results/stage2/p19.9/eval-* 2>/dev/null | tail -1 || true)
    [[ -n "$latest_train" && -n "$latest_eval" ]] || {
      printf 's52-p19.9: screen needs a train and an eval receipt\n' >&2
      exit 78
    }
    cmd=(python3 scripts/s52_p19_9_router.py screen
      --train "$latest_train/train.json"
      --eval "$latest_eval/eval.json"
      --out "results/stage2/s52-p19.9-screen-$(date -u +%Y%m%dT%H%M%SZ).json")
    court=(scripts/court oracle)
    cpus=2; pids=512; tmp=512m
    ;;
esac

if ((dry_run)); then
  printf 's52-p19.9: stage=%s cpus=%s pids=%s tmp=%s mem=%s\n' "$stage" \
    "$cpus" "$pids" "$tmp" "${SELVEDGE_MEM:-<unset: required at launch>}"
  printf 's52-p19.9: command:\n  '
  printf '%q ' "${court[@]}" -- "${cmd[@]}"
  printf '\n'
  exit 0
fi

available=$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo)
need=$(python3 -c 'import re,sys; print(int(re.match(r"([0-9]+)", sys.argv[1]).group(1)))' "$SELVEDGE_MEM")
if ((available < need + 4)); then
  printf 's52-p19.9: host MemAvailable %s GiB < limit %s GiB + 4 GiB reserve\n' \
    "$available" "$need" >&2
  exit 75
fi

SELVEDGE_MEM="$SELVEDGE_MEM" SELVEDGE_CPUS="$cpus" SELVEDGE_PIDS="$pids" \
  SELVEDGE_TMP="$tmp" "${court[@]}" -- "${cmd[@]}"
