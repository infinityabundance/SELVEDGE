#!/usr/bin/env bash
# S49: score the bar rerun's milestones on the anchored harness.
#
#   SELVEDGE_MEM=6g scripts/s49_bar_milestones.sh --milestone m5
#   SELVEDGE_MEM=6g scripts/s49_bar_milestones.sh
#   scripts/s49_bar_milestones.sh --dry-run
#
# Milestones are the registered ones: m5 = ckpt-003000 (6,144,000 tokens),
# m10 = ckpt-005000 (10,240,000 tokens), m20 = ckpt-009765 (19,998,720 tokens).
# Each is scored by the amended total-cap driver at the full basis, with the
# anchor instrument's SDPA attention and the S47 sequence-correctness block.
#
# The 6 GiB container is 1.83x the 3.269 GiB cgroup peak measured for this same
# harness and model during the S48 bar score; the limit is still mandatory and
# is checked against that measured peak.
set -Eeuo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
repo_root=$(cd -- "$script_dir/.." && pwd -P)
cd "$repo_root"

run_dir="results/stage2/runs/full-20m-math-s49"
eval_root="results/stage2/eval/full-20m-math-s49"
registration="docs/ws19-anchored-subset-amendment-preregistration-v2.json"
milestone=""
dry_run=0
while (($#)); do
  case "$1" in
    --milestone) milestone="${2-}"; shift 2 ;;
    --milestone=*) milestone="${1#*=}"; shift ;;
    --dry-run) dry_run=1; shift ;;
    -h | --help) sed -n '2,14p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) printf 's49-milestones: unknown argument %s\n' "$1" >&2; exit 64 ;;
  esac
done

[[ -f "$registration" ]] || { printf 's49-milestones: missing registration %s\n' "$registration" >&2; exit 78; }

declare -A step_of=([m5]=3000 [m10]=5000 [m20]=9765)
declare -A tokens_of=([m5]=6144000 [m10]=10240000 [m20]=19998720)
if [[ -n "$milestone" ]]; then
  [[ -n "${step_of[$milestone]:-}" ]] || {
    printf 's49-milestones: --milestone must be m5, m10 or m20\n' >&2
    exit 64
  }
  names=("$milestone")
else
  names=(m5 m10 m20)
fi

[[ -n "${SELVEDGE_MEM:-}" ]] || {
  if ((dry_run == 0)); then
    printf 's49-milestones: set SELVEDGE_MEM (6g is 1.83x the measured 3.269 GiB eval peak)\n' >&2
    exit 78
  fi
}

for name in "${names[@]}"; do
  step="${step_of[$name]}"
  checkpoint=$(printf '%s/ckpt-%06d.pt' "$run_dir" "$step")
  out="$eval_root/$name/eval.json"
  if [[ ! -f "$checkpoint" ]]; then
    printf 's49-milestones: %s missing %s\n' "$name" "$checkpoint" >&2
    exit 78
  fi
  cmd=(python3 scripts/s43_score_total_cap.py
    --arm full-distill
    --full-checkpoint "$checkpoint"
    --tasks mmlu_pro,mmlu,gsm8k,minerva_math500
    --total-cap-map mmlu=14042
    --registration "$registration"
    --vram-limit-gib 12.5
    --out "$out")
  if ((dry_run)); then
    printf 's49-milestones: %s limits: SELVEDGE_MEM=%s SELVEDGE_CPUS=8 SELVEDGE_PIDS=2048 SELVEDGE_TMP=2g VRAM=12.5 GiB\n' \
      "$name" "${SELVEDGE_MEM:-<unset: required at launch>}"
    printf 's49-milestones: %s command:\n  ' "$name"
    printf '%q ' scripts/court --gpu eval -- "${cmd[@]}"
    printf '\n'
    continue
  fi
  if [[ -f "$out" ]]; then
    printf 's49-milestones: %s already scored at %s; not rescored\n' "$name" "$out"
    continue
  fi
  mkdir -p "$(dirname "$out")"
  available=$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo)
  need=$(python3 -c 'import re,sys; v=sys.argv[1]; print(int(re.match(r"([0-9]+)", v).group(1)))' "$SELVEDGE_MEM")
  if ((available < need + 4)); then
    printf 's49-milestones: host MemAvailable %s GiB < limit %s GiB + 4 GiB reserve\n' "$available" "$need" >&2
    exit 75
  fi
  SELVEDGE_MEM="$SELVEDGE_MEM" SELVEDGE_CPUS=8 SELVEDGE_PIDS=2048 SELVEDGE_TMP=2g \
    scripts/court --gpu eval -- "${cmd[@]}"
  # The receipt is written by the scorer; check the checkpoint identity it
  # recorded rather than trusting this script's mapping of name to step.
  python3 - "$out" "$step" "$name" <<'PY'
import json, sys
out, step, name = sys.argv[1], int(sys.argv[2]), sys.argv[3]
payload = json.load(open(out, encoding="utf-8"))
recorded = (payload.get("arm_report") or {}).get("checkpoint_step")
if recorded is not None and int(recorded) != step:
    raise SystemExit(
        f"s49-milestones: {name} receipt scored checkpoint step {recorded}, expected {step}")
print(f"s49-milestones: {name} receipt recorded checkpoint step {recorded}")
PY
done
