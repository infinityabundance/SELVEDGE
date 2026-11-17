#!/usr/bin/env bash
# S48: score ONE trained 2M screen arm — held-out CE/KL and gate statistics
# first, then the anchored math tasks with the S47 sequence-correctness block.
#
#   SELVEDGE_MEM=8g scripts/s48_screen_eval.sh --arm corrective-kl-2m-math
#   scripts/s48_screen_eval.sh --arm gated-dense-2m-math --dry-run
#
# The checkpoint is the arm's final one (step 976). The anchored run uses the
# same harness, attention kernel and basis as the arm table; only the two math
# tasks are scored, because the screen does not measure the knowledge family and
# says so in its registration.
set -Eeuo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
repo_root=$(cd -- "$script_dir/.." && pwd -P)
cd "$repo_root"

registration="docs/ws19-stage2b-screen-preregistration-v1.json"
heldout_manifest="results/stage2/stage2-heldout-20261117T024542Z.json"
arm=""
dry_run=0
while (($#)); do
  case "$1" in
    --arm) arm="${2-}"; shift 2 ;;
    --arm=*) arm="${1#*=}"; shift ;;
    --dry-run) dry_run=1; shift ;;
    -h | --help) sed -n '2,12p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) printf 's48-eval: unknown argument %s\n' "$1" >&2; exit 64 ;;
  esac
done

if [[ ! -f "$registration" ]]; then
  printf 's48-eval: missing registration %s\n' "$registration" >&2
  exit 78
fi

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

if [[ -z "$arm" ]] || ! python3 - "$registration" "$arm" <<'PY'
import json, sys
raise SystemExit(0 if sys.argv[2] in json.load(open(sys.argv[1], encoding="utf-8"))["arms"] else 1)
PY
then
  printf 's48-eval: --arm must be one of the registered screen arms\n' >&2
  exit 64
fi

steps=$(read_field common steps)
run_dir=$(read_field arms "$arm" run_dir)
family=$(read_field arms "$arm" family)
checkpoint=$(printf '%s/ckpt-%06d.pt' "$run_dir" "$steps")
if [[ ! -f "$checkpoint" ]]; then
  printf 's48-eval: missing checkpoint %s\n' "$checkpoint" >&2
  exit 78
fi

memory_init=""
while IFS= read -r candidate; do memory_init="$candidate"; done < <(
  ls -1 results/stage2/graft/corrective/*/memory-init.pt 2>/dev/null | sort)

case "$family" in
  corrective)
    [[ -n "$memory_init" ]] || { printf 's48-eval: missing corrective init\n' >&2; exit 78; }
    arm_args=(--memory-init "$memory_init" --memory-checkpoint "$checkpoint")
    ;;
  matched_control_data)
    arm_args=(--matched-adapter-checkpoint "$checkpoint")
    ;;
  matched_control_gate)
    arm_args=(--gated-dense-checkpoint "$checkpoint")
    ;;
  *) printf 's48-eval: unknown arm family %s\n' "$family" >&2; exit 78 ;;
esac

heldout_out="results/stage2/s48-heldout-${arm}.json"
eval_out="results/stage2/eval/${arm}/eval.json"
heldout_cmd=(python3 scripts/s40_compare.py
  --arms "base,${arm}" --reference-arm base
  --heldout-manifest "$heldout_manifest"
  "${arm_args[@]}"
  --registration "$registration"
  --vram-limit-gib 12.5
  --out "$heldout_out")
anchored_cmd=(python3 scripts/s40_score.py
  --arm "$arm"
  --tasks gsm8k,minerva_math500
  "${arm_args[@]}"
  --registration "$registration"
  --vram-limit-gib 12.5
  --out "$eval_out")

if ((dry_run)); then
  printf 's48-eval: arm=%s family=%s checkpoint=%s\n' "$arm" "$family" "$checkpoint"
  printf 's48-eval: held-out command:\n  '
  printf '%q ' scripts/court --gpu oracle -- "${heldout_cmd[@]}"
  printf '\ns48-eval: anchored command:\n  '
  printf '%q ' scripts/court --gpu eval -- "${anchored_cmd[@]}"
  printf '\n'
  exit 0
fi

if [[ -z "${SELVEDGE_MEM:-}" ]]; then
  printf 's48-eval: set SELVEDGE_MEM from the measured peak (held-out: student+teacher+arm; anchored: the harness)\n' >&2
  exit 78
fi
available=$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo)
need=$(python3 -c 'import re,sys; v=sys.argv[1]; print(int(re.match(r"([0-9]+)", v).group(1)))' "$SELVEDGE_MEM")
if ((available < need + 4)); then
  printf 's48-eval: host MemAvailable %s GiB < limit %s GiB + 4 GiB reserve\n' \
    "$available" "$need" >&2
  exit 75
fi
if systemctl --user --quiet is-active selvedge-fulldistill-math-*.service 2>/dev/null; then
  printf 's48-eval: the 20M bar run is still active; refusing to compete with it\n' >&2
  exit 75
fi

mkdir -p "$(dirname "$eval_out")"
SELVEDGE_MEM="$SELVEDGE_MEM" SELVEDGE_CPUS=8 SELVEDGE_PIDS=1024 SELVEDGE_TMP=2g \
  scripts/court --gpu oracle -- "${heldout_cmd[@]}"
SELVEDGE_MEM="$SELVEDGE_MEM" SELVEDGE_CPUS=8 SELVEDGE_PIDS=1024 SELVEDGE_TMP=2g \
  scripts/court --gpu eval -- "${anchored_cmd[@]}"
