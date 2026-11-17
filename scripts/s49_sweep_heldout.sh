#!/usr/bin/env bash
# S49: held-out likelihood for ONE stability-sweep arm, paired against base on
# the registered 3,906-chunk slice.
#
#   SELVEDGE_MEM=12g scripts/s49_sweep_heldout.sh --arm stability-lr3e-5-2m
#   scripts/s49_sweep_heldout.sh --arm stability-lr3e-5-2m --dry-run
#
# The arm and its run directory come from the S49 registration; the checkpoint
# is the arm's final one (step 976). Base is recomputed in the same pass so the
# paired per-chunk delta is against the same tokens and the same teacher.
set -Eeuo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
repo_root=$(cd -- "$script_dir/.." && pwd -P)
cd "$repo_root"

registration="docs/ws19-full-distillation-amendment-stability-s49-preregistration-v1.json"
heldout_manifest="results/stage2/stage2-heldout-20261117T024542Z.json"
arm=""
dry_run=0
while (($#)); do
  case "$1" in
    --arm) arm="${2-}"; shift 2 ;;
    --arm=*) arm="${1#*=}"; shift ;;
    --dry-run) dry_run=1; shift ;;
    -h | --help) sed -n '2,12p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) printf 's49-heldout: unknown argument %s\n' "$1" >&2; exit 64 ;;
  esac
done

[[ -f "$registration" ]] || { printf 's49-heldout: missing registration %s\n' "$registration" >&2; exit 78; }
[[ -n "$arm" ]] || { printf 's49-heldout: --arm is required\n' >&2; exit 64; }
[[ -f "$heldout_manifest" ]] || { printf 's49-heldout: missing held-out manifest %s\n' "$heldout_manifest" >&2; exit 78; }

read_arm() {
  python3 - "$registration" "$arm" "$1" <<'PY'
import json, sys
registration, name, field = sys.argv[1:4]
for entry in json.load(open(registration, encoding="utf-8"))["stability_sweep"]["arms"]:
    if entry["name"] == name:
        print(entry[field]); raise SystemExit(0)
raise SystemExit(f"unknown arm {name}")
PY
}

if [[ -f "$(python3 - "$registration" "$arm" <<'PY'
import json, sys
registration, name = sys.argv[1:3]
for entry in json.load(open(registration, encoding="utf-8"))["stability_sweep"]["arms"]:
    if entry["name"] == name:
        print(entry["run_dir"] + "/abort.json"); raise SystemExit(0)
raise SystemExit("unknown arm")
PY
)" ]]; then
  printf 's49-heldout: %s recorded an abort; a recorded abort is not scored\n' "$arm" >&2
  exit 78
fi

run_dir=$(read_arm run_dir)
steps=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["stability_sweep"]["common"]["steps"])' "$registration")
checkpoint=$(printf '%s/ckpt-%06d.pt' "$run_dir" "$steps")
[[ -f "$checkpoint" ]] || {
  printf 's49-heldout: missing checkpoint %s; train the arm first\n' "$checkpoint" >&2
  exit 78
}

out="results/stage2/s49-heldout-${arm}.json"
cmd=(python3 scripts/s40_compare.py
  --arms "base,full-distill" --reference-arm base
  --heldout-manifest "$heldout_manifest"
  --full-checkpoint "$checkpoint"
  --registration "$registration"
  --vram-limit-gib 12.5
  --out "$out")

if ((dry_run)); then
  printf 's49-heldout: arm=%s run_dir=%s checkpoint=%s\n' "$arm" "$run_dir" "$checkpoint"
  printf 's49-heldout: limits: SELVEDGE_MEM=%s SELVEDGE_CPUS=8 SELVEDGE_PIDS=2048 SELVEDGE_TMP=2g VRAM=12.5 GiB\n' \
    "${SELVEDGE_MEM:-<unset: required at launch>}"
  printf 's49-heldout: command:\n  '
  printf '%q ' scripts/court --gpu oracle -- "${cmd[@]}"
  printf '\n'
  exit 0
fi

[[ -n "${SELVEDGE_MEM:-}" ]] || {
  printf 's49-heldout: set SELVEDGE_MEM from the measured peak (student+teacher+arm)\n' >&2
  exit 78
}
available=$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo)
need=$(python3 -c 'import re,sys; v=sys.argv[1]; print(int(re.match(r"([0-9]+)", v).group(1)))' "$SELVEDGE_MEM")
if ((available < need + 4)); then
  printf 's49-heldout: host MemAvailable %s GiB < limit %s GiB + 4 GiB reserve\n' "$available" "$need" >&2
  exit 75
fi
if pgrep -f 'oracle/frontier/train.py' >/dev/null 2>&1; then
  printf 's49-heldout: a training process is running; refusing to compete\n' >&2
  exit 75
fi

SELVEDGE_CPUS=8 SELVEDGE_PIDS=2048 SELVEDGE_TMP=2g \
  exec scripts/court --gpu oracle -- "${cmd[@]}"
