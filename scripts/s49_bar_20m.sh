#!/usr/bin/env bash
# S49: rerun the 20M full-backbone distillation bar with the recipe the
# registered stability sweep chose.
#
#   SELVEDGE_MEM=14g scripts/s49_bar_20m.sh
#   scripts/s49_bar_20m.sh --dry-run
#
# The recipe comes from the newest decision receipt
# (results/stage2/s49-stability-decision-<stamp>.json); this script refuses to
# launch without one, or when the registered rule selected no arm. Everything
# else is the base registration as amended, with the tripwire on, checkpoints
# every 1,000 steps and 10 retained so the 5M/10M/20M milestones survive.
#
# SELVEDGE_MEM is mandatory. The floor is derived from measured peaks of the
# same workload: the previous 20M bar's own cgroup peak (8.761 GiB) and the
# chosen arm's 20-step smoke. The limit must be at least
# max(1.5 x peak, peak + 1 GiB) for each.
set -Eeuo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
repo_root=$(cd -- "$script_dir/.." && pwd -P)
cd "$repo_root"

registration="docs/ws19-full-distillation-amendment-stability-s49-preregistration-v1.json"
dry_run=0
while (($#)); do
  case "$1" in
    --dry-run) dry_run=1; shift ;;
    -h | --help) sed -n '2,20p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) printf 's49-bar: unknown argument %s\n' "$1" >&2; exit 64 ;;
  esac
done

[[ -f "$registration" ]] || { printf 's49-bar: missing registration %s\n' "$registration" >&2; exit 78; }

decision=""
while IFS= read -r candidate; do decision="$candidate"; done < <(
  ls -1 results/stage2/s49-stability-decision-*.json 2>/dev/null | sort)
[[ -n "$decision" ]] || {
  printf 's49-bar: no stability decision receipt; run scripts/s49_stability_table.py --write first\n' >&2
  exit 78
}

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

chosen_json=$(python3 - "$decision" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
if payload.get("second_factor_required") or not payload.get("chosen"):
    raise SystemExit("s49-bar: the registered rule selected no arm; the second factor is required")
chosen = payload["chosen"]
print(f"{chosen['arm']}\t{chosen['lr']}\t{chosen['run_dir']}")
PY
) || exit 78
IFS=$'\t' read -r chosen_arm chosen_lr chosen_dir <<<"$chosen_json"

run_dir=$(read_field bar_rerun run_dir)
stream_manifest=$(read_field bar_rerun stream)
stream_sha=$(read_field bar_rerun stream_sha256)
tokens=$(read_field bar_rerun tokens)
steps=$(read_field bar_rerun steps)
temperature=$(read_field stability_sweep common teacher_temperature)
kl_weight=$(read_field stability_sweep common kl_weight)
optim_block=$(read_field stability_sweep common optim_block_elements)
optim_numerics=$(read_field stability_sweep common optimizer_numerics)
min_lr_ratio=$(read_field stability_sweep common min_lr_ratio)
grad_clip=$(read_field stability_sweep common grad_clip)
seed=$(read_field stability_sweep common seed)
vram=$(read_field stability_sweep common vram_limit_gib)
tripwire_factor=$(read_field stability_sweep early_loss_tripwire factor)

[[ -f "$stream_manifest" ]] || { printf 's49-bar: missing stream manifest %s\n' "$stream_manifest" >&2; exit 78; }
recorded=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["sha256"])' "$stream_manifest")
[[ "$recorded" == "$stream_sha" ]] || {
  printf 's49-bar: stream %s records %s, registration says %s\n' "$stream_manifest" "$recorded" "$stream_sha" >&2
  exit 78
}

mkdir -p "$run_dir"
resume=()
if [[ -f "$run_dir/latest.pt" ]]; then
  resume=(--resume latest)
  printf 's49-bar: resuming from %s/latest.pt\n' "$run_dir"
else
  printf 's49-bar: fresh run in %s\n' "$run_dir"
fi

# Warmup 200 is the registered 20M value; the trainer clamps it to steps//10,
# which is 976 at 9,765 steps, so the registered 200 is honoured unchanged.
cmd=(python3 oracle/frontier/train.py
  --arm full
  --stream-manifest "$stream_manifest"
  --tokens "$tokens"
  --seq-len 512
  --micro-batch 1
  --grad-accum 4
  --lr "$chosen_lr"
  --temperature "$temperature"
  --kl-weight "$kl_weight"
  --optimizer adam8bit
  --optim-block "$optim_block"
  --optim-numerics "$optim_numerics"
  --schedule cosine
  --min-lr-ratio "$min_lr_ratio"
  --warmup 200
  --grad-clip "$grad_clip"
  --seed "$seed"
  --ckpt-every 1000
  --keep-checkpoints 10
  --log-every 100
  --run-dir "$run_dir"
  --vram-limit-gib "$vram"
  --abort-ce-factor "$tripwire_factor"
  --abort-ce-baseline-steps 100
  --abort-ce-window 50
  "${resume[@]}")

if ((dry_run)); then
  printf 's49-bar: decision=%s\n' "$decision"
  printf 's49-bar: chosen arm=%s lr=%s (trained in %s)\n' "$chosen_arm" "$chosen_lr" "$chosen_dir"
  printf 's49-bar: run_dir=%s tokens=%s steps=%s\n' "$run_dir" "$tokens" "$steps"
  printf 's49-bar: limits: SELVEDGE_MEM=%s SELVEDGE_CPUS=8 SELVEDGE_PIDS=1024 SELVEDGE_TMP=2g VRAM=%s GiB\n' \
    "${SELVEDGE_MEM:-<unset: required at launch>}" "$vram"
  printf 's49-bar: command:\n  '
  printf '%q ' scripts/court --gpu oracle -- "${cmd[@]}"
  printf '\n'
  exit 0
fi

[[ -n "${SELVEDGE_MEM:-}" ]] || {
  printf 's49-bar: set SELVEDGE_MEM from the measured long-run peak (registration: 14g)\n' >&2
  exit 78
}
export SELVEDGE_MEM
mem_bytes=$(python3 -c 'import re,sys; v=sys.argv[1]; n=int(re.match(r"([0-9]+)", v).group(1)); u=v[-1].lower(); print(n*{"k":1024,"m":1048576,"g":1073741824}[u])' "$SELVEDGE_MEM")
available=$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo)
need=$((mem_bytes / 1073741824))
if ((available < need + 4)); then
  printf 's49-bar: host MemAvailable %s GiB < limit %s GiB + 4 GiB reserve\n' "$available" "$need" >&2
  exit 75
fi
self_unit="${SELVEDGE_UNIT:-}"
if [[ -n "$self_unit" ]]; then
  while IFS= read -r unit; do
    [[ -z "$unit" ]] && continue
    [[ "$unit" == "${self_unit%.service}.service" ]] && continue
    printf 's49-bar: unit %s is still active; one job at a time\n' "$unit" >&2
    exit 75
  done < <(systemctl --user list-units --state=active --no-legend --plain 'selvedge-s49-*.service' 2>/dev/null | awk '{print $1}')
else
  if systemctl --user --quiet is-active selvedge-s49-*.service 2>/dev/null; then
    printf 's49-bar: an S49 unit is still active; one job at a time\n' >&2
    exit 75
  fi
fi
if pgrep -f 'oracle/frontier/train.py' >/dev/null 2>&1; then
  printf 's49-bar: a training process is already running; refusing to compete\n' >&2
  exit 75
fi

python3 - "$SELVEDGE_MEM" "results/stage2/runs/full-20m-math/receipt.json" "${chosen_dir}-smoke/receipt.json" "${chosen_dir}/receipt.json" <<'PY'
import glob, json, os, re, sys
limit_text, prior_bar, smoke, chosen_receipt = sys.argv[1:5]
n = int(re.match(r"([0-9]+)", limit_text).group(1))
unit = limit_text[-1].lower()
limit = n * {"k": 1024, "m": 1048576, "g": 1073741824}[unit]
candidates = [(prior_bar, "the previous 20M bar"),
              (smoke, "the chosen recipe's smoke"),
              (chosen_receipt, "the chosen recipe's 2M arm")]
for path in sorted(glob.glob("results/stage2/runs/stability-*/receipt.json")):
    candidates.append((path, "a completed stability arm"))
floors = []
for path, label in candidates:
    if not path or not os.path.exists(path):
        continue
    memory = json.load(open(path)).get("container_memory") or {}
    peak = memory.get("memory_peak") or memory.get("peak") or 0
    peak = int(peak)
    if peak <= 0:
        continue
    floors.append((label, peak, max(int(peak * 1.5), peak + 1073741824)))
if not floors:
    raise SystemExit("s49-bar: no measured peak is available; refusing to size the bar from nothing")
for label, peak, floor in floors:
    if limit < floor:
        raise SystemExit(
            f"s49-bar: SELVEDGE_MEM={limit} is below {label}'s floor {floor} "
            f"(max(1.5 x {peak}, {peak} + 1 GiB)); refusing to rerun the bar undersized")
    print(f"s49-bar: limit {limit} >= {label} floor {floor} (measured {peak})")
PY

python3 - "$run_dir" "$chosen_arm" "$chosen_lr" "$decision" "$SELVEDGE_MEM" "$vram" "$stream_manifest" "$stream_sha" <<'PY'
import json, os, re, sys
from datetime import datetime, timezone
(run_dir, arm, lr, decision, mem, vram, manifest, sha) = sys.argv[1:9]
n = int(re.match(r"([0-9]+)", mem).group(1))
unit = mem[-1].lower()
mem_bytes = n * {"k": 1024, "m": 1048576, "g": 1073741824}[unit]
prior = "results/stage2/runs/full-20m-math/receipt.json"
prior_peak = None
if os.path.exists(prior):
    memory = json.load(open(prior)).get("container_memory") or {}
    prior_peak = memory.get("memory_peak")
record = {
    "schema": "selvedge/s49-bar-launch/v1",
    "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "run_dir": run_dir,
    "recipe": {"source_arm": arm, "lr": float(lr), "decision_receipt": decision},
    "limits": {
        "container_memory_bytes": mem_bytes,
        "container_memory_swap_bytes": mem_bytes,
        "container_cpus": 8,
        "container_pids": 1024,
        "container_tmp": "2g",
        "vram_limit_gib": float(vram),
        "previous_bar_cgroup_peak_bytes": prior_peak,
        "sizing": "max(1.5 x measured peak, measured peak + 1 GiB) over the previous 20M bar and the chosen recipe's smoke",
    },
    "stream": {"manifest": manifest, "sha256": sha},
    "checkpoints": {"every_steps": 1000, "keep": 10,
                    "milestones": [3000, 5000, 9765]},
    "tripwire": {"factor": 1.5, "baseline_steps": 100, "window_steps": 50},
    "launcher": "scripts/s49_bar_20m.sh",
    "unit": os.environ.get("SELVEDGE_UNIT", "foreground"),
}
with open(os.path.join(run_dir, "launch.json"), "w", encoding="utf-8") as handle:
    json.dump(record, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY

SELVEDGE_CPUS=8 SELVEDGE_PIDS=1024 SELVEDGE_TMP=2g \
  exec scripts/court --gpu oracle -- "${cmd[@]}"
