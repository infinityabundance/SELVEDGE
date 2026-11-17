#!/usr/bin/env bash
# S49: rerun the 20M full-backbone distillation bar under the registered
# stability escalation.
#
#   SELVEDGE_MEM=20g scripts/s49_bar_20m.sh --arm stability-lr3e-5-2m
#   scripts/s49_bar_20m.sh --arm stability-lr1e-5-2m --dry-run
#
# The arm must be the next entry of the escalation order registered in
# docs/ws19-full-distillation-amendment-stability-s49b-preregistration-v1.json:
# lr 3e-5 first, then 1e-5, then 5e-6, each attempt with the tripwire, and the
# first attempt that completes 19,998,720 tokens is the bar. An attempt that
# trips the tripwire is a recorded result and is never retried; a completed
# attempt is never retried at another rate.
#
# SELVEDGE_MEM is mandatory. The floor is derived from measured peaks of the
# same workload (the previous 20M bar, the chosen rate's 20-step smoke, its 2M
# arm, and every completed stability arm) at max(1.5 x peak, peak + 1 GiB).
set -Eeuo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
repo_root=$(cd -- "$script_dir/.." && pwd -P)
cd "$repo_root"

registration="docs/ws19-full-distillation-amendment-stability-s49-preregistration-v1.json"
amendment="docs/ws19-full-distillation-amendment-stability-s49b-preregistration-v1.json"
arm=""
dry_run=0
while (($#)); do
  case "$1" in
    --arm) arm="${2-}"; shift 2 ;;
    --arm=*) arm="${1#*=}"; shift ;;
    --amendment) amendment="${2-}"; shift 2 ;;
    --dry-run) dry_run=1; shift ;;
    -h | --help) sed -n '2,18p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) printf 's49-bar: unknown argument %s\n' "$1" >&2; exit 64 ;;
  esac
done

[[ -f "$registration" ]] || { printf 's49-bar: missing registration %s\n' "$registration" >&2; exit 78; }
[[ -f "$amendment" ]] || { printf 's49-bar: missing amendment %s\n' "$amendment" >&2; exit 78; }
[[ -n "$arm" ]] || { printf 's49-bar: --arm is required (the next escalation entry)\n' >&2; exit 64; }

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

selection=$(python3 - "$amendment" "$registration" "$arm" <<'PY'
import hashlib, json, os, sys
amendment_path, registration_path, arm = sys.argv[1:4]
amendment = json.load(open(amendment_path, encoding="utf-8"))
registration = json.load(open(registration_path, encoding="utf-8"))
order = amendment["selection_amendment"]["escalation_order"]
if arm not in order:
    raise SystemExit(f"s49-bar: {arm} is not in the registered escalation order {order}")
arms = {entry["name"]: entry for entry in registration["stability_sweep"]["arms"]}
if arm not in arms:
    raise SystemExit(f"s49-bar: {arm} is not a registered stability arm")
index = order.index(arm)
two_m = arms[arm]["run_dir"]
if os.path.exists(os.path.join(two_m, "abort.json")):
    raise SystemExit(f"s49-bar: {arm} tripped the 2M tripwire; a recorded abort is final")
receipt_path = os.path.join(two_m, "receipt.json")
if not os.path.exists(receipt_path):
    raise SystemExit(f"s49-bar: {arm} has no 2M training receipt")
tripwire = json.load(open(receipt_path, encoding="utf-8")).get("tripwire") or {}
if tripwire.get("tripped"):
    raise SystemExit(f"s49-bar: {arm} records a tripped 2M tripwire")

def attempt_dir(position):
    if position == 0:
        return registration["bar_rerun"]["run_dir"]
    suffix = order[position].replace("stability-", "").replace("-2m", "")
    return registration["bar_rerun"]["run_dir"] + "-" + suffix

for earlier in range(index):
    directory = attempt_dir(earlier)
    if not (os.path.exists(os.path.join(directory, "abort.json"))
            or os.path.exists(os.path.join(directory, "receipt.json"))):
        raise SystemExit(
            f"s49-bar: earlier escalation entry {order[earlier]} has no recorded "
            f"result in {directory}; the registered order does not skip rates")
directory = attempt_dir(index)
if os.path.exists(os.path.join(directory, "receipt.json")):
    raise SystemExit(f"s49-bar: {directory} already holds a completed bar attempt")
if os.path.exists(os.path.join(directory, "abort.json")):
    raise SystemExit(f"s49-bar: {directory} already holds an aborted attempt")
digest = hashlib.sha256(open(amendment_path, "rb").read()).hexdigest()
print(f"{arm}\t{arms[arm]['lr']}\t{directory}\t{index}\t{digest}\t{amendment_path}")
PY
) || exit 78
IFS=$'\t' read -r chosen_arm chosen_lr run_dir attempt_index amendment_sha amendment_path <<<"$selection"

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
chosen_dir=$(python3 - "$registration" "$chosen_arm" <<'PY'
import json, sys
registration, name = sys.argv[1:3]
for entry in json.load(open(registration, encoding="utf-8"))["stability_sweep"]["arms"]:
    if entry["name"] == name:
        print(entry["run_dir"]); raise SystemExit(0)
raise SystemExit(f"unknown arm {name}")
PY
)

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
  printf 's49-bar: authority=%s sha256=%s order_index=%s\n' "$amendment_path" "$amendment_sha" "$attempt_index"
  printf 's49-bar: arm=%s lr=%s (2M receipt in %s)\n' "$chosen_arm" "$chosen_lr" "$chosen_dir"
  printf 's49-bar: run_dir=%s tokens=%s steps=%s\n' "$run_dir" "$tokens" "$steps"
  printf 's49-bar: limits: SELVEDGE_MEM=%s SELVEDGE_CPUS=8 SELVEDGE_PIDS=1024 SELVEDGE_TMP=2g VRAM=%s GiB\n' \
    "${SELVEDGE_MEM:-<unset: required at launch>}" "$vram"
  printf 's49-bar: command:\n  '
  printf '%q ' scripts/court --gpu oracle -- "${cmd[@]}"
  printf '\n'
  exit 0
fi

[[ -n "${SELVEDGE_MEM:-}" ]] || {
  printf 's49-bar: set SELVEDGE_MEM from the measured long-run peak (amendment: 20g)\n' >&2
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

python3 - "$run_dir" "$chosen_arm" "$chosen_lr" "$attempt_index" "$amendment_path" "$amendment_sha" "$SELVEDGE_MEM" "$vram" "$stream_manifest" "$stream_sha" <<'PY'
import json, os, re, sys
from datetime import datetime, timezone
(run_dir, arm, lr, index, amendment, amendment_sha, mem, vram,
 manifest, sha) = sys.argv[1:11]
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
    "recipe": {"source_arm": arm, "lr": float(lr),
               "authority": amendment, "authority_sha256": amendment_sha,
               "escalation_index": int(index)},
    "limits": {
        "container_memory_bytes": mem_bytes,
        "container_memory_swap_bytes": mem_bytes,
        "container_cpus": 8,
        "container_pids": 1024,
        "container_tmp": "2g",
        "vram_limit_gib": float(vram),
        "previous_bar_cgroup_peak_bytes": prior_peak,
        "sizing": "max(1.5 x measured peak, measured peak + 1 GiB) over the previous 20M bar, the chosen recipe's smoke and 2M arm, and every completed stability arm",
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
