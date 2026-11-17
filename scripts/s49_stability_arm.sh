#!/usr/bin/env bash
# S49: run ONE registered stability-sweep arm, or its smoke, on the S43
# math-weighted stream.
#
# Everything about the arm — lr, tokens, budget, optimiser, schedule, tripwire —
# is read from the registration, not from this script's defaults:
#
#   docs/ws19-full-distillation-amendment-stability-s49-preregistration-v1.json
#
#   SELVEDGE_MEM=12g scripts/s49_stability_arm.sh --smoke --arm stability-lr3e-5-2m
#   SELVEDGE_MEM=10g scripts/s49_stability_arm.sh --arm stability-lr3e-5-2m
#   scripts/s49_stability_arm.sh --arm stability-lr3e-5-2m --dry-run
#
# SELVEDGE_MEM is mandatory and the 2M launch must be at least the larger of
# 1.5x the arm's own smoke cgroup peak and that peak + 1 GiB; the smoke must run
# at a bound it does not reach. The script refuses to launch while another S49
# unit is active or when host memory is short, because the sweep is sequential.
set -Eeuo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
repo_root=$(cd -- "$script_dir/.." && pwd -P)
cd "$repo_root"

registration="docs/ws19-full-distillation-amendment-stability-s49-preregistration-v1.json"
arm=""
smoke=0
dry_run=0
while (($#)); do
  case "$1" in
    --arm) arm="${2-}"; shift 2 ;;
    --arm=*) arm="${1#*=}"; shift ;;
    --smoke) smoke=1; shift ;;
    --dry-run) dry_run=1; shift ;;
    -h | --help) sed -n '2,22p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) printf 's49-arm: unknown argument %s\n' "$1" >&2; exit 64 ;;
  esac
done

[[ -f "$registration" ]] || { printf 's49-arm: missing registration %s\n' "$registration" >&2; exit 78; }
[[ -n "$arm" ]] || { printf 's49-arm: --arm is required\n' >&2; exit 64; }

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

if ! python3 - "$registration" "$arm" <<'PY'
import json, sys
known = {entry["name"] for entry in json.load(open(sys.argv[1], encoding="utf-8"))
         ["stability_sweep"]["arms"]}
raise SystemExit(0 if sys.argv[2] in known else 1)
PY
then
  printf 's49-arm: --arm must be one of the registered stability arms\n' >&2
  python3 - "$registration" <<'PY'
import json, sys
for entry in json.load(open(sys.argv[1], encoding="utf-8"))["stability_sweep"]["arms"]:
    print(f"  {entry['name']}  lr={entry['lr']}")
PY
  exit 64
fi

lr=$(python3 - "$registration" "$arm" <<'PY'
import json, sys
registration, name = sys.argv[1], sys.argv[2]
for entry in json.load(open(registration, encoding="utf-8"))["stability_sweep"]["arms"]:
    if entry["name"] == name:
        print(entry["lr"]); raise SystemExit(0)
raise SystemExit(f"unknown arm {name}")
PY
)
run_dir=$(python3 - "$registration" "$arm" <<'PY'
import json, sys
registration, name = sys.argv[1], sys.argv[2]
for entry in json.load(open(registration, encoding="utf-8"))["stability_sweep"]["arms"]:
    if entry["name"] == name:
        print(entry["run_dir"]); raise SystemExit(0)
raise SystemExit(f"unknown arm {name}")
PY
)
stream_manifest=$(read_field stability_sweep stream manifest)
stream_sha=$(read_field stability_sweep stream sha256)
common_tokens=$(read_field stability_sweep common tokens)
common_steps=$(read_field stability_sweep common steps)
temperature=$(read_field stability_sweep common teacher_temperature)
kl_weight=$(read_field stability_sweep common kl_weight)
optim_block=$(read_field stability_sweep common optim_block_elements)
optim_numerics=$(read_field stability_sweep common optimizer_numerics)
min_lr_ratio=$(read_field stability_sweep common min_lr_ratio)
warmup=$(read_field stability_sweep common warmup_requested)
grad_clip=$(read_field stability_sweep common grad_clip)
seed=$(read_field stability_sweep common seed)
vram=$(read_field stability_sweep common vram_limit_gib)
ckpt_every=$(read_field stability_sweep common checkpoints every_steps)
keep=$(read_field stability_sweep common checkpoints keep)
log_every=$(read_field stability_sweep common log_every)
tripwire_factor=$(read_field stability_sweep early_loss_tripwire factor)

[[ -f "$stream_manifest" ]] || { printf 's49-arm: missing stream manifest %s\n' "$stream_manifest" >&2; exit 78; }
recorded=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["sha256"])' "$stream_manifest")
[[ "$recorded" == "$stream_sha" ]] || {
  printf 's49-arm: stream %s records %s, registration says %s\n' "$stream_manifest" "$recorded" "$stream_sha" >&2
  exit 78
}

if ((smoke)); then
  dir="${run_dir}-smoke"
  tokens=40960
  steps=20
  ckpt_every_smoke=20
  keep_smoke=1
  log_every_smoke=1
  warmup_smoke=5
  tripwire=()
  printf 's49-arm: smoke %s at 40,960 tokens (20 steps), tripwire off\n' "$arm"
else
  dir="$run_dir"
  tokens="$common_tokens"
  steps="$common_steps"
  ckpt_every_smoke="$ckpt_every"
  keep_smoke="$keep"
  log_every_smoke="$log_every"
  warmup_smoke="$warmup"
  tripwire=(--abort-ce-factor "$tripwire_factor" --abort-ce-baseline-steps 100
    --abort-ce-window 50)
fi
mkdir -p "$dir"
if [[ -f "$dir/abort.json" ]]; then
  printf 's49-arm: %s already recorded an abort; a recorded abort is final\n' "$dir" >&2
  exit 78
fi

resume=()
if [[ -f "$dir/latest.pt" ]]; then
  resume=(--resume latest)
  printf 's49-arm: resuming from %s/latest.pt\n' "$dir"
else
  printf 's49-arm: fresh run in %s\n' "$dir"
fi

cmd=(python3 oracle/frontier/train.py
  --arm full
  --stream-manifest "$stream_manifest"
  --tokens "$tokens"
  --seq-len 512
  --micro-batch 1
  --grad-accum 4
  --lr "$lr"
  --temperature "$temperature"
  --kl-weight "$kl_weight"
  --optimizer adam8bit
  --optim-block "$optim_block"
  --optim-numerics "$optim_numerics"
  --schedule cosine
  --min-lr-ratio "$min_lr_ratio"
  --warmup "$warmup_smoke"
  --grad-clip "$grad_clip"
  --seed "$seed"
  --ckpt-every "$ckpt_every_smoke"
  --keep-checkpoints "$keep_smoke"
  --log-every "$log_every_smoke"
  --run-dir "$dir"
  --vram-limit-gib "$vram"
  "${tripwire[@]}"
  "${resume[@]}")

if ((dry_run)); then
  printf 's49-arm: arm=%s lr=%s tokens=%s steps=%s smoke=%s\n' "$arm" "$lr" "$tokens" "$steps" "$smoke"
  printf 's49-arm: stream=%s sha256=%s\n' "$stream_manifest" "$stream_sha"
  printf 's49-arm: limits: SELVEDGE_MEM=%s SELVEDGE_CPUS=8 SELVEDGE_PIDS=1024 SELVEDGE_TMP=2g VRAM=%s GiB\n' \
    "${SELVEDGE_MEM:-<unset: required at launch>}" "$vram"
  printf 's49-arm: command:\n  '
  printf '%q ' scripts/court --gpu oracle -- "${cmd[@]}"
  printf '\n'
  exit 0
fi

[[ -n "${SELVEDGE_MEM:-}" ]] || {
  printf 's49-arm: set SELVEDGE_MEM from the arm smoke peak (1.5x, at least +1 GiB)\n' >&2
  exit 78
}
export SELVEDGE_MEM
mem_bytes=$(python3 -c 'import re,sys; v=sys.argv[1]; n=int(re.match(r"([0-9]+)", v).group(1)); u=v[-1].lower(); print(n*{"k":1024,"m":1048576,"g":1073741824}[u])' "$SELVEDGE_MEM")
available=$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo)
need=$((mem_bytes / 1073741824))
if ((available < need + 4)); then
  printf 's49-arm: host MemAvailable %s GiB < limit %s GiB + 4 GiB reserve\n' "$available" "$need" >&2
  exit 75
fi
if systemctl --user --quiet is-active selvedge-s49-*.service 2>/dev/null; then
  printf 's49-arm: an S49 unit is still active; the sweep is sequential\n' >&2
  exit 75
fi

if ((smoke == 0)); then
  python3 - "$SELVEDGE_MEM" "${run_dir}-smoke/receipt.json" <<'PY'
import glob, json, re, sys
limit_text, smoke_receipt = sys.argv[1], sys.argv[2]
n = int(re.match(r"([0-9]+)", limit_text).group(1))
unit = limit_text[-1].lower()
limit = n * {"k": 1024, "m": 1048576, "g": 1073741824}[unit]
receipts = []
if smoke_receipt and __import__("os").path.exists(smoke_receipt):
    receipts.append((smoke_receipt, "this arm's smoke"))
for path in sorted(glob.glob("results/stage2/runs/stability-*/receipt.json")):
    if path != smoke_receipt:
        receipts.append((path, "a completed stability arm"))
for path, label in receipts:
    memory = json.load(open(path)).get("container_memory") or {}
    peak = memory.get("memory_peak") or memory.get("peak") or 0
    peak = int(peak)
    if peak <= 0:
        raise SystemExit(f"s49-arm: {path} has no cgroup memory peak; refusing to size from nothing")
    floor = max(int(peak * 1.5), peak + 1073741824)
    if limit < floor:
        raise SystemExit(
            f"s49-arm: SELVEDGE_MEM={limit} is below the {label} floor {floor} "
            f"(max(1.5 x {peak}, {peak} + 1 GiB)) at {path}; refusing to inherit "
            "an undersized limit")
    print(f"s49-arm: limit {limit} >= {label} floor {floor} (measured {peak})")
if not receipts:
    raise SystemExit("s49-arm: no smoke receipt for this arm; run --smoke first")
PY
fi

python3 - "$dir" "$arm" "$lr" "$tokens" "$steps" "$SELVEDGE_MEM" "$vram" "$stream_manifest" "$stream_sha" <<'PY'
import json, os, sys
from datetime import datetime, timezone
run_dir, arm, lr, tokens, steps, mem, vram, manifest, sha = sys.argv[1:10]
n = int(__import__("re").match(r"([0-9]+)", mem).group(1))
unit = mem[-1].lower()
mem_bytes = n * {"k": 1024, "m": 1048576, "g": 1073741824}[unit]
record = {
    "schema": "selvedge/s49-stability-launch/v1",
    "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "arm": arm,
    "lr": float(lr),
    "tokens": int(tokens),
    "steps": int(steps),
    "limits": {
        "container_memory_bytes": mem_bytes,
        "container_memory_swap_bytes": mem_bytes,
        "container_cpus": 8,
        "container_pids": 1024,
        "container_tmp": "2g",
        "vram_limit_gib": float(vram),
        "sizing": "SELVEDGE_MEM checked against this arm's own smoke peak by scripts/s49_stability_arm.sh",
    },
    "stream": {"manifest": manifest, "sha256": sha},
    "launcher": "scripts/s49_stability_arm.sh",
    "unit": os.environ.get("SELVEDGE_UNIT", "foreground"),
}
with open(os.path.join(run_dir, "launch.json"), "w", encoding="utf-8") as handle:
    json.dump(record, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY

SELVEDGE_CPUS=8 SELVEDGE_PIDS=1024 SELVEDGE_TMP=2g \
  exec scripts/court --gpu oracle -- "${cmd[@]}"
