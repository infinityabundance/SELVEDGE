#!/usr/bin/env bash
# Launch (or resume) the 20M full-distillation run on the S43 math-weighted
# stream. The container memory limit must come from the arm's own smoke receipt:
# this script refuses to run without it and never falls back to the court default.
#
#   SELVEDGE_MEM=9g scripts/s43_fulldistill_math.sh
#
# The stream, objective, optimiser, schedule and checkpoint policy are the ones
# registered in docs/ws19-full-distillation-preregistration-v1.json as amended by
# docs/ws19-full-distillation-amendment-math-stream-preregistration-v1.json.
# `--resume latest` is added automatically when a checkpoint exists; train.py
# verifies the config and stream hashes before loading.
set -Eeuo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
repo_root=$(cd -- "$script_dir/.." && pwd -P)
cd "$repo_root"

: "${SELVEDGE_MEM:?set SELVEDGE_MEM from the smoke cgroup peak (1.5x, at least +1 GiB)}"
export SELVEDGE_MEM

stream_manifest="results/stage2/stage2-math-stream-20261117T071120Z.json"
stream_sha="9c5fb53b20a8f9165566b8c4e237103eb2e4c658f944bb221b3909d1dc8b6d56"
run_dir="results/stage2/runs/full-20m-math"

if [[ ! -f "$stream_manifest" ]]; then
  printf 's43: missing stream manifest %s\n' "$stream_manifest" >&2
  exit 78
fi
# Fail closed on a drifted registration before spending a GPU-hour. The training
# process re-checks the same stream hash inside the court before the first step.
recorded=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["sha256"])' "$stream_manifest")
if [[ "$recorded" != "$stream_sha" ]]; then
  printf 's43: stream %s records %s, expected %s\n' "$stream_manifest" "$recorded" "$stream_sha" >&2
  exit 78
fi

mkdir -p "$run_dir"
resume=()
if [[ -f "$run_dir/latest.pt" ]]; then
  resume=(--resume latest)
  printf 's43: resuming from %s/latest.pt\n' "$run_dir"
else
  printf 's43: fresh run in %s\n' "$run_dir"
fi

SELVEDGE_CPUS=8 SELVEDGE_PIDS=1024 SELVEDGE_TMP=2g \
  exec scripts/court --gpu oracle -- python3 oracle/frontier/train.py \
    --arm full \
    --stream-manifest "$stream_manifest" \
    --tokens 20000000 \
    --seq-len 512 \
    --micro-batch 1 \
    --grad-accum 4 \
    --lr 0.0001 \
    --temperature 2.0 \
    --kl-weight 0.5 \
    --optimizer adam8bit \
    --optim-block 256 \
    --optim-numerics stochastic \
    --schedule cosine \
    --min-lr-ratio 0.1 \
    --warmup 200 \
    --grad-clip 1.0 \
    --seed 0 \
    --ckpt-every 1000 \
    --keep-checkpoints 2 \
    --log-every 100 \
    --run-dir "$run_dir" \
    --vram-limit-gib 12.5 \
    "${resume[@]}"
