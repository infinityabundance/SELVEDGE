#!/usr/bin/env bash
# S41 full-backbone distillation, 20M tokens, resumable.
#
# Registered in docs/ws19-full-distillation-preregistration-v1.json. The tied
# token embedding is frozen by oracle/frontier/full.py; only the remaining
# 498,113,344 text-tower elements are trainable, with block-wise 8-bit Adam.
#
# Resumption: if latest.pt exists the same command is re-issued with
# `--resume latest`; train.py verifies the config and stream hashes before
# loading. A completed run is not restarted.
#
# `--optim-numerics stochastic` is the S42 registered default (see
# docs/ws19-full-distillation-amendment-preregistration-v1.json).
set -Eeuo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
cd "$repo_root"

run_dir="results/stage2/runs/full-20m"
if [[ -f "$run_dir/ckpt-009765.pt" ]]; then
  printf 's41: %s already has ckpt-009765.pt; refusing to restart a completed run\n' "$run_dir"
  exit 0
fi

resume=()
if [[ -f "$run_dir/latest.pt" ]]; then
  resume=(--resume latest)
fi

exec scripts/court --gpu oracle -- python3 oracle/frontier/train.py \
  --arm full \
  --stream-manifest results/stage2/stage2-stream-20261117T004849Z.json \
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
