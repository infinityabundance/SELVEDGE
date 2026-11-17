#!/usr/bin/env bash
# S41 full-distillation 20-step smoke. Registered in
# docs/ws19-full-distillation-preregistration-v1.json (`smoke_command`).
#
# The measured peak from this run replaces the modelled activation term in
# results/stage2/s40-full-arithmetic-20261117T024151Z.json before the 20M-token
# launch. The host memory limit is set by the systemd unit, not here.
#
# `--optim-numerics stochastic` is the S42 default: bf16 weight updates below
# half an ulp change the weight with probability update/ulp instead of never.
# The run measures the bitwise-unchanged fraction at the first step, the first
# full-LR step and the final step and writes them into its receipt.
set -Eeuo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
cd "$repo_root"

exec scripts/court --gpu oracle -- python3 oracle/frontier/train.py \
  --arm full \
  --stream-manifest results/stage2/stage2-stream-20261117T004849Z.json \
  --tokens 40960 \
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
  --warmup 5 \
  --grad-clip 1.0 \
  --seed 0 \
  --ckpt-every 20 \
  --keep-checkpoints 2 \
  --log-every 1 \
  --run-dir results/stage2/runs/full-smoke \
  --vram-limit-gib 12.5
