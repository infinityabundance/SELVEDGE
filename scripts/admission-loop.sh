#!/bin/sh
# Build the T1 admission records inside the Rust court.
#
# Called by scripts/train-admission.sh. Each arm config is admitted through the
# Rust ledger; an over-budget arm exits non-zero and stops this script, so no
# admitted path exists for it.
set -eu

cargo build --release -p selvedge-cli
mkdir -p results/train/admission
for arm in D A A0 B C C2; do
  printf 'admission: %s\n' "$arm"
  ./target/release/selvedge admit \
    --config "results/train/admission-configs/${arm}.json" \
    --out "results/train/admission/${arm}.json"
done
printf 'admission: all records written\n'
