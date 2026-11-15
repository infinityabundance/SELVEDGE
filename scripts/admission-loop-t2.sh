#!/bin/sh
# Build the T2 admission records inside the Rust court.
#
# Called by scripts/train-admission-t2.sh. Each T2 arm config is admitted
# through the Rust ledger; an over-budget arm exits non-zero and stops this
# script, so no admitted path exists for it. The ledger is the source of
# truth; a Python mirror cross-check runs after this in the python court.
set -eu

cargo build --release -p selvedge-cli
mkdir -p results/train/admission/t2
for arm in D M X H 'H*'; do
  printf 'admission: %s\n' "$arm"
  ./target/release/selvedge admit \
    --config "results/train/admission-configs/t2/${arm}.json" \
    --out "results/train/admission/t2/${arm}.json"
done
printf 'admission: all T2 records written\n'
