#!/usr/bin/env bash
# P8.2 — admit the bounded-gate composition through the Rust ledger.
#
#   scripts/p8-composition-admission.sh
#
# The composed arm is the frozen core plus T2 `D`'s dense adapter plus T2 `M`'s
# hashed memory plus one `hidden -> 2` gate per layer. The config is emitted by
# the Python court (no model access); the admission record — including the union
# of the two arms' execution slices by identity — is written by the Rust court,
# which is the only place the parameter ceiling is decided. The Python mirror
# prints the same arithmetic so a disagreement is visible immediately.
#
# As with the T2 arms, the *accessible* set of the composed arm is above the
# ceiling while its *active* union is not: the ceiling is a per-token execution
# budget, not a storage budget.
set -Eeuo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
repo_root=$(cd -- "$script_dir/.." && pwd -P)
cd "$repo_root"

scripts/court oracle -- python3 oracle/train/composition.py \
    --emit-config results/train/admission-configs/p8/composition.json

scripts/court rust -- sh -c '
cargo build --release -q -p selvedge-cli
mkdir -p results/train/admission/p8
./target/release/selvedge admit \
    --config results/train/admission-configs/p8/composition.json \
    --out results/train/admission/p8/composition.json
'

stamp=$(date -u +%Y%m%dT%H%M%SZ)
out="results/p8/composition-admission-${stamp}.json"
scripts/court oracle -- python3 oracle/train/composition.py --wrap-admission \
    results/train/admission-configs/p8/composition.json \
    results/train/admission/p8/composition.json \
    "$out"
printf 'admission receipt: %s\n' "$out"
