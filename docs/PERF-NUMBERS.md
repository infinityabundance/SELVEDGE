# Generated numbers

Every number in this file is emitted by `scripts/perf_numbers.py` from
`docs/perf-numbers.json`. Each block names the receipt file and JSON pointer it
comes from, plus the receipt stamp and git revision that produced it. Editing a
value inside a `BEGIN GENERATED` region is a defect: run
`scripts/court perf_numbers --write` to regenerate and `scripts/court
perf_numbers` to check.

Nothing here is a performance claim. These are machine facts about the
workstation and toolchain this program is built against, published so that later
experiments can be compared against a named environment.

**Not every generated region is a number**, and those are declared rather than
merged into this file. A multi-line block (the status blocks in `README.md` and
`results/REPORT.md`, the per-arm state table) is owned by its own generator and
listed in the manifest under `external_regions` with an `owner`. This checker
still refuses an undeclared region and refuses a declared region the document does
not print; it never renders one. Current owners: `scripts/status_consistency.py`
(`readme-status`, `report-status`, `report-t2-arm-state`).

Host facts come from the hardware probe receipt under `results/hardware/`. That
receipt names, in its `evidence` field, the court receipts that produced each
section, and it was collected inside a court container with a hard memory limit,
a pids limit, a CPU limit and a bounded `/tmp`. Toolchain strings come from those
court receipts directly.

## CPU

**Logical cores visible to the host kernel**

<!-- BEGIN GENERATED: cpu-logical-cores -->
16
<!-- END GENERATED: cpu-logical-cores -->

**Physical cores**

<!-- BEGIN GENERATED: cpu-physical-cores -->
8
<!-- END GENERATED: cpu-physical-cores -->

## Memory and swap

**Total host RAM**

<!-- BEGIN GENERATED: ram-total-gib -->
125.47 GiB
<!-- END GENERATED: ram-total-gib -->

**Total swap (zram on this host)**

<!-- BEGIN GENERATED: swap-total-gib -->
125.47 GiB
<!-- END GENERATED: swap-total-gib -->

## Storage

**Capacity of the filesystem holding this repository**

<!-- BEGIN GENERATED: repo-fs-total-gib -->
1328.42 GiB
<!-- END GENERATED: repo-fs-total-gib -->

## Kernel

**Kernel release**

<!-- BEGIN GENERATED: kernel-release -->
7.3.3-1-cachyos
<!-- END GENERATED: kernel-release -->

## Toolchain

**Rust compiler reported by the Rust court**

<!-- BEGIN GENERATED: rustc-version -->
rustc 1.98.1 (48a229cea 2026-09-01)
<!-- END GENERATED: rustc-version -->

**Python interpreter reported by the Python court**

<!-- BEGIN GENERATED: python-version -->
Python 3.12.16
<!-- END GENERATED: python-version -->

## GPU

**Device name as seen by `nvidia-smi` inside a court container**

<!-- BEGIN GENERATED: gpu-name -->
NVIDIA GeForce RTX 4080 SUPER
<!-- END GENERATED: gpu-name -->

**Device VRAM as reported inside a court container**

<!-- BEGIN GENERATED: gpu-vram -->
16376 MiB
<!-- END GENERATED: gpu-vram -->

## Auditing

```bash
scripts/court perf_numbers              # verify every region against its receipt
scripts/court perf_numbers --write      # regenerate regions atomically
scripts/court perf_numbers --list       # print id -> receipt -> pointer -> value
```

A block fails closed when its receipt is missing or untracked, when the declared
stamp or revision no longer matches the receipt, when the receipt's revision is
not a commit in this repository, when the region text is stale, when a numeric
source renders no digits, or when a generated region in any tracked document has
no manifest block.


## Runtime numbers (M4 part 1)

The CPU latency, throughput, byte-traffic and parity numbers of the single-device
runtime are generated into the document that carries their method:

- `docs/RUNTIME.md` §2 (backend decision weights),
- `docs/RUNTIME.md` §1 (real-model parity against the independent NumPy forward),
- `docs/RUNTIME.md` §10 (cold and warm CPU latency, bytes per token, memory peak,
  active-unique and bundle size).

`docs/perf-numbers.json` names the receipt and JSON pointer for each of them, with
`doc` set to `docs/RUNTIME.md`, so `--check` validates them exactly as it
validates the numbers below.
