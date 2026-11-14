# SELVEDGE

SELVEDGE is a research program measuring how much useful neural computation can
be obtained from a learned substrate under a hard **1,000,000,000 unique active
learned-parameter ceiling per output token**. Storage may live on NVMe or host
RAM; the active ceiling is independent of how large the substrate is.

The scientific question, stated so it can fail:

> Can a causal, interaction-aware, physically costed shard selector and
> execution scheduler improve the task-quality / active-parameter / wall-clock /
> physical-I/O Pareto frontier over the strongest existing sparse inference and
> offloading baselines?

This repository is in **bootstrap state**. It contains the reproducible
execution and evidence infrastructure; it does not yet contain the Rust model
crates, the conversion path, or any experimental result. Nothing here is a
demonstrated new architecture, and no performance claim has been measured.

## Rules this repository enforces mechanically

- **No host builds or tests.** Everything runs in an OOM-protected Docker court
  (`scripts/docker-court.sh`) with a hard memory limit, swap limited to the same
  value, a pids limit, a CPU limit and a bounded `/tmp`. Missing limits refuse to
  run; there is no host fallback.
- **No fabricated evidence.** Every court run writes a JSON receipt under
  `results/courts/` containing the command, image reference and image id, the
  applied limits (verified by inspecting the live container), network mode,
  exit code, duration and toolchain versions.
- **No hand-written published numbers.** Numbers in tracked documentation live
  between `BEGIN GENERATED` / `END GENERATED` sentinels produced by
  `scripts/perf_numbers.py` from `docs/perf-numbers.json`. See
  `docs/PERF-NUMBERS.md`.
- **Operator commit identity only.** `scripts/check-commit-identity.sh` fails
  closed unless the repository-local identity is the operator's and every scanned
  commit carries it.

## Commands that work today

```bash
# build (or rebuild) the pinned court images; writes docker/IMAGES.lock
scripts/build-courts.sh

# toolchain probes inside the courts
scripts/court rust -- cargo --version
scripts/court rust -- rustc --version
scripts/court python -- python3 --version

# integrity courts
scripts/court commit_identity
scripts/court perf_numbers

# hardware/toolchain probe -> results/hardware/probe-<stamp>.json
scripts/probe-hardware.sh

# generic court passthrough (no network unless --net is given)
scripts/docker-court.sh rust -- cargo test
scripts/docker-court.sh python -- python3 -c 'print(1)'
```

`scripts/court` is a thin alias over `scripts/docker-court.sh`:
`scripts/court rust test` runs `cargo test` in the Rust court,
`scripts/court python -m pytest` runs pytest in the Python court.

Limit overrides are accepted only as valid non-zero values, e.g.
`SELVEDGE_MEM=2g SELVEDGE_CPUS=4 scripts/court rust build`.

## Layout

```text
scripts/docker-court.sh        fail-closed Docker court (receipts, limits)
scripts/court                  short aliases over the court
scripts/build-courts.sh        build court images, update docker/IMAGES.lock
scripts/probe-hardware.sh      hardware/toolchain probe, through the courts
scripts/check-commit-identity.sh
scripts/perf_numbers.py        generated-numbers check/write
scripts/courts/                helpers that run *inside* a court
docker/                        court Dockerfiles, pinned requirements, image lock
docs/                          mandate copy, work log, generated numbers
results/                       court receipts, logs, hardware probes
```

The implementation mandate this program follows is tracked at
`docs/MANDATE.md`. The plan and current status are in `docs/WORK-LOG.md`.

## License

Apache-2.0. See `LICENSE`.
