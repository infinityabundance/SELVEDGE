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
scripts/court perf_numbers --write
scripts/court perf_numbers --list

# negative self-tests (these must exit zero and print PASS lines)
scripts/docker-court.sh python -- bash scripts/courts/perf_numbers_negative_cases.sh
scripts/docker-court.sh python -- bash scripts/courts/commit_identity_negative_cases.sh

# hardware/toolchain probe -> results/hardware/probe-<stamp>.json
scripts/probe-hardware.sh

# generic court passthrough (no network unless --net is given)
scripts/docker-court.sh rust -- cargo --version
scripts/docker-court.sh python -- python3 -c 'print(1)'
scripts/docker-court.sh -- python3 -c 'print("generic passthrough")'
```

`scripts/court` is a thin alias over `scripts/docker-court.sh`:
`scripts/court rust test` runs `cargo test` in the Rust court,
`scripts/court python -m pytest` runs pytest in the Python court.

Limit overrides are accepted only as valid non-zero values, e.g.
`SELVEDGE_MEM=2g SELVEDGE_CPUS=4 scripts/court rust build`.

## Court interface for later work

- **Rust builds and tests:** `scripts/court rust test`, `... build`, `... clippy`,
  or `scripts/docker-court.sh rust -- cargo <args>`. The container runs as the
  host user with `CARGO_HOME=/cargo` (persistent registry/git cache) and
  `CARGO_TARGET_DIR=/work/target` (persistent target dir). Network is disabled,
  and `CARGO_NET_OFFLINE=true` is set, so dependency resolution fails loudly
  instead of hanging. Fetch dependencies with an explicit networked run:
  `scripts/court --net rust -- cargo fetch` and commit `Cargo.lock`.
- **Python oracle jobs:** `scripts/court python scripts/oracle/<job>.py` (or
  `-m pytest`). The container has pinned `numpy`/`pytest` from
  `docker/requirements-court.txt`, `PIP_CACHE_DIR=/pycache/pip`, and
  `PYTHONPYCACHEPREFIX=/pycache/pycache`; create a venv under `/pycache` if a job
  needs more packages, and add pins to the requirements file for anything that
  must be reproducible.
- **I/O courts:** `scripts/court io -- <argv>` uses the Python court image
  (`fio`, `dd`, `df`) with the same limits.
- **GPU:** add `--gpu`. On this host that resolves to explicit device
  passthrough because `--gpus all` fails without a CDI specification; the
  receipt records which mode was used and why. A GPU court fails closed rather
  than falling back to CPU.
- **Limits:** `SELVEDGE_MEM` (default `8g`), `SELVEDGE_CPUS` (`8`),
  `SELVEDGE_PIDS` (`1024`), `SELVEDGE_TMP` (`2g`), `SELVEDGE_NOFILE` (`65536`).
  Empty, zero or malformed values refuse to run.

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
