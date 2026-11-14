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

This repository contains the reproducible execution and evidence
infrastructure, the Rust data contracts (`selvedge-core`), the versioned shard
container (`selvedge-format`), exact SwiGLU conversion (`selvedge-convert`), the
`selvedge` CLI, deterministic fixtures, a pinned Hugging Face teacher oracle,
measured real-checkpoint accounting and reconstruction, and a pre-registered,
receipted measurement of the quality-versus-active-parameter frontier under
exact FFN group omission. Nothing here is a demonstrated new architecture, and
no capability, speed or latency claim has been measured.

## Current result (M1 part 2)

The registered experiment (`docs/EXPERIMENT-CONTRACT.md`) asked whether a
static selection of whole FFN channel groups can keep Qwen3-1.7B within the
`<=1e9` unique-active-parameter ceiling without changing its next-token
distribution. **It cannot, in the measured family.** Every configuration at or
below the ceiling is far outside the pre-registered acceptable tier; the
quality/parameter curve is a cliff near the fully dense model, above the
ceiling. The 0.6B control is already dense-compliant and needs no sparsity.

The result is bounded and honest: it covers static, token-independent masks at
the sixteen-group granularity, with no retraining and no learned routing. It
does not bound dynamic or trained selectors. The full report, frontier table,
plot and receipts are in `results/M1-FRONTIER.md`; limitations are in
`docs/LIMITATIONS.md`.

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

# Rust workspace checks (offline, in the Rust court)
scripts/court rust fmt --all --check
scripts/court rust clippy --workspace --all-targets -- -D warnings
scripts/court rust test --workspace

# regenerate the tracked synthetic fixture
scripts/court rust -- cargo run -p selvedge-fixtures -- fixtures

# CLI (built by the previous test/build run; use `cargo run -p selvedge-cli --`
# to build on demand inside the court)
scripts/court rust -- target/debug/selvedge doctor --json
scripts/court rust -- target/debug/selvedge inspect fixtures/tiny-synthetic.slv
scripts/court rust -- target/debug/selvedge format validate fixtures/tiny-synthetic.slv
scripts/court rust -- target/debug/selvedge court c
scripts/court rust -- target/debug/selvedge court d --workdir target/court-d

# real-model path: build the oracle image, fetch once with network, then stay offline
scripts/build-courts.sh oracle
scripts/fetch-models.sh
scripts/fetch-models.sh --verify-only
scripts/fetch-datasets.sh
scripts/fetch-datasets.sh --verify-only
scripts/court oracle -- python3 oracle/dataset_prep.py --preregistration-commit \
  dad6619c8514cb4a52b3420c007a47fc43e1b47e
scripts/court --gpu oracle -- python3 oracle/probe_oracle.py --json --require-gpu
scripts/court rust -- cargo run --release -p selvedge-cli -- inspect-model \
  /hf-cache/models/Qwen/Qwen3-0.6B/c1899de289a04d12100db370d81485cdf75e47ca --json

# M1 frontier: statistics, exactness gates, sweeps, bootstrap and plot
scripts/court --gpu oracle -- python3 oracle/frontier_sweep.py stats \
  --model-id Qwen/Qwen3-1.7B \
  --model-revision 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e \
  --accounting results/accounting/20261114T201533Z-qwen3-1.7b.json \
  --preregistration-commit dad6619c8514cb4a52b3420c007a47fc43e1b47e \
  --out results/artifacts/frontier/plans-1.7b.json
scripts/court --gpu oracle -- python3 oracle/frontier_sweep.py sweep \
  --model-id Qwen/Qwen3-1.7B \
  --model-revision 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e \
  --accounting results/accounting/20261114T201533Z-qwen3-1.7b.json \
  --preregistration-commit dad6619c8514cb4a52b3420c007a47fc43e1b47e \
  --plans results/artifacts/frontier/plans-1.7b.json --arms uniform
scripts/court oracle -- python3 oracle/frontier_sweep.py analyze \
  --preregistration-commit dad6619c8514cb4a52b3420c007a47fc43e1b47e

# Court B unit tests for the masked-MLP instrument (CPU, synthetic)
scripts/court oracle -- python3 -m pytest oracle/tests -q
scripts/court rust -- cargo run --release -p selvedge-cli -- convert \
  --model /hf-cache/models/Qwen/Qwen3-0.6B/c1899de289a04d12100db370d81485cdf75e47ca \
  --out results/artifacts/qwen3-0.6b-ffn-g16 --groups 16 --complete --alignment 4096
scripts/court rust -- cargo run --release -p selvedge-cli -- convert verify \
  --bundle results/artifacts/qwen3-0.6b-ffn-g16/<bundle>.slv \
  --model /hf-cache/models/Qwen/Qwen3-0.6B/c1899de289a04d12100db370d81485cdf75e47ca --json

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

- **Rust builds and tests:** `scripts/court rust test`, `... build`, `... fmt --all
  --check`, `... clippy --workspace --all-targets -- -D warnings`, or
  `scripts/docker-court.sh rust -- cargo <args>`. The container runs as the
  host user with `CARGO_HOME=/cargo` (persistent registry/git cache) and
  `CARGO_TARGET_DIR=/work/target` (persistent target dir). Network is disabled,
  and `CARGO_NET_OFFLINE=true` is set, so dependency resolution fails loudly
  instead of hanging. Fetch dependencies with an explicit networked run:
  `scripts/court --net rust -- cargo fetch` and commit `Cargo.lock`.
- **Python oracle jobs:** `scripts/court oracle -- python3 oracle/<job>.py`.
  The oracle image pins torch, transformers, tokenizers, safetensors and numpy;
  it mounts `~/.cache/selvedge/hf` read-write at `/hf-cache` and is offline
  unless `--net` is passed. `trust_remote_code` is never enabled. The plain
  `python` court remains the lightweight court for `perf_numbers` and helper
  scripts.
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
Cargo.toml                     Rust workspace (resolver 3, edition 2024)
crates/selvedge-core/          canonical IDs, tensor contracts, ledger, config
crates/selvedge-format/        manifest v1, validator, atomic publisher, readers
crates/selvedge-fixtures/      deterministic synthetic bundle generator
crates/selvedge-cli/           the `selvedge` binary
oracle/                        pinned teacher instrumentation (Python, research-only)
models/                        tracked model spec, lock and reference configs (no weights)
fixtures/                      tracked tiny bundle, prompts and reproduction instructions
docker/                        court Dockerfiles, pinned requirements, image lock
scripts/docker-court.sh        fail-closed Docker court (receipts, limits)
scripts/court                  short aliases over the court
scripts/build-courts.sh        build court images, update docker/IMAGES.lock
scripts/probe-hardware.sh      hardware/toolchain probe, through the courts
scripts/check-commit-identity.sh
scripts/perf_numbers.py        generated-numbers check/write
scripts/courts/                helpers that run *inside* a court
docs/                          mandate copy, format/architecture docs, work log
docs/PERF-NUMBERS.md           numbers generated from receipts
results/                       court receipts, logs, hardware probes
```

The implementation mandate this program follows is tracked at
`docs/MANDATE.md`. The plan and current status are in `docs/WORK-LOG.md`.

## License

Apache-2.0. See `LICENSE`.
