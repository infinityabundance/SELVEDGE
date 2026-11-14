# Work log

Chronological, honest record. Every claim below names the receipt that supports
it. Numbers are never written here: machine facts live in
`docs/PERF-NUMBERS.md` (generated) and in the hardware probe receipt under
`results/hardware/`.

## 2026-11-14 — S0: repository bootstrap and court infrastructure

### Plan

The program is implemented in dependency order. S0 (this session) delivers only
the reproducibility substrate that every later subphase depends on:

1. **S0 — bootstrap and courts (this session).** Repository, licence, mandate
   copy, fail-closed Docker court with receipts, pinned court images, commit
   identity court, generated-number machinery, hardware probe.
2. **S1 — Rust workspace skeleton.** `selvedge-core` (deterministic IDs,
   budgets, errors, counters), `selvedge-format` (versioned shard manifests,
   validator, checksums), `selvedge-convert` (tensor-aligned SwiGLU
   decomposition and repacking), plus CPU test court wiring (Court A).
3. **S2 — synthetic fixtures and exact reconstruction.** Tiny non-proprietary
   SwiGLU weights, full-group reconstruction parity, group permutations,
   shape/orientation variants, tolerance calibration by dtype.
4. **S3 — oracle path.** Pinned Python teacher instrumentation, activation and
   intervention capture, causal utility observations (Courts B and C).
5. **S4 — atlas, simulator, scheduler contracts.** Parameter ledger, selection
   decisions, residency snapshots, transfer requests/completions, trace replay,
   cost model, deadline simulation.
6. **S5+ — measured I/O and single-GPU integration.** Only after the quality
   gates in `docs/MANDATE.md` pass.

This log records S0 only. No model crate exists yet and no experiment has been
run; no result is claimed.

### Environment

Hardware and toolchain facts are collected by `scripts/probe-hardware.sh`
through the courts and pinned by stamp and revision in
`results/hardware/probe-20261114T184814Z.json`; the subset intended for
quotation is rendered into `docs/PERF-NUMBERS.md` from that receipt. Both are
regenerable and neither is edited by hand.

Probe evidence chain (all receipts are under `results/courts/`):

- host CPU/RAM/swap/NVMe/kernel facts: Python court run with the host root
  mounted read-only at `/hostroot`;
- GPU facts: `--gpu` court run executing `nvidia-smi` inside the container;
- Rust and Python versions: the `toolchain` block of the Rust and Python court
  receipts named in the probe's `evidence` field.

### What was built

- `scripts/docker-court.sh` — the only execution path for builds and tests.
  Hard `--memory`, `--memory-swap` equal to memory, `--pids-limit`, `--cpus`,
  bounded `/tmp`, `nofile`, `no-new-privileges`, `cap-drop ALL`, `--init`.
  Refuses to run when a limit is empty, zero or malformed, when Docker is
  missing, or when the daemon reports that limits cannot be enforced. After the
  container exits it inspects the container and compares the *applied*
  configuration against the requested one; a mismatch writes a receipt recording
  the mismatch and exits non-zero.
- `scripts/court` — short aliases (`rust test`, `python -m pytest`, `io`,
  `commit_identity`, `perf_numbers`).
- `docker/Dockerfile.rust-court`, `docker/Dockerfile.python-court`,
  `docker/requirements-court.txt`, `docker/IMAGES.lock` — content-addressed tags
  (`<date>-<base-tag>-<sha256(inputs)[:12]>`), base image digests, observed
  toolchains, and the image id of each build.
- `scripts/build-courts.sh` — builds both images under bounded resources and
  rewrites the lock file atomically.
- `scripts/check-commit-identity.sh` — fails closed unless the repository-local
  identity is the operator's, every scanned commit's author and committer are
  the operator, and no commit carries an agent identity or AI attribution.
- `scripts/perf_numbers.py`, `docs/perf-numbers.json`, `docs/PERF-NUMBERS.md` —
  generated numbers from receipts by JSON pointer, with stamp/revision pinning,
  tracked-evidence enforcement, orphan-sentinel detection and unprinted-number
  detection.
- `scripts/probe-hardware.sh` plus `scripts/courts/hardware_probe.py`,
  `gpu_probe.py`, `assemble_hardware_probe.py` — hardware probe assembled from
  court runs.
- Negative self-tests that live in the repository so later agents can re-run
  them: `scripts/courts/perf_numbers_negative_cases.sh`,
  `scripts/courts/commit_identity_negative_cases.sh`.

### Verification performed (receipts, not prose)

| what was verified | evidence |
| --- | --- |
| Rust court image runs, offline | `results/courts/rust-20261114T184409Z-105945.json` |
| Rust compiler version | `results/courts/rust-20261114T184409Z-2db7ae.json` |
| Python court image runs, offline | `results/courts/python-20261114T184409Z-f09b4c.json` |
| Real `cargo test` in a temporary crate inside bounded `/tmp` | `results/courts/rust-20261114T184410Z-b0d54e.json` and its log |
| Memory cap enforced (allocator killed) | `results/courts/python-20261114T184414Z-60a177.json` (`exit_code` 137, `oom_killed` true) |
| `/tmp` bounded (write stops at the configured tmpfs size) | `results/courts/io-20261114T184414Z-e69e04.json` and its log |
| Fail-closed refusals for empty/zero/malformed limits | `results/courts/refusals/fail-closed-20261114T184422Z.log` |
| Hardware probe through the courts (including GPU access) | `results/hardware/probe-20261114T184814Z.json` |
| Generated numbers match their receipts | `results/courts/perf_numbers-20261114T184543Z-36310b.json` |
| Generator fails closed on nine deliberate breakages | `results/courts/python-20261114T184538Z-e0ef83.json` and its log |
| Commit identity passes for HEAD and for all commits | `results/courts/commit_identity-20261114T184548Z-6cbdd2.json`, `results/courts/commit_identity-20261114T184549Z-1e7389.json` |
| Identity court fails closed on four breakages | `results/courts/python-20261114T184559Z-76decd.json` and its log |

Every one of those receipts records the image reference, image id, applied
limits (verified by inspecting the live container), network mode, exit code,
duration and the toolchain versions observed inside the container.

### Known limitations and decisions made in S0

1. **GPU access inside containers does not go through `--gpus all` on this
   host.** Docker reports `failed to discover GPU vendor from CDI: no known GPU
   vendor found` because no NVIDIA container toolkit / CDI specification is
   installed. The court therefore falls back to explicit device nodes plus the
   host driver libraries, records `gpu.mode="device-passthrough"`, and records
   the exact `--gpus all` error text in every GPU receipt. A GPU court fails
   closed if neither path works; it never silently runs on CPU. Installing the
   container toolkit would remove the workaround and is an operator decision.
2. **Locally built images have no registry digest.** Receipts record
   `image_digest_source="image-id (locally built, never pushed to a registry)"`
   so a reader can tell an image id from a registry digest. Nothing pretends an
   image id is a registry-pinned digest.
3. **`docker build` has no pids flag.** Image builds are bounded by memory, swap
   and `nofile` only, and are not courts. Every *run* has the full limit set.
4. **Receipts must be committed before `perf_numbers --check` passes.**
   Untracked evidence is treated as absent evidence, deliberately. The order is:
   run the court, commit the receipt, then check.
5. **The court has no network by default.** `--net` is required and is recorded
   in the receipt as `network_requested`. Image builds and dependency fetching
   are separate, explicit, orchestrated steps.

### Next subphase

S1: create the Cargo workspace and the `selvedge-core` / `selvedge-format` /
`selvedge-convert` crates with Court A tests run through `scripts/court rust
test`. Nothing in S1 may quote a number that is not generated from a receipt.
