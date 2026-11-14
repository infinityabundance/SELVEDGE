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
`results/hardware/probe-20261114T184849Z.json`; the subset intended for
quotation is rendered into `docs/PERF-NUMBERS.md` from that receipt. Both are
regenerable and neither is edited by hand. Rebuilding the manifest after a
re-probe is `python3 scripts/courts/build_manifest_from_probe.py --probe <receipt>`
followed by `scripts/court perf_numbers --write`. Earlier probe files in the
same directory are retained as history; this log and the manifest name the
current one.

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
| Hardware probe through the courts (including GPU access) | `results/hardware/probe-20261114T184849Z.json` |
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

## 2026-11-14 — S1: workspace, data contracts, shard format, CLI and Courts C/D

Revision: `099e0d815968c1333e15ee4e1f608f2b61fdd7d8`. All receipts below were
produced with a clean worktree at that revision, offline, in the Rust court with
the image recorded in `docker/IMAGES.lock`. No performance number was measured
and none is claimed.

### Plan

S1 delivers the foundation every later phase codes against, in dependency
order: a Cargo workspace; `selvedge-core` data contracts with a real budget
gate; `selvedge-format` with manifest v1, validation and atomic publication;
the `selvedge` CLI (`doctor`, `inspect`, `format validate`); deterministic
fixtures; and executable Courts C and D. Tensor conversion (S2) and the
oracle/selectors (S3) are explicitly out of scope.

### What was built

- `selvedge-core`: canonical little-endian byte writer and sorted-key JSON;
  `Sha256Digest`; `ModelDigest` over a normalized checkpoint fileset;
  content-derived `ShardId`; `TensorRole`, `DType` (including packed and
  explicit-unsupported variants), `QuantScheme`, `StorageOrder`,
  `TensorSlice`, `ChannelSet`, `ParamIdentity`, `ElementCount`; checked size
  arithmetic; `ParameterLedger` with union accounting, repeated-read counters,
  FLOPs, tier bytes, non-learned auxiliary bytes, `Budget`/`BudgetVerdict`,
  `DecisionPlan`, a within-budget fallback hook and `TokenAccounting`;
  `SelectionDecision` with `NativeSetAttestation`/`ExactSelection`;
  residency and transfer records; `CostModel`/`Backend` traits; typed `Config`
  with hard limits and canonical hash; test-support temp directories.
- `selvedge-format`: manifest v1 with per-shard role/layer/channels/shape/
  orientation/dtype/quantization/offset/length/checksum/alignment/
  compatibility id/transform/links/calibration; validator covering length,
  magic, version, bounded manifest parsing, semantic checks, content-derived
  ids, extent bounds, overlap, alignment, payload hashes, expected model digest
  and compatibility id, selected/structural verification modes and
  cancellation; atomic publisher (temporary file, fsync, full validation of the
  temporary file, hard-link or rename, directory fsync); sidecar publication
  and staleness detection; `MemoryBundle`; `BundleFile` with capped and
  cancellable reads; hot-reload catalog.
- `selvedge-cli`: `doctor [--json] [--probe-host]`, `inspect <bundle|manifest>
  [--json]`, `format validate <bundle>` with expected-digest/compatibility and
  verification-mode flags, and executable `court c` / `court d`.
- `selvedge-fixtures`: in-repo xorshift PRNG and a two-layer, four-group
  synthetic bundle with one block-quantized shard; tracked artifact and
  regeneration command in `fixtures/README.md`.
- `docker/Dockerfile.rust-court`: adds the `rustfmt` and `clippy` components so
  the offline format and lint courts work; `docker/IMAGES.lock` updated by
  `scripts/build-courts.sh rust`.
- `docs/SHARD-FORMAT.md`, `docs/ARCHITECTURE.md`, `fixtures/README.md`.

### Hypotheses and results

| hypothesis | result | evidence |
| --- | --- | --- |
| Content-derived shard ids can be recomputed and tamper-detected without hashing physical layout fields. | Confirmed. The validator recomputes every id; a mutated role produces `ShardIdMismatch`, duplicate ids are rejected, and every identity field changes the id. | `court_d` tests `stored_id_that_is_not_content_derived_is_rejected`, `duplicate_shard_ids_are_rejected`; core `ids` unit tests. |
| Union accounting can count tied/overlapping/recurrent parameters once while preserving reads and FLOPs separately. | Confirmed on the synthetic cases. Tied embeddings count once with both reads; overlapping ranges merge; recurrent reuse reports one unique set and every read; adapters/memory/router/verifier count as learned. | Court C named scenarios and tests, receipt `619bb7`. |
| A compliance budget gate can make an over-budget decision impossible to ignore. | Confirmed. Exactly at the limit is `Within`; one element over is `Err(Exceeded)`; teacher mode returns only a labelled non-compliant verdict; a fallback is re-checked and a still-over-budget fallback is rejected. | Court C boundary/fallback tests. |
| Quantization metadata can be excluded from learned counts without losing it. | Confirmed. Scales and zero points are reported as auxiliary bytes; the learned count is unchanged. | Court C `quantized_metadata_excluded_from_learned_count`. |
| Atomic publication plus full validation of the temporary file prevents torn reads under crash and concurrency. | Confirmed on the tested platform. Interrupted and short/long streams leave no final artifact and preserve an existing bundle; concurrent readers only ever see complete bundles; immutable bundles refuse overwrite. | Court D tests; receipt `c40df8`. |
| The parser can reject malformed, truncated, overlapping, misaligned, version-skewed and unsupported inputs without panicking. | Confirmed by named tests, a deterministic hostile corpus, and property tests over arbitrary bytes and random byte mutations. | `court_d` proptest cases and hostile-corpus test. |

### Verification performed (receipts, not prose)

| what was verified | evidence |
| --- | --- |
| Fixture regenerated and committed deterministically | `results/courts/rust-20261114T190600Z-14db24.json` and its log |
| `cargo fmt --all --check` | `results/courts/rust-20261114T190640Z-b701f5.json` |
| `cargo clippy --workspace --all-targets -- -D warnings` | `results/courts/rust-20261114T190643Z-14200a.json` |
| `cargo test --workspace` (unit, Court C, Court D, fixture reproducibility, CLI end-to-end) | `results/courts/rust-20261114T190647Z-1c906b.json` |
| `selvedge doctor --json` (all checks pass; CUDA reported not compiled in) | `results/courts/rust-20261114T190651Z-685d81.json` |
| `selvedge inspect fixtures/tiny-synthetic.slv --json` | `results/courts/rust-20261114T190655Z-dac5bb.json` |
| `selvedge format validate` with expected model digest and compatibility id | `results/courts/rust-20261114T190658Z-63dc2e.json` |
| `selvedge court c` | `results/courts/rust-20261114T190701Z-619bb7.json` |
| `selvedge court d --workdir target/court-d` | `results/courts/rust-20261114T190704Z-c40df8.json` |

Every receipt records the image reference and id, applied limits verified by
container inspection, network mode (`none`), exit code, duration and toolchain
versions. The CLI and fixture runs use the same court and limits.

### Decisions and known limitations

1. **Edition and resolver.** rustc 1.98.1 supports edition 2024, so the
   workspace uses edition 2024, `rust-version = "1.98"` and resolver 3. The
   toolchain is pinned in `docker/IMAGES.lock`.
2. **Manifest and payload are not signed.** SHA-256 detects corruption and naive
   tampering; an attacker who rewrites the whole bundle can produce a
   self-consistent artifact. Callers should pin the model digest, revision and
   compatibility id externally. The sidecar binds the manifest digest but is a
   separate file. Padding bytes are not hashed. This is stated in
   `docs/SHARD-FORMAT.md` rather than hidden.
3. **Alignment is not benchmarked here.** 4096 is the configured default and a
   benchmark target, not a measured optimum. The fixture uses 512 to keep the
   tracked artifact small.
4. **No model path.** There is no safetensors/GGUF reader, no tensor
   conversion, no oracle, no selector and no runtime. `CostModel` and `Backend`
   are contracts only. CUDA is not compiled into this build and `doctor` says
   so instead of inventing GPU facts.
5. **Exactness is typed, not tested end-to-end.** `ExactSelection` can only be
   constructed by native-set equality or explicit reclassification. Court E
   parity against a real backend remains future work.
6. **`cargo-fuzz` is not checked in.** The parser boundary is covered by
   `proptest` and a deterministic hostile corpus; a fuzz target is documented as
   future work in `docs/SHARD-FORMAT.md`.
7. **The fixture is synthetic.** Its model/tokenizer digests are labels for the
   fixture, not a real checkpoint, and the quantized shard's scales are
   synthetic.

### Next experiment

S2 (tensor conversion) should use the interfaces listed in
`docs/ARCHITECTURE.md`: build `ShardRecord`s with `ShardId::derive`, use
`TensorSlice`/`bytes_for` for byte math and `publish_bundle` for publication,
then prove **Court A** exact reconstruction on a deterministic synthetic SwiGLU
before touching any upstream checkpoint. The first measurable question is
whether full-group reconstruction is exact in F32 within a stated tolerance and
whether grouped slices survive permutation and axis-orientation changes. No
timings or model-quality numbers should be published from that increment until
they come from a receipt.
