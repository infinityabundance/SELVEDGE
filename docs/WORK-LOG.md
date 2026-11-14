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

Revision: `5fe73aeef887ab6c7e0c8adc70a1708b7b6ac753`. All receipts below were
produced with a clean worktree at that revision, offline, in the Rust court with
the image recorded in `docker/IMAGES.lock`. No performance number was measured
and none is claimed. (The development receipt set committed just before this
revision contains the same tests at `099e0d8`; the only later code changes were
the `AuxiliaryKind` stable-string serialization and the CLI verification-mode
default documented in this log.)

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
| Fixture regenerated and committed deterministically | `results/courts/rust-20261114T190600Z-14db24.json` and its log (generation at `099e0d8`); `results/courts/rust-20261114T191040Z-188d95.json` (byte-for-byte `--check` at the verified revision) |
| `cargo fmt --all --check` | `results/courts/rust-20261114T191009Z-013b4c.json` |
| `cargo clippy --workspace --all-targets -- -D warnings` | `results/courts/rust-20261114T191009Z-75eb72.json` |
| `cargo test --workspace` (unit, Court C, Court D, fixture reproducibility, CLI end-to-end) | `results/courts/rust-20261114T191010Z-3d0b43.json` |
| `selvedge doctor --json` (all checks pass; CUDA reported not compiled in) | `results/courts/rust-20261114T191015Z-3cc54e.json` |
| `selvedge inspect fixtures/tiny-synthetic.slv --json` | `results/courts/rust-20261114T191019Z-b5b3a9.json` |
| `selvedge format validate` with expected model digest and compatibility id | `results/courts/rust-20261114T191022Z-e04c2f.json` |
| `selvedge court c` | `results/courts/rust-20261114T191025Z-1738ec.json` |
| `selvedge court d --workdir target/court-d` | `results/courts/rust-20261114T191028Z-7fe25d.json` |
| Generated numbers match their receipts | `results/courts/perf_numbers-20261114T191031Z-4033b4.json` |
| Every commit carries the operator identity | `results/courts/python-20261114T191032Z-e815b8.json` |

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
8. **Cache invalidation is not implemented because no cache exists yet.** This
   increment has residency records and a hot-reload catalog, but no KV or shard
   cache whose invalidation could be tested. Court D covers the catalog/id
   rejection; cache invalidation belongs to the store phase and must get a named
   test there.

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

## 2026-11-14 — S2: safetensors, exact SwiGLU groups, Court A

Revision: `c0c68a454f936b0979f4a7afc7754f68c445328b`. The Court A receipt is
`results/court-a/20261114T192745Z.json` (stamp `20261114T192745Z`, revision
`c0c68a454f936b0979f4a7afc7754f68c445328b`). Every command below ran offline
(`--network none`) in the Rust court with the image and limits recorded in its
receipt. No model-quality or timing number is claimed.

### Plan

Deliver the exact SwiGLU decomposition in dependency order: a safe streaming
safetensors reader; the typed `FfnTensorMap` and deterministic grouping rules; a
memory-bounded converter that repacks contiguous group extents through the S1
writer; a real reconstruction proof from published bytes (Court A); the CLI
surface (`convert`, `convert verify`, `court a`); a committed synthetic
safetensors fixture and offline demo; and documentation of the down-projection
transpose and tolerance policy.

### What was built

- `selvedge-convert` with modules `safetensors`, `tensor_source`, `ffn`,
  `convert`, `verify`, `synthetic`, `court_a` and `error`.
- Safe safetensors reader: 8-byte LE header length, JSON header with duplicate-key
  detection and `__metadata__`, F32/F16/BF16/I8/U8, typed rejection of every
  other dtype, header and offset caps, overlap/byte-length/shape checks, f16 and
  bf16 exact widening to f32, positional `read_at` reads, and no `unsafe` or
  memory mapping.
- Typed `FfnTensorMap { gate, up, down, orientation }`; `intermediate_first` and
  `hidden_first`; deterministic contiguous groups (`size = ceil(I/n)`) and
  explicit channel lists; disjointness/in-range/completeness validation.
- Repacking: one shard per `(group, role)`. Gate/up are `[|G|, hidden]` row-major.
  Down is a logical `[|G|, hidden]` shard with
  `StorageOrder::Transposed { axes: [1, 0] }` and transform
  `swiglu.down.transpose.v1`, so its physical bytes are the contiguous
  `[hidden, |G|]` copy. Every extent is aligned, hashed and listed in `co_access`.
- Streaming conversion through a spool file: one group at a time, an explicit
  in-flight byte budget, checked arithmetic, no whole-matrix residency.
- `selvedge convert`, `selvedge convert verify` and `selvedge court a`, all with
  `--json`; `--plan` prints the shard plan without writing.
- `tiny-swiglu.safetensors` (one layer, hidden 8, intermediate 10, seed
  `0x5E1AED6E51E12026`), `tiny-swiglu` generation in `selvedge-fixtures`, and
  `scripts/demo-offline.sh`.
- `docs/SHARD-FORMAT.md` layout section and `docs/ARCHITECTURE.md` conversion
  dataflow, memory rule and tolerance policy.

### Hypotheses and results

| hypothesis | result | evidence |
| --- | --- | --- |
| H0: the full group sum reconstructs the dense FFN in f32 within a stated bound. | Confirmed on the synthetic SwiGLU for 21 named scenarios; the measured maxima are printed below from the receipt. | Court A receipt `results/court-a/20261114T192745Z.json`; `court_a` tests. |
| The decomposition survives group permutation, sizes 1/2/3/prime, non-divisible grouping and the empty subset. | Confirmed within the same measured bound. The empty subset is exactly `+0.0`; this is a mathematical zero, not a skip-quality proof. | Named scenarios `group_permutation_invariance`, `group_size_*`, `non_divisible_grouping`, `empty_subset_is_exact_zero`. |
| Repacking is orientation-independent for column-major and explicitly transposed sources. | Confirmed for both storage orders. | Named scenarios `column_major_source`, `transposed_source`. |
| The down shard can be a contiguous transposed copy without a gather-heavy fragment. | Confirmed by decoding the manifest-declared `transposed[1,0]` order and matching the source. | `swiglu.down.transpose.v1` shards; `shard_ids_and_hashes_match_manifest`. |
| f16/bf16 sources preserve the same f32 arithmetic bound after exact widening. | Confirmed on the synthetic fixtures. | Named scenarios `f16_source`, `bf16_source`. |
| Conversion is deterministic and memory-bounded. | Two identical runs produce byte-identical bundles and ids; a 128-byte ceiling succeeds with peak 128 bytes. | `determinism_same_bytes_and_ids`, `small_memory_ceiling`. |

### Measured reconstruction (generated)

Every value below is emitted from the Court A receipt by
`scripts/perf_numbers.py`; it is not hand-written.

- all-groups f32 maximum absolute error:
<!-- BEGIN GENERATED: court-a-all-groups-max-abs-f32 -->
9.336e-07
<!-- END GENERATED: court-a-all-groups-max-abs-f32 -->
- all-groups f32 maximum relative error:
<!-- BEGIN GENERATED: court-a-all-groups-max-rel-f32 -->
2.100e-05
<!-- END GENERATED: court-a-all-groups-max-rel-f32 -->
- all-groups f64 maximum absolute error:
<!-- BEGIN GENERATED: court-a-all-groups-max-abs-f64 -->
4.441e-16
<!-- END GENERATED: court-a-all-groups-max-abs-f64 -->
- all-groups f64 maximum relative error:
<!-- BEGIN GENERATED: court-a-all-groups-max-rel-f64 -->
1.760e-14
<!-- END GENERATED: court-a-all-groups-max-rel-f64 -->
- f16-source f32 maximum absolute error:
<!-- BEGIN GENERATED: court-a-f16-max-abs-f32 -->
2.696e-07
<!-- END GENERATED: court-a-f16-max-abs-f32 -->
- bf16-source f32 maximum absolute error:
<!-- BEGIN GENERATED: court-a-bf16-max-abs-f32 -->
2.141e-07
<!-- END GENERATED: court-a-bf16-max-abs-f32 -->
- acceptance absolute tolerance:
<!-- BEGIN GENERATED: court-a-tolerance-abs -->
1.0e-05
<!-- END GENERATED: court-a-tolerance-abs -->
- acceptance relative tolerance:
<!-- BEGIN GENERATED: court-a-tolerance-rel -->
1.0e-05
<!-- END GENERATED: court-a-tolerance-rel -->
- named scenarios:
<!-- BEGIN GENERATED: court-a-scenario-count -->
21
<!-- END GENERATED: court-a-scenario-count -->
- compared f32 components in the all-groups case:
<!-- BEGIN GENERATED: court-a-compared-components -->
176
<!-- END GENERATED: court-a-compared-components -->
- shards in the all-groups bundle:
<!-- BEGIN GENERATED: court-a-shard-count -->
12
<!-- END GENERATED: court-a-shard-count -->
- all-groups payload bytes:
<!-- BEGIN GENERATED: court-a-payload-bytes -->
6144
<!-- END GENERATED: court-a-payload-bytes -->
- peak in-flight bytes during the all-groups conversion:
<!-- BEGIN GENERATED: court-a-peak-inflight-bytes -->
96
<!-- END GENERATED: court-a-peak-inflight-bytes -->

The pass gate is elementwise `|a - b| <= abs_tol + rel_tol * |b|`. The printed
maximum relative error can exceed `rel_tol` when an individual reference
component is near zero; the absolute term is then the binding part. No bitwise
equality is claimed, and the f64 path confirms the identity holds to f64
rounding.

### Verification performed (receipts, not prose)

| what was verified | evidence |
| --- | --- |
| `cargo fmt --all --check` | `results/courts/rust-20261114T192803Z-9a04be.json` |
| `cargo clippy --workspace --all-targets -- -D warnings` | `results/courts/rust-20261114T192804Z-2bc9f5.json` |
| `cargo test --workspace` (unit, safetensors hostile/proptest, decomposition, named Court A, CLI) | `results/courts/rust-20261114T192804Z-bfd748.json` |
| `selvedge court a` | `results/courts/rust-20261114T192808Z-58ccc8.json`; report `results/court-a/20261114T192745Z.json` |
| `scripts/demo-offline.sh` (generate, convert, validate, verify, court a) | `results/courts/rust-20261114T192809Z-d2af65.json`, `...192810Z-8813b1.json`, `...192811Z-490a12.json`, `...192811Z-80757b.json`, `...192812Z-d2c0de.json`; report `results/demo-offline/20261114T192809Z.json` |
| Every commit carries the operator identity | `results/courts/commit_identity-20261114T192812Z-fe8b25.json` |
| Generated numbers match their receipts | `results/courts/perf_numbers-20261114T192852Z-fd0571.json` (write), `...192857Z-ed9bc8.json` and `...192909Z-901902.json` (checks) |

### Decisions and known limitations

1. **The converter writes unquantized f32.** f16/bf16 sources are widened exactly
   before the f32 reference math; quantization is not part of S2.
2. **The down shard is logically transposed.** Its manifest shape is
   `[|G|, hidden]` and its storage order is `transposed[1,0]`; the bytes are the
   contiguous `[hidden, |G|]` copy. A reader must use the declared order, not the
   shape alone.
3. **Groups are disjoint but completeness is optional.** `--complete` enforces
   exact coverage; without it a partial group set is recorded and the
   compatibility id says `partial`. A partial bundle is not a full FFN
   decomposition.
4. **Orientation is a caller declaration.** Shape inspection cannot distinguish
   `intermediate_first` from `hidden_first` when the two dimensions are swapped,
   so the converter records the declared orientation and validates shapes
   against it.
5. **Tolerance is measured, not proved.** The acceptance constant lives in code;
   the measured maxima and the constant itself are generated from the receipt.
   The relative maximum can be large for near-zero reference components; the
   gate includes the absolute term.
6. **Court A is synthetic and CPU-only.** It proves algebra and byte handling,
   not quality under group skipping. No real checkpoint has been converted yet.
7. **The safetensors reader is not a fuzzer target.** Proptest covers arbitrary
   bytes and random mutations offline; a checked-in `cargo-fuzz` target remains
   future work.
8. **GGUF is not implemented.** Packed block boundaries must not be sliced as
   unquantized floats; that path is deliberately absent.

### Next experiment

S3 (oracle and selectors) can consume the group bundles and the measured Court A
bound. It must not treat a passing full reconstruction as evidence that skipping
groups preserves model quality; that requires the teacher divergence and
counterfactual work in H1/Court B.
