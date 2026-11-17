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

Revision: `52225ef03f1d1924867a60863a5c0480953eb29f`. The Court A receipt is
`results/court-a/20261114T193212Z.json` (stamp `20261114T193212Z`, revision
`52225ef03f1d1924867a60863a5c0480953eb29f`). Every command below ran offline
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
| H0: the full group sum reconstructs the dense FFN in f32 within a stated bound. | Confirmed on the synthetic SwiGLU for the full named scenario matrix; the measured maxima are printed below from the receipt. | Court A receipt `results/court-a/20261114T193212Z.json`; `court_a` tests. |
| The decomposition survives group permutation, sizes one/two/three/prime, non-divisible grouping and the empty subset. | Confirmed within the same measured bound. The empty subset is exactly `+0.0`; this is a mathematical zero, not a skip-quality proof. | Named scenarios `group_permutation_invariance`, `group_size_*`, `non_divisible_grouping`, `empty_subset_is_exact_zero`. |
| Repacking is orientation-independent for column-major and explicitly transposed sources. | Confirmed for both storage orders. | Named scenarios `column_major_source`, `transposed_source`. |
| The down shard can be a contiguous transposed copy without a gather-heavy fragment. | Confirmed by decoding the manifest-declared `transposed[1,0]` order and matching the source. | `swiglu.down.transpose.v1` shards; `shard_ids_and_hashes_match_manifest`. |
| f16/bf16 sources preserve the same f32 arithmetic bound after exact widening. | Confirmed on the synthetic fixtures. | Named scenarios `f16_source`, `bf16_source`. |
| Conversion is deterministic and memory-bounded. | Repeated identical runs produce byte-identical bundles and ids; a tight ceiling succeeds without exceeding it. | `determinism_same_bytes_and_ids`, `small_memory_ceiling`. |

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
128
<!-- END GENERATED: court-a-peak-inflight-bytes -->

The pass gate is elementwise `|a - b| <= abs_tol + rel_tol * |b|`. The printed
maximum relative error can exceed `rel_tol` when an individual reference
component is near zero; the absolute term is then the binding part. No bitwise
equality is claimed, and the f64 path confirms the identity holds to f64
rounding.

### Verification performed (receipts, not prose)

| what was verified | evidence |
| --- | --- |
| `cargo fmt --all --check` | `results/courts/rust-20261114T193201Z-745b98.json` |
| `cargo clippy --workspace --all-targets -- -D warnings` | `results/courts/rust-20261114T193201Z-6e1e55.json` |
| `cargo test --workspace` (unit, safetensors hostile/proptest, decomposition, named Court A, CLI) | `results/courts/rust-20261114T193202Z-b1b1e9.json` |
| `selvedge court a` | `results/courts/rust-20261114T193212Z-98eac9.json`; report `results/court-a/20261114T193212Z.json` |
| `scripts/demo-offline.sh` (generate, convert, validate, verify, court a) | `results/courts/rust-20261114T193242Z-bf3cae.json`, `...193244Z-32ebf0.json`, `...193245Z-1df203.json`, `...193245Z-428f12.json`, `...193247Z-d018e4.json`; report `results/demo-offline/20261114T193242Z.json` |
| Every commit carries the operator identity | `results/courts/commit_identity-20261114T193248Z-14c438.json` |
| Generated numbers match their receipts | `results/courts/perf_numbers-20261114T193222Z-457677.json` (write) and `...193224Z-bee417.json` (check) |

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

## 2026-11-14 — S3a: oracle court, real checkpoints, exact accounting, real-activation parity

Revision: `a67671aecf9296a41528cd6a842437d6e3c629dd` for the real-activation
and real-checkpoint verification receipts, `d9bb4ce17c699a8df22c204e890f59c6b970bc35`
for the accounting, model-fetch and oracle-probe receipts, `9bccaffe8b4f3e11113e0c389d8fb3418ab779a4`
for the 0.6B conversion receipt, and `8b94ad5b...` for the final lint/test
receipts. All are real commits; every receipt names its own revision and stamp.
No quality or timing number is claimed.

### Plan

Answer two questions before any selector work: is the 1e9 unique-active
ceiling actually binding for the candidate teachers, and can a real checkpoint
be converted into group shards that reproduce the teacher's FFN on real
activations? Deliver: a pinned oracle court; pinned, hash-verified model
acquisition; exact Rust accounting from real safetensors; real conversion and
real-activation parity; receipts and generated numbers for each.

### What was built

- `docker/Dockerfile.oracle` plus `docker/requirements-oracle.txt`: digest-pinned
  Python base, pinned torch CUDA build, transformers, tokenizers, safetensors,
  numpy and pytest. `docker/IMAGES.lock` gains an `oracle` entry;
  `scripts/build-courts.sh oracle` builds it.
- `scripts/docker-court.sh` gains the `oracle` court: persistent HF cache mounted
  read-write at `/hf-cache`, `HF_HOME`/`HF_HUB_CACHE` set from it, offline by
  default with `--net` for fetch steps, `--gpu` device passthrough, and the mount
  plus GPU mode recorded in every receipt. The Rust court mounts the same cache
  read-only.
- `oracle/probe_oracle.py`: pinned CPU and GPU matmul/SiLU checks and version
  report. `oracle/fetch_models.py` + `scripts/fetch-models.sh`: idempotent,
  LFS-oid/SHA-256 verification, fail-closed on mismatch, prints the resolved
  revision. `oracle/capture_activations.py`: teacher MLP hooks in bf16 and f32,
  with TF32 off and bf16 reduced-precision reduction off.
- `oracle/tokenizer_identity.py`, `scripts/courts/config_projection.py`,
  `scripts/courts/disk_free.py`.
- `models/model-spec.json`, `models/MODELS.lock.json` (tracked hashes, no
  weights), `models/reference-configs/qwen3-4b-config.json`,
  `fixtures/prompts/qwen3-teacher-prompts.json`, `docs/MODELS.md`,
  `docs/BUDGET-MATH.md`.
- `selvedge-convert::accounting` and `selvedge inspect-model`: exact component
  accounting from real headers, verified embedding/output tie, category totals,
  backbone/FFN split, ceiling arithmetic and documented FLOP estimates.
- `selvedge-convert::real_parity`, `selvedge convert verify-real` and
  `selvedge court a-real`: dense-vs-grouped and grouped-vs-teacher comparisons
  on real activations, with an explicit bf16-emulation path.

### Hypotheses and results

| hypothesis | result | evidence |
| --- | --- | --- |
| CUDA device passthrough works inside the oracle container. | Confirmed with a real GPU matmul and SiLU; the probe and receipt record the device, capability and `--gpus all` fallback reason. | `results/oracle-probe/20261114T201533Z.json`; `results/courts/oracle-20261114T201545Z-58530d.json` |
| The strict ceiling is not binding for Qwen3-0.6B. | Confirmed: the full dense checkpoint is below the ceiling with substantial headroom. Sparsity there is a cost choice, not compliance. | `results/accounting/20261114T201533Z-qwen3-0.6b.json`; `docs/BUDGET-MATH.md` |
| The ceiling is binding for Qwen3-1.7B, and the always-on backbone still fits. | Confirmed: dense weights exceed the ceiling, the backbone fits, and only a minority share of FFN channels can run per token. | `results/accounting/20261114T201533Z-qwen3-1.7b.json`; `docs/BUDGET-MATH.md` |
| Qwen3-4B would make the ceiling binding in a useful way. | **Rejected.** Its always-on backbone alone exceeds the ceiling, so no group selection can make it compliant. It is recorded as not downloaded. | `results/accounting/20261114T205700Z-qwen3-4b-config-projection.json` |
| Full-group reconstruction from the published bundle is exact on real weights and real activations. | Confirmed: the grouped f32 path and the dense f32 path agreed bitwise for every layer and every activation row. | `results/real-parity/20261114T204209Z/real-parity-qwen3-0.6b.json`, `...-1.7b.json` |
| The bundle reproduces the teacher's f32-equivalent computation. | Confirmed within the documented f32 bound, with zero violations across both checkpoints. | same receipts |
| The bundle reproduces the teacher's native bf16 output when the teacher's rounding sequence is emulated. | Confirmed within the documented bf16 bound, with zero violations; plain f32 arithmetic on bf16 inputs does not reproduce it and is reported as informational. | same receipts |
| bf16 storage would be the honest representation. | **Rejected in favour of f32.** f32 is an exact widening and gives a defined f32-equivalent path; a bf16 copy would still need the explicit rounding sequence, so the two-byte saving is not worth an ambiguous claim. | `docs/ARCHITECTURE.md` storage section; receipt sizes |

### Measured conversion and parity (generated)

Qwen3-0.6B, default grouping of sixteen contiguous groups per layer:

- layers verified:
  <!-- BEGIN GENERATED: worklog-layers -->
28
  <!-- END GENERATED: worklog-layers -->
- activation rows per layer:
  <!-- BEGIN GENERATED: worklog-rows -->
64
  <!-- END GENERATED: worklog-rows -->
- manifest shards validated:
  <!-- BEGIN GENERATED: worklog-06b-shards -->
1344
  <!-- END GENERATED: worklog-06b-shards -->
- payload bytes:
  <!-- BEGIN GENERATED: worklog-06b-payload -->
1056964608
  <!-- END GENERATED: worklog-06b-payload -->
- whole bundle bytes:
  <!-- BEGIN GENERATED: worklog-06b-bundle-file -->
1057882112
  <!-- END GENERATED: worklog-06b-bundle-file -->
- grouped-from-bundle vs dense-from-checkpoint max abs:
  <!-- BEGIN GENERATED: worklog-06b-bundle-dense-max-abs -->
0.0
  <!-- END GENERATED: worklog-06b-bundle-dense-max-abs -->
- grouped f32 vs teacher f32 max abs:
  <!-- BEGIN GENERATED: worklog-06b-f32-teacher-max-abs -->
0.005859375
  <!-- END GENERATED: worklog-06b-f32-teacher-max-abs -->
- grouped bf16-emulated vs teacher native bf16 max abs:
  <!-- BEGIN GENERATED: worklog-06b-bf16-teacher-max-abs -->
0.5
  <!-- END GENERATED: worklog-06b-bf16-teacher-max-abs -->

Qwen3-1.7B, same grouping:

- manifest shards validated:
  <!-- BEGIN GENERATED: worklog-17b-shards -->
1344
  <!-- END GENERATED: worklog-17b-shards -->
- payload bytes:
  <!-- BEGIN GENERATED: worklog-17b-payload -->
4227858432
  <!-- END GENERATED: worklog-17b-payload -->
- whole bundle bytes:
  <!-- BEGIN GENERATED: worklog-17b-bundle-file -->
4228780032
  <!-- END GENERATED: worklog-17b-bundle-file -->
- grouped-from-bundle vs dense-from-checkpoint max abs:
  <!-- BEGIN GENERATED: worklog-17b-bundle-dense-max-abs -->
0.0
  <!-- END GENERATED: worklog-17b-bundle-dense-max-abs -->
- grouped f32 vs teacher f32 max abs:
  <!-- BEGIN GENERATED: worklog-17b-f32-teacher-max-abs -->
0.021484375
  <!-- END GENERATED: worklog-17b-f32-teacher-max-abs -->
- grouped bf16-emulated vs teacher native bf16 max abs:
  <!-- BEGIN GENERATED: worklog-17b-bf16-teacher-max-abs -->
4.0
  <!-- END GENERATED: worklog-17b-bf16-teacher-max-abs -->

Oracle environment:

- torch:
  <!-- BEGIN GENERATED: worklog-oracle-torch -->
2.15.0+cu132
  <!-- END GENERATED: worklog-oracle-torch -->
- CUDA runtime:
  <!-- BEGIN GENERATED: worklog-oracle-cuda -->
13.2
  <!-- END GENERATED: worklog-oracle-cuda -->
- transformers:
  <!-- BEGIN GENERATED: worklog-oracle-transformers -->
5.23.0
  <!-- END GENERATED: worklog-oracle-transformers -->
- GPU name:
  <!-- BEGIN GENERATED: worklog-gpu-name -->
NVIDIA GeForce RTX 4080 SUPER
  <!-- END GENERATED: worklog-gpu-name -->
- GPU capability:
  <!-- BEGIN GENERATED: worklog-gpu-capability -->
8.9
  <!-- END GENERATED: worklog-gpu-capability -->
- GPU SiLU maximum absolute error in the probe:
  <!-- BEGIN GENERATED: worklog-gpu-silu-max-abs -->
1.1920928955078125e-07
  <!-- END GENERATED: worklog-gpu-silu-max-abs -->

Qwen3-0.6B checkpoint weight bytes:
  <!-- BEGIN GENERATED: worklog-fetch-06b-weight-bytes -->
1503300328
  <!-- END GENERATED: worklog-fetch-06b-weight-bytes -->

### Verification performed (receipts, not prose)

| what was verified | evidence |
| --- | --- |
| Oracle image builds; CPU probe passes | `results/courts/oracle-20261114T201540Z-131c57.json` |
| GPU probe runs a real CUDA matmul and SiLU inside the container | `results/courts/oracle-20261114T201545Z-58530d.json`; `results/oracle-probe/20261114T201533Z.json` |
| Model cache verifies offline against the lock | `results/courts/oracle-20261114T201552Z-20d92f.json`; `results/models/fetch-20261114T201604Z.json` |
| Exact accounting on both real checkpoints | `results/accounting/20261114T201533Z-qwen3-0.6b.json`, `...-1.7b.json` |
| Tokenizer identity recorded | `results/models/tokenizer-20261114T205700Z-qwen3-0.6b.json`, `...-1.7b.json` |
| Qwen3-4B backbone projection from its pinned config | `results/accounting/20261114T205700Z-qwen3-4b-config-projection.json` |
| Disk space measured, not assumed | `results/disk/disk-20261114T205532Z.json` |
| Real conversion, 0.6B and 1.7B, sixteen groups per layer | `results/courts/rust-20261114T201446Z-ee1d5e.json` (0.6B), `results/courts/rust-20261114T201625Z-305069.json` (1.7B) |
| Random-input reconstruction at real weight scale | `results/courts/rust-20261114T204213Z-97474e.json` (0.6B), `results/courts/rust-20261114T204359Z-b277f9.json` (1.7B); the earlier 1.7B failure that triggered recalibration is `results/courts/rust-20261114T201929Z-62b121.json` |
| Real-activation Court A additions, both checkpoints | `results/real-parity/20261114T204209Z/real-parity-qwen3-0.6b.json`, `...-1.7b.json`; court receipts `results/courts/rust-20261114T205417Z-58abb5.json`, `results/courts/rust-20261114T205423Z-124bf4.json` |
| `cargo fmt --all --check` | `results/courts/rust-20261114T210129Z-aa00a9.json` |
| `cargo clippy --workspace --all-targets -- -D warnings` | `results/courts/rust-20261114T210129Z-51235e.json` |
| `cargo test --workspace` | `results/courts/rust-20261114T210129Z-00b9eb.json` |
| Generated numbers match their receipts | `results/courts/perf_numbers-20261114T210015Z-3d3fd3.json` |
| Every scanned commit carries the operator identity | `results/courts/commit_identity-20261114T210016Z-6862c1.json` |

### Decisions and known limitations

1. **f32 bundle storage is deliberate.** It is an exact widening of bf16 and
   the basis of the f32-equivalent claim. A bf16 bundle would save two bytes per
   element but would not remove the need for the documented rounding sequence.
2. **The default grouping is sixteen contiguous groups per layer.** That is a
   conversion default, not a measured optimum. It keeps shards large enough to
   be meaningful on NVMe and small enough to exercise partial-group arithmetic.
3. **Embedding and output projection are counted as whole matrices.** With the
   verified tie they are counted once. A row-only convention would change the
   budget arithmetic and is not used.
4. **The tokenizer vocabulary is smaller than the embedding table.** Unused
   rows are still counted as learned parameters. This is recorded in
   `docs/MODELS.md`.
5. **The real conversations use one prompt set and 64 activation rows.** That
   is enough to test the FFN identity on real inputs, not enough to support a
   quality claim. Skipping groups still changes the function; Court B must
   measure that.
6. **bf16 emulation is a model of PyTorch's arithmetic, not a proof of its
   kernels.** It matches the captured outputs within the documented bound on
   these inputs. A different backend or fused kernel could round differently.
7. **No measured FLOPs or latency.** All FLOP values are formula estimates.
8. **`--gpus all` still fails without a CDI specification.** The court uses
   explicit device passthrough and records the fallback; this is inherited from
   S0 and remains an operator decision.
9. **The 1.7B random-input verification needed recalibration.** The
   tiny-fixture bound was tighter than the f32 error of a real wide FFN. The
   compile-time bound is documented in `selvedge-convert`, its measured maxima
   are in receipts, and the f64 path still agrees to a much tighter measured
   error. This is calibration, not a relaxed claim: a structural error would be
   orders of magnitude larger.

### Next experiment

S3b: capture per-group counterfactual teacher divergence on a calibrated prompt
set, starting with the 1.7B teacher because it is the checkpoint whose compliant
budget forces FFN selection. The first measurable question is whether a
per-token top-k group selection at the affordable share stays closer to the
teacher than random or magnitude selection, with a pre-registered kill
condition. The 0.6B model remains the control where the ceiling is not binding.

## 2026-11-14 — S3b: pre-registered quality/active-parameter frontier under exact group omission

Revisions: `dad6619c8514cb4a52b3420c007a47fc43e1b47e` (pre-registration and
dataset pins), `d66c6e3b44ab3a1084971c71844f514d4dce1bb7` (instrument, sweep and
Rust cross-check code), `9ea017fd6071a8fb75e88f4254a9f6cf83b23c30` (hash-checked
split receipt and stable dataset lock). Every measurement receipt in this
section names `9ea017f` and was produced from a clean worktree; an earlier set
of receipts from a dirty worktree is retained under
`results/frontier/superseded-dirty-752172b/` and is bitwise identical to the
clean set on every per-arm aggregate (checked for the uniform sweep; the same
determinism is visible in the greedy and control receipts).

No measured number is repeated in this log. The frontier report is the source:
`results/M1-FRONTIER.md`, generated from `results/frontier/frontier.json` and
the receipts it names.

### Plan

Register the experiment before measuring it, pin the evaluation data, build an
exact masked-MLP instrument and validate it against the dense teacher and the
Rust grouped path, then measure the quality-versus-active-parameter frontier of
Qwen3-1.7B under the strict ceiling with Qwen3-0.6B as a control. The registered
thresholds and kill conditions are in `docs/EXPERIMENT-CONTRACT.md`.

### What was built

- `docs/EXPERIMENT-CONTRACT.md`: the first real contract. Hypotheses H0/H1/H1a-c,
  primary and secondary metrics, arms, sample sizes, split rule, leakage
  prevention, bootstrap method, numeric thresholds, kill conditions K1-K5, and
  two dated amendments (A1: article-level documents and license correction,
  before measurement; A2: frequency-selector formula/gloss conflict, before any
  sweep).
- `data/dataset-spec.json`, `data/datasets.lock.json`,
  `oracle/fetch_datasets.py`, `scripts/fetch-datasets.sh`: pinned parquet files
  with HF LFS object ids verified as SHA-256. `docs/DATASETS.md` records
  revisions, licenses, roles and contamination caveats.
- `oracle/dataset_prep.py`: article parsing, per-document hash ranking,
  5-shingle near-duplicate check, windowing, code/math selection, and a split
  receipt with per-document hashes and counts. The derived token file is
  gitignored and its SHA-256 is recorded; the file is content-only so
  re-derivation is hash-stable.
- `oracle/masked_mlp.py`: the exact instrument. `grouped_forward` computes each
  group's contribution explicitly; `MaskedMlp` applies a per-channel 0/1 mask
  and reuses the original projection modules; `MaskController` suspends and
  resumes the wrappers so the dense teacher can run unmodified; calibration
  statistics and active-parameter counting live here.
- `oracle/frontier_sweep.py`: `stats`, `exactness`, `cross-check`, `sweep`
  (uniform/layers/control), `greedy` and `analyze`. Every subcommand writes a
  receipt with stamp, revision, environment, model revision, data hashes,
  per-article metrics and timings. `analyze` performs the paired bootstrap,
  applies the registered thresholds, and writes the JSON, SVG and table.
- `oracle/tests/test_masked_mlp.py`: Court B unit tests with hand-computed
  ground truth: group bounds, group sums, a partial sum that must differ from
  dense, mask/dense equality, mask/grouped equality, empty selection, error
  propagation through a second layer, and the KL/NLL definitions.
- Rust: `selvedge-convert::masked` and `selvedge court b-real` read the real f32
  bundle one layer at a time and compute the grouped sum for an arbitrary group
  subset. `load_bundle_layer` reads only the requested layer. The input/output
  are raw little-endian f32 with a bounded read and a SHA-256 in the report.
- `results/M1-FRONTIER.md`, `results/frontier/frontier.{json,svg,md}` and
  `docs/LIMITATIONS.md`.

### Hypotheses and results

| hypothesis | result | evidence |
| --- | --- | --- |
| G1: the masked-all path equals the unmodified dense teacher. | **Confirmed bitwise** on real held-out windows; fast path versus explicit group sum is within the registered f32 bound; the Rust grouped subset agrees with the Python grouped reference with zero violations. | `results/frontier/exactness-Qwen-Qwen3-1.7B-20261114T220536Z.json`, `results/frontier/cross-check-20261114T220555Z.json`, `results/frontier/rust-b-real-20261114T220539Z.json` |
| H1: some `<=1e9` static group selection preserves the teacher distribution within the registered threshold. | **Rejected.** The best compliant arm is orders of magnitude outside both the acceptable and usable tiers. | `results/frontier/frontier.json`; `results/M1-FRONTIER.md` |
| H1a: calibration statistics beat random at equal sparsity. | **Directionally supported but practically irrelevant.** Magnitude and frequency rank better than random on mean KL at the affordable share; all are catastrophic. | same |
| H1b/K3: non-uniform allocation beats matched-budget uniform. | **Registered test fails.** The exactly matched greedy-swap arm improves by a small amount whose CI includes zero. A near-matched additive greedy improves substantially, but with two extra groups; reported as exploratory, not a K3 pass. | `results/frontier/sweep-greedy-*.json` |
| H1c/K4: the non-uniform gain survives domain shift. | **Partially.** For the additive greedy the gain does not collapse on code or math (CIs exclude zero), retaining roughly half; for the registered swap arm the code CI includes zero. | same |
| The control model behaves qualitatively differently. | **No.** The 0.6B also collapses under static group omission at comparable relative sparsity, but it is dense-compliant and needs no sparsity. | `results/frontier/sweep-control-*.json` |
| Whole-layer omission is a useful coarse fallback. | **Rejected at the compliant budget.** Single-layer omission cost ranges over an order of magnitude by layer; the affordable whole-layer allocations are catastrophic. | `results/frontier/sweep-layers-*.json` |

### Verification performed (receipts, not prose)

| what was verified | evidence |
| --- | --- |
| Pre-registration committed before any sweep | `dad6619`; receipts carry it as `preregistration_commit` |
| Datasets fetched, LFS-object hashes verified | `results/datasets/fetch-20261114T223120Z.json` |
| Split built, near-duplicate check, per-document hashes | `results/splits/split-20261114T223127Z.json`; derived file SHA-256 is content-only and stable across re-derivation |
| Instrument exactness (G1) | `results/frontier/exactness-Qwen-Qwen3-1.7B-20261114T220536Z.json` |
| Independent Rust grouped sum | `results/frontier/rust-b-real-20261114T220539Z.json`; `results/frontier/cross-check-20261114T220555Z.json` |
| Court B unit tests | oracle court receipt in `results/courts/` for `python3 -m pytest oracle/tests` |
| Calibration statistics and selector rankings, both models | `results/frontier/stats-*-20261114T2205*.json` |
| Uniform frontier, 1.7B | `results/frontier/sweep-uniform-Qwen-Qwen3-1.7B-20261114T221355Z.json` |
| Whole-layer omission, 1.7B | `results/frontier/sweep-layers-Qwen-Qwen3-1.7B-20261114T221904Z.json` |
| Greedy allocations, 1.7B | `results/frontier/sweep-greedy-Qwen-Qwen3-1.7B-20261114T222628Z.json` |
| Control, 0.6B | `results/frontier/sweep-control-Qwen-Qwen3-0.6B-20261114T223054Z.json` |
| Bootstrap, thresholds, JSON/SVG/table | `results/frontier/frontier.json`, `results/frontier/frontier.svg`, `results/frontier/frontier.md` |
| Two independent complete runs agree bitwise | clean receipts versus `results/frontier/superseded-dirty-752172b/`; max per-arm KL difference zero |

### Failures and deviations, verbatim

1. The oracle image rebuild with the new `pyarrow` pin failed once under the
   default build limit:
   `ERROR: failed to build: failed to solve: process "/bin/sh -c apt-get update ..." did not complete successfully: exit code: 137`
   with `Killed` in the pip layer. It succeeded with
   `SELVEDGE_BUILD_MEM=12g`. Builds are not courts, so no court receipt exists
   for the failed attempt; the successful image id is in `docker/IMAGES.lock`.
2. The first `exactness` run failed with
   `TypeError: string indices must be integers, not 'str'` in `base_receipt`
   (split counts were treated as lists). The court receipt and log are retained
   (`results/courts/oracle-20261114T213245Z-230e70.json`); the fixed rerun is
   the receipt cited above.
3. A `python3 -m py_compile` attempt in the oracle court failed with
   `[Errno 13] Permission denied: '/pycache'` because the oracle court does not
   mount the Python bytecode cache. It was a syntax check, not a test; imports
   and the program run correctly (bytecode writing is disabled in the image).
4. Amendment A2: the frequency selector's formula and gloss disagreed. The
   degenerate first calibration receipt
   (`results/frontier/stats-Qwen-Qwen3-1.7B-20261114T213138Z.json`) is retained.
   The corrected definition is in the contract; no held-out measurement existed
   at the time of the amendment.
5. A dataset re-fetch from cache initially rewrote the lock's
   `verified_source` to `cache_rehash` and changed the lock hash. The fetcher
   now preserves the previous verified entry, and the lock is byte-identical
   after offline re-verification.
6. No NaN, Inf, OOM or timeout occurred in any sweep; every arm records
   `nonfinite: 0`.

### Decisions and known limitations

1. **The result is a negative result for a family, not for sparsity in
   general.** Static, token-independent, sixteen-group masks with no retraining
   and no routing are bounded. Dynamic, causal, trained or interaction-aware
   selectors are untested.
2. **The teacher is f32-equivalent, not native bf16.** The f32 path is the
   registered reference; native bf16 can differ within the S3a bound.
3. **The greedy oracle is deliberately non-deployable.** It uses teacher KL on
   calibration text and pruned candidate sets. Its receipts say so.
4. **K3 failed as registered.** The additive greedy observation is reported
   separately with its budget caveat. It must not be merged into a K3 pass.
5. **The control sharpens the conclusion.** The 1e9 ceiling makes the 1.7B
   teacher infeasible under this intervention; the 0.6B is already compliant
   and needs no sparsity. A smaller compliant base plus external capacity, or
   retraining, is the remaining path.
6. **No capability, latency or I/O claim is made.** See
   `docs/LIMITATIONS.md`.

### Next experiment

M2, but not as originally scoped: the static-mask frontier shows that uniform
and near-uniform omission cannot buy compliance. The highest-information next
step is a per-token, causal selector at the same realized budget, because the
registered H1a ordering is the only part of the static result that points in a
useful direction, and it is swamped by the collapse. Before training a
predictor, measure the oracle per-token potential: for a small held-out sample,
compute the best achievable KL for each token with the same per-token group
budget using exhaustive or bounded search. If the oracle per-token potential is
also catastrophically far from the teacher, no selector can rescue this teacher
at this granularity, and the retraining/upcycling branch becomes the primary
path. Either outcome is informative and should be pre-registered next.

## 2026-11-14 — M2: per-token causal selection, oracle potential and the Rust Atlas crate

> **Later correction (2026-11-15, S5b).** The `deployable` flag used in this
> entry is superseded by the two typed verdicts; the M2 instrument reported
> logical selection counts, not physical reads. The oracle bound is retained as
> a logical-omission counterfactual. See `docs/M2-CORRECTION.md` and the S5b
> entry below.

Revisions: `a0b107adc23f23cbda43a09cbd6d229b0ea024a1` (M2
pre-registration, committed before measurement), `1e623062379239a4bed3337ed2ed91d1a7f5411b`
(the selector harness, oracle search, Rust Atlas crate and Court B tests used
for the clean measurements), `362d2c7156feff66d55a04d84e2aff548c48e95f`
(the reporting revision that emits the generated tables and the fixtures), and
`40f19dbf62de8395cd55ebf27f5685d71769347c` (the test commit; the oracle and
analysis were regenerated from it to add the cross-window stability field).
The prepare, evaluation and interaction receipts name `1e62306`; the later
revisions add reporting code and tests, not measurement logic. The development
run from the uncommitted worktree is retained under
`results/selectors/superseded-dirty-a0b107a/` and is numerically identical on
every deterministic arm.

The report is `results/M2-SELECTORS.md`; the registered contract is the M2
section of `docs/EXPERIMENT-CONTRACT.md`, including amendment A3. No measured
number is repeated here.

### Plan

Pre-register the per-token question, the oracle protocol and the
matched-budget arms; build an exact per-token masked-MLP instrument; measure
the oracle-potential ceiling and the 16/64/4-group granularity comparison
before training any predictor; then measure every deployable arm at matched
realized active parameters on the held-out split and the OOD slices. Build the
Rust Atlas boundary in parallel so the selector decisions, the interaction
graph and the realized-budget accounting are executable, not prose.

### What was built

- `docs/EXPERIMENT-CONTRACT.md` M2 section: budgets B140/B142/G, oracle
  protocol, granularity study, interaction protocol, learned-selector
  protocol, leakage rule, metrics, sample sizes, seeds, CIs, K6-K9, and the
  explicit continuation criterion. Amendment A3 records the unattainable
  exact-element match for a learned controller.
- `oracle/masked_mlp.py`: per-token masks, hidden-state capture, exact
  per-group contributions, the Gram-matrix local error identity, and the
  gradient-capture flag. The M1 paths and their exactness are unchanged.
- `oracle/m2_selectors.py`: calibration per-document statistics; exact
  intervention labels; ridge-fitted learned unary and interaction models;
  `prepare`, `interactions`, `oracle`, `evaluate`, `analyze` and `fixtures`
  subcommands. Every subcommand writes a receipt with stamp, revision,
  pre-registration commit, environment, data hashes, per-item metrics and
  non-finite accounting.
- `crates/selvedge-atlas`: `QualityObservation`, `InteractionGraph` with typed
  invariants, the leakage-bounded `SelectorInput`, the `Selector` trait and
  seven selectors, ridge `fit`, independent `compare` with a Rust paired
  bootstrap, reliability/calibration helpers, and the two-view
  `value_if_executed` / `cost_to_materialize_now` contract with an explicit
  unavailable cost head.
- `selvedge atlas fit|select|compare|cost`: CLI subcommands with `--json`.
  `select --check-expected` fails closed on a parity mismatch.
- `crates/selvedge-atlas/tests/court_b.rs`: parity on the exported real
  fixtures, determinism, graph invariants, no-leakage deserialization, fit
  parity, calibration monotonicity, matched-budget accounting and the two-view
  contract.
- `results/M2-SELECTORS.md`, the generated table/plot/JSON under
  `results/selectors/`, and the updated architecture, limitations and report.

### Hypotheses and results

| hypothesis | result | evidence |
| --- | --- | --- |
| H2.1/K7: a per-token oracle at the compliant budget reaches the usable tier. | **Rejected, decisively.** The best oracle (64 groups at matched elements) is far outside the KL, PPL-ratio and top-1 bounds. | `results/selectors/oracle-20261114T231601Z.json`; `results/M2-SELECTORS.md` |
| H2.2/K8: finer fragments at matched elements rescue the tier. | **Partially: granularity matters but is not binding.** 64 groups is better than 16; 4 is worse; 64 is still far outside the usable tier. | same |
| H2.3/K6: a deployable per-token selector beats activation magnitude by 5% at matched budget. | **Rejected.** Every matched deployable arm is worse than `token-magnitude`; the best is about 5% worse, the learned arms about 57% worse. | `results/selectors/eval-full-20261114T232206Z.json`; `results/selectors/selectors.json` |
| H2.4: the learned interaction-aware predictor beats the unary predictor. | **Rejected.** The relative change is negative with a CI that includes zero. | `results/selectors/selectors.json` `/learned_comparison` |
| H2.5: a low-rank/graph surrogate predicts held-out pairwise interaction. | **Rejected.** The leave-one-window-out Spearman is negative; the co-activation/mean-norm surrogate anti-predicts. | `results/selectors/interactions-20261114T232338Z.json` |
| The per-token direction is at least positive. | **Supported but insufficient.** Per-token magnitude beats the best static M1 arm at the same B142 budget; it remains two orders of magnitude outside the acceptable tier. | `results/selectors/eval-full-20261114T232206Z.json` |
| The Rust selectors reproduce the Python arms. | **Confirmed exactly** on the exported real fixtures for all seven selectors and on the fitted learned models. | `cargo test -p selvedge-atlas`; `results/courts/rust-20261114T232455Z-7dbe87.json` |
| The Rust accounting reproduces the Python matched-budget table. | **Confirmed.** Thirteen arms, twelve paired comparisons, the same relative directions and CIs. | `results/selectors/atlas-compare.json`; the `atlas compare` court receipt |

### Verification performed (receipts, not prose)

| what was verified | evidence |
| --- | --- |
| Pre-registration committed before any M2 measurement | `a0b107a`; every receipt carries `preregistration_commit` |
| Clean harness revision | `1e62306`; every clean receipt carries it |
| Prepare: calibration statistics, labels, fitted weights, fit data | `results/selectors/prepare-Qwen-Qwen3-1.7B-20261114T231248Z.json` |
| Oracle potential, three passes, four granularities, bounded refinement | `results/selectors/oracle-20261114T231601Z.json` |
| Full held-out selector table | `results/selectors/eval-full-20261114T232206Z.json` |
| Oracle-sample deployable comparison | `results/selectors/eval-oracle-20261114T232422Z.json` |
| Pairwise interaction and surrogate | `results/selectors/interactions-20261114T232338Z.json` |
| Bootstrap, table, plot, JSON | `results/selectors/selectors.json`, `table.md`, `selectors.svg` |
| Rust parity fixtures | `results/selectors/fixtures/selector-inputs.json` |
| Rust fit / cost / compare | `results/selectors/atlas-fit-weights.json`, `rust-selector-cost.json`, `atlas-compare.json` |
| Rust fmt / clippy / test | the `rust` court receipts in `results/courts/` |
| Python M2 helper tests (24 passed) | `results/courts/oracle-20261114T233225Z-fe70cb.json` |
| Generated numbers match their receipts | the `perf_numbers` court receipt |
| Every scanned commit carries the operator identity | the `commit_identity` court receipt |

### Failures and deviations, verbatim

1. Amendment A3: exact element equality is impossible for a learned controller
   at whole-group granularity. The learned arms are 868 elements above the
   baseline; the rule and the direction are recorded. No threshold changed.
2. The 64-group oracle uses bounded greedy (beam width 1), not a wider beam;
   exact `C(64,20)` enumeration is infeasible. Each greedy step is exact under
   the incremental local objective.
3. The final refinement is a 2-window, 32-position, one-pass probe, not the
   full eight-window, two-pass search.
4. A development `fixtures` run failed with a CPU/CUDA device mismatch when
   constructing the interaction weights; the fix moves the weight tensors to
   the feature device. The failed court receipt and log are retained.
5. A development `analyze` run failed with `KeyError: 'positions'` because the
   final-refinement window record omitted the field; the analysis now derives
   it from `nll_positions` and the oracle writes it explicitly. The failed
   receipt is retained.
6. The gradient arms are not bitwise reproducible in the last printed digit
   across runs; the effect is far below any conclusion. The deterministic arms
   are bitwise identical between the clean and development runs.
7. No NaN, Inf, OOM or timeout occurred in the clean M2 runs; every evaluated
   arm records `nonfinite: 0`.

### Decisions and known limitations

1. **The inference-only post-hoc selection branch is dead for this teacher.**
   The oracle bound, not the deployable selector ranking, is the decisive
   evidence. The next branch is training or upcycling: teach the model to work
   with the sparse FFN, or move to a compliant dense base plus external
   capacity.
2. **Per-token magnitude is the strongest implementable baseline in this
   family.** Future selectors must beat it, not the static M1 arm.
3. **Granularity is real but insufficient.** A fine-grained deployable selector
   is worth one more experiment only if a trained branch needs it; it is not a
   path to compliance on the dense checkpoint.
4. **The interaction-aware hypothesis is not supported at group granularity.**
   The pairwise term is context-dependent, and the tested graph surrogate does
   not predict it. The Atlas graph remains the right container for a richer
   measurement, not a demonstrated selector.
5. **The learned label does not transfer.** A one-layer local intervention
   label trains a model that predicts the local objective but not final-KL
   quality. Any future learned selector needs a final-KL or multi-layer label.
6. **The Rust boundary is real.** The Atlas crate, CLI, accounting, leakage
   boundary and parity tests are implemented and pass; the cost head is
   explicitly unfilled until M3.

### Next experiment

The trained/upcycled branch, with a fresh pre-registration: compare (i) a dense
compliant Qwen3-0.6B base plus external capacity, (ii) an UpIT/SIMoE-style
dense-to-sparse upcycle of Qwen3-1.7B with the FFN trained to tolerate the
compliant group budget, and (iii) a distilled smaller teacher, all matched for
active parameters, FLOPs and training tokens. The M2 oracle bound says a
post-hoc selector cannot close the gap; the next test is whether training can
change the model so the same budget is usable. Keep the Atlas graph and the
two-view head for the eventual scheduler, and do not resume inference-only
selector tuning without a new oracle bound.

## 2026-11-15 — S5b: exactness boundary, dual verdicts, measured accounting, one-pass selection

### What was built

- **Defect 1 — exactness boundary.** `crates/selvedge-core/src/selection.rs` no
  longer has `reclassify_from_approximate` or `NativeSetAttestation`. The only
  path to `ExactSelection` is `from_witness`, and the only source of a
  `NativeSetWitness` is `NativeSetWitness::compare` against a bundle-declared
  `NativeComputationSet` (bundle id, full set equality, declared and executed
  digests). `SemanticExactness::classify` records `Proven` or `Approximate`
  with a typed `SemanticGap`. `ExecutionScope` is documented as scope integrity
  only and cannot be converted. `crates/selvedge-core/tests/ui/` plus
  `tests/compile_fail.rs` prove, by compiling five forbidden programs with
  `rustc`, that the removed constructor does not exist, that a witness cannot
  be forged, that the real store scope is not a witness, that a boolean cannot
  be a physical verdict, and that an exact-labelled decision cannot be minted
  from outside the crate.
- **Defect 2 — two verdicts.** `crates/selvedge-core/src/verdict.rs` defines
  `CausalAvailability` and `PhysicalCompliance`, each evidence-bearing with
  named failing reasons. A compliant physical verdict requires measured reads.
  The Python harness emits both per arm; the `deployable` field is gone from
  `oracle/m2_selectors.py`, its receipts, `results/selectors/table.md` and the
  generated M2 table.
- **Defect 3 — measured accounting.** `masked_mlp.sparse_grouped_forward` slices
  gate/up rows and down columns before the matmul, groups identical per-token
  subsets, and counts the exact sliced elements. `audit_sparse_reads` cross-checks
  the counter against `torch.profiler(record_shapes=True)`; five real layers
  matched exactly (`results/selectors/sparse-audit-20261115T002900Z.json`).
  `ParameterLedger::admit_selection_path` checks the union of selector and
  execution uses before execution and is the only source of
  `AdmittedSelectionPath`.
- **Defect 4 — one-pass selection.** `online_one_pass_select` selects inside a
  forward pre-hook from the state the layer receives under its own earlier
  masks. It is compared with the two-pass arm on the oracle sample.
- **Storage WIP finished.** `prefetch_bytes_wasted = read - used` is validated
  with unit tests and Court D tests (all-used, partial, saturating,
  serialised, never-delivered).
- **Corrected M2 measurement.** Full held-out re-evaluation at
  `results/selectors/eval-full-20261115T005000Z.json`; corrected table and
  analysis regenerated. Six of 23 arms are physically compliant (static/random
  masks plus `token-random`, all with unusable quality). The oracle bound
  survives as a logical-omission counterfactual. `docs/M2-CORRECTION.md` is the
  correction of record.

### Measured findings

- The sparse path reproduces the masked-dense KL to the printed precision for
  every arm (for example both magnitude variants at KL 6.676772).
- `token-magnitude`'s selector reads 225,485,783,040 parameter elements over the
  160 evaluation windows, and the per-token union of selector plus execution is
  above the ceiling (see the generated block in `docs/M2-CORRECTION.md`). It
  was never physically compliant.
- The one-pass arm is better than the two-pass arm at both B140 and B142; the
  paired difference excludes zero (`docs/M2-CORRECTION.md`). The earlier M2
  numbers therefore depended on teacher-state iteration, and the dependence was
  unfavourable.
- `token-task-static-nn` was found to read whole-window shingles; it is now
  `causal: unavailable, future_tokens`.

### Limitations

- The six compliant arms are not learned and not useful.
- The one-pass comparison is eight windows, one seed, one teacher.
- No exact neural-execution runtime consumes `ExactSelection` yet.
- Raw JSONL metrics from the storage bench are kept out of git; summaries and
  the regenerating commands are tracked (see `docs/STORAGE.md`).

### Next experiment

The trained/upcycled branch, as stated at the end of the M2 entry. Do not
resume inference-only selector tuning without a new oracle bound and without a
physically compliant controller whose selection reads fit the budget headroom.

### Second-session verification pass (2026-11-15, after the WIP checkpoint)

The S5b work above was audited against the mandates rather than trusted. The
Rust fmt/clippy/workspace-test courts, the oracle Court B tests, and the
`perf_numbers` and `commit_identity` courts all passed at HEAD before any edit.
The audit then found two accounting defects and one weak barrier, all fixed:

- **One-pass receipt accounting.** In `onepass-20261115T012700Z.json` the
  online-random row reported a physical failing reason of
  `selection_reads_outside_active_set` with detail about gate/up reads even
  though its measured `selector_elements_read` was zero, and its
  `selector_seconds` (a naive scalar splitmix generator) exceeded its forward
  time because the online rows nested selection inside the timed forward while
  the other rows did not. The reason is now derived from the row's own evidence
  by `onepass_physical_verdict`, and every row carries `selector_seconds`,
  `execution_seconds` and `total_seconds` with the convention stated. The
  measurement was re-run; the KL values reproduce exactly, the corrected receipt
  is `results/selectors/onepass-20261115T044200Z.json`, and the old receipt is
  retained. Two Court B tests pin the reason mapping and the table helper.
- **Logical versus physical counts in the generated table.** The main M2 table
  still had one column named `active unique` holding the logical allocation whose
  misreading the correction exists to prevent. The generator now prints
  `logical active unique` and `physical unique` (the selector-plus-execution
  union). The token-magnitude family is visibly over the ceiling in the second
  column. The analysis was regenerated (`selectors.json`, `table.md`, the table
  in `results/M2-SELECTORS.md`) against the same measurement receipt.
- **Scope compile-fail case used a local lookalike.** The barrier now imports the
  real `selvedge_store::ExecutionScope` (added as a dev-dependency, which does not
  create a normal-dependency cycle) and fails to compile when it is passed to
  `ExactSelection::from_witness`.

Documentation was corrected where it had drifted: `results/REPORT.md` had
claimed that no latency or throughput had been measured while
`docs/STORAGE.md` reports measured NVMe/host-RAM baselines; the report now
distinguishes storage baselines from model speed and hardware-counter FLOPs.
The two residual trust boundaries the types cannot close (the honesty of a
native-set declaration, and the receipt behind a `Measured` label) are now
stated in `docs/ARCHITECTURE.md` and `docs/LIMITATIONS.md`. The remaining
`non-deployable` prose in the oracle helpers was replaced with the causal and
physical vocabulary.

A fresh `io` court smoke sweep ran at this HEAD as a reproducibility check
(`results/courts/io-20261115T043846Z-978215.json`); it is not a new benchmark
and no storage number in `docs/STORAGE.md` was changed by it. One attempt to
re-run the one-pass measurement at the default court memory limit was killed
with exit 137 (out of memory); the receipt is retained as
`results/courts/oracle-20261115T043608Z-0848fe.json` and the run was repeated
with an explicit, still-enforced larger limit.

## 2026-11-15 — S7a: program plan, prior-art ledger and the T1 infrastructure court

### What was built

- **`docs/PROGRAM-PLAN.md`** — the exhaustive TODO for the next phase, 36
  numbered items across six workstreams (P1.1–P6.5). Each item has what, why,
  dependencies, acceptance evidence, an honest status and an effort estimate.
  It includes the verified capacity-scaling ladder and a `NOW` section naming
  the single next experiment.
- **`docs/PRIOR-ART-AND-NOVELTY.md`** — the ledger the mandate requires. 53
  named works plus 4 found during verification, each with URL, retrieval date,
  status (`VERIFIED`/`PARTIAL`/`UNSUPPORTED`/`PROPOSED`) and the fetched
  evidence, plus a feature-overlap matrix and the plain statement of the target
  claim.
- **`scripts/prior_art_fetch.py`** — the reproducible fetch instrument. It
  records HTTP status, final URL, retrieval time, title and abstract for every
  source into `results/prior-art/*.jsonl`, so a status can be checked against
  bytes rather than trusted.
- **`docs/TRAINING.md`** — the training path, the registered objective, the
  numerical tail floor and the determination table for the five recorded test
  failures. `oracle/train/loop.py` already referenced this file; it did not
  exist.
- **`oracle/tests/test_t1_infrastructure.py`** — the five recorded failures are
  resolved with reasons, and the replacement tests are stricter than the ones
  they replace.

### The five failures, determined rather than silenced

The recorded log (`results/courts/logs/oracle-20261115T070709Z-262951.log` at
revision `56c648f`) was **stale by the time of the WIP commit**: HEAD's test
file already carried corrected expectations and the suite passed. That is not a
resolution on its own, so each case was re-derived:

- `document_text` formatted before checking for a missing field → real
  implementation defect, fixed.
- the mirror test expected 460 where the fixture's union is 310 → wrong test.
- the pack/unpack test compared NaN-bearing bf16 bit patterns by value → wrong
  test; now bit-exact and length-checked.
- the adapter zero-init test perturbed a leaf parameter outside `no_grad` →
  wrong test.
- the distillation-loss test expected the **full-vocabulary** cross-entropy,
  while the registered T1.4 target is top-16 support plus one tail bucket →
  wrong expectation. The implementation is the registered objective. The new
  tests prove that with an independent float64 transcription, support-order
  invariance, and the decisive case: at full support the objective equals the
  ordinary cross-entropy to `1e-9` with a finite gradient.

### Verified courts at HEAD

- `scripts/court rust test` — exit 0
  (`results/courts/rust-20261115T072334Z-2d1680.json`).
- `scripts/court oracle -- python3 -m pytest oracle/tests -q` — the full oracle
  suite passed.
- `scripts/court oracle -- python3 -m pytest oracle/tests/test_t1_infrastructure.py -q`
  — the infrastructure suite passed after the edits
  (`results/courts/oracle-20261115T073420Z-287ce4.json`).
- `scripts/court commit_identity` — OK, the scanned commit carries the operator
  identity.
- `scripts/court perf_numbers` — OK, every declared block verified.

### Findings that change the plan

- **SpecPrefetch already publishes P3.7's design**: a separate lightweight
  adapter predicts next-layer experts while the frozen router decides
  execution. The program's contribution cannot be predictor/router separation.
- **SpAx already publishes P4.3's three-way decision** (retain / approximate /
  omit), so only predicted-error tier assignment is open.
- **Temporally Extended MoE publishes a stronger version of the stickiness
  trade-off** than the program's own 2026 negative result: switch rates from
  over 50% to below 5% while retaining up to 90% of base accuracy. P3.7 must
  cite it and justify the no-stickiness rule rather than claim novelty.
- **SpecMD reports that expert access is not temporally local**, which
  undercuts any locality premise in prefetch or cache design.
- **`HOPE` and `NorSA` could not be verified** as papers with the claimed
  mechanisms; `Engram`, `D²-MoE` and `CMoE` are known only through citing
  works. These are marked `UNSUPPORTED` / `PARTIAL` instead of cited.
- **The Arm B sizing defect is confirmed by arithmetic**: the whole bank is
  352,321,536 elements and the whole model 948,600,832, under the ceiling, so
  Arm B is a compute saver and not a capacity-extraction experiment (P2.1).

### Limitations

- No training run was started; the T1 arms have no measured result.
- The capacity ladder in `docs/PROGRAM-PLAN.md` is arithmetic over registered
  constants, not an admission receipt; P2.2 exists to turn it into one.
- The prior-art ledger cannot cover closed venues or non-acronym titles, so
  `UNSUPPORTED` means "not found by the recorded searches".
- The fetch instrument records abstracts and page descriptions, not full texts;
  a full-text read could narrow the overlap claims.

### Next experiment

The five-arm pilot of P3.4, after the P2.4 teacher-target instrument check: it is
the only experiment whose outcome decides whether the trained-capacity branch
continues, and every later workstream assumes its answer.

## 2026-11-15 — S8a: T2 registration, capacity modules, P2.4 instrument and the five-arm smoke

### Plan

Deliver the five-arm error-directed matrix (P3.4) as a registered, instrumented,
smoke-proven pilot without starting a full run:

1. **Register T2** before any measured comparison: five arms, the memory and
   expert paths, the utility signal, the warm-up, the budget rule, the seeds,
   the metrics and K15-K19.
2. **Implement** the memory path, the four capacity wrappers, the utility
   labeller and the two router treatments, and route every arm through the Rust
   admission ledger.
3. **Run the P2.4 instrument check** against a live full-vocabulary teacher and
   register the decision.
4. **Smoke every arm**: step-0 parity, 50 steps, determinism, resume, VRAM and
   throughput, then derive the budget and leave exact launch commands.

### Hypothesis and result

| # | hypothesis | result | evidence |
| ---: | --- | --- | --- |
| H1 | A deterministic hash over the token prefix can address a 2^18-row per-layer table with no learned index, and the active union is only the retrieved rows plus the projection and gate | confirmed at the module level: 525,824 active elements per insertion layer, 2,103,296 over four; every read is causal in a unit test | `oracle/tests/test_t2_capacity.py`, `results/train/admission/t2/M.json` |
| H2 | The Rust ledger can refuse an over-budget T2 arm before training | confirmed for all five arms: each config is admitted at its design count and each mutated config is refused with no admitted path | `crates/selvedge-cli/tests/t2_admission.rs`, `results/courts/rust-*.json` |
| H3 | A zero-initialised memory projection and specialist `W_down` give exact step-0 parity for every branch (`D`, `M`, `X`, `H`, `H*`) | confirmed, but only after a defect was found: an fp32 extra added to a bf16 base promoted the residual stream and moved the logits far outside the `1e-3` tolerance. Casting the extra to the base dtype makes the delta exactly 0 on all five arms | `zero-point.json` in each smoke run |
| H4 | The compact T1.4 target ranks students like the full-vocabulary teacher on the held-out sample | the bias is real and near constant (the number is below); both statistically distinguishable decisive pairs keep their sign; `H*-H` is inside its CI under both targets | `results/train/instrument-20261115T130334Z.json` |
| H5 | The five arms fit 16 GiB and train at a rate that makes a 5-hour budget meaningful | confirmed: the slowest arm stays under the 16 GiB device, the declared budget fits every arm, and the five arms fit the 25 GPU-hour envelope | `results/train/budget-20261115T130456Z.json` | |
| H6 | T2 execution can be bitwise deterministic, so the pilot's noise floor is not dominated by kernel atomics | confirmed for all five arms only with `torch.use_deterministic_algorithms(True)`; without it even the dense control is not bitwise reproducible | `results/train/smoke-20261115T10*.json`, `...T114908Z.json`, `...T125618Z.json` |

### The measured numbers behind H4 and H5

Compact-target bias on the held-out sample (nats/token, `D0`):

<!-- BEGIN GENERATED: worklog-t2-instrument-bias -->
-0.400
<!-- END GENERATED: worklog-t2-instrument-bias -->

Slowest arm throughput (tokens/s, `H`):

<!-- BEGIN GENERATED: worklog-t2-slowest-tokens-per-second -->
4767.47
<!-- END GENERATED: worklog-t2-slowest-tokens-per-second -->

Declared budget (tokens):

<!-- BEGIN GENERATED: worklog-t2-budget-tokens -->
67108864
<!-- END GENERATED: worklog-t2-budget-tokens -->

Five arms at the declared budget (GPU-hours):

<!-- BEGIN GENERATED: worklog-t2-total-gpu-hours -->
15.68
<!-- END GENERATED: worklog-t2-total-gpu-hours -->

### The P2.4 finding, stated plainly

The T1.4 objective is the top-16 support plus **one tail bucket**. On every
held-out window with the live 1.7B teacher, the teacher keeps a small but real
tail mass, the tail has its own entropy, and the student's tail differs from it;
the measured bias, tail mass, entropy and divergence are the generated numbers
above and in contract A9. The compact loss is therefore *below* the full loss,
and the identity `L_compact - L_full = -p_tail (H + KL)` holds to six decimals
on the pooled valid positions. The bias is almost identical across the
six students, which is why the ranking survives: both distinguishable decisive
pairs keep their sign, and the one that flips (`H*-H`) is inside its CI under
both targets. The decision is to keep the compact target for training and
report both targets at evaluation; the rule that produced it was written after
the measurement and is labelled exploratory, while the both-targets safeguard is
unconditional. A hard-label control `HL` is registered as a deferred option.

### Two defects found and fixed

- **Residual dtype promotion.** `AdditiveFFN` added an fp32 projection output to
  a bf16 base, promoting the rest of the network. The step-0 identity check
  caught it (delta 3.64 on `X`, `H`, `H*`, exactly 0 on `D`, `M`), and the fix
  also halved activation memory.
- **Memory-fit OOM on the hybrids.** The hybrids exceeded 16 GiB. Storing the
  memory table in bf16 (its deployment dtype) brought them to 10.85 GiB without
  changing any count, and a resume that loaded its checkpoint straight to the
  GPU was the other 2.5 GiB.

### Verified courts

- `scripts/court oracle -- python3 -m pytest oracle/tests -q` passes, including
  the new T2 tests.
- `scripts/court rust test` passes, including the two new T2 admission tests in
  `crates/selvedge-cli/tests/t2_admission.rs`.
- `scripts/court commit_identity` and `scripts/court perf_numbers` — see the
  closing receipts at this revision.

### Limitations

- No full-budget T2 run has started. The smoke is 50 steps per arm.
- The utility labels are measured once in the full-bank reference set, so an
  expert that duplicates another is under-penalised.
- The `H*` router is frozen after its label phase; periodic relabelling is
  deferred to P3.5 phase B.
- The one-seed pilot cannot separate a small `H*`-`H` difference unless the
  replication runs; the arithmetic fits the envelope, and A8 records the
  fallback.

### Next experiment

Launch the five arms at 2^26 tokens with the commands in `results/T2-PLAN.md`,
then evaluate with `oracle/eval/baseline.py` and the both-target instrument
check. K15-K19 decide whether WS3 is scaled or retired.

## 2026-11-15 — S8b: the T2 pilot, launched detached

### Plan

Run the pre-registered five-arm matrix at the A8 budget, evaluate every arm on
completion with the registered harness, and apply the T2.9/T2.10 decision rule.
The pilot is a long GPU job, so the plan has two halves:

1. make a run survive a lost session, and make every step leave a receipt;
2. only then spend the GPU time.

### What changed in the courts

- `scripts/docker-court.sh` gains `--detach` and a `wait` court. The detached
  launch uses `docker run -d` with *exactly* the same limits, mounts, GPU mode,
  network policy and environment as the foreground path — the receipt argument
  list is built once and shared — and `wait` blocks on the container, captures
  its log, inspects the applied limits and writes the same receipt. A dead
  client therefore no longer loses a run, and a detached run cannot drift from a
  supervised one.
- A detached run does not use `--rm`: the container is inspected and removed by
  `wait`, so the enforcement comparison still runs. A container that has
  disappeared is a hard error, not a missing receipt.
- The wait path takes the container's exit code from `docker wait`'s *stdout*
  (`docker wait` itself exits 0 when it successfully waited); a non-numeric
  answer fails closed. This was found by launching a container that exits 7 and
  observing the wait report success.
- `scripts/run-t2-pilot.sh` orchestrates: launch guard, shared target cache,
  the five arms strictly sequentially, evaluation on completion, the P2.4 check
  and the decision. `scripts/poll-t2-pilot.sh` reports progress, ETA and what
  remains. `scripts/t2_pilot.py` does the bookkeeping and never touches a model.

### Disk, honestly

The repository filesystem could not hold the pilot next to the recorded
50-step smoke checkpoints: the guard's arithmetic (in
`results/t2/launch-guard.json` at the launch stamp and revision) put the target
cache, the retained checkpoint pairs and the H\* utility checkpoint well beyond
the free space. The smoke checkpoints at `results/artifacts/train/smoke-t2/`
were **moved, not deleted**, to a separate filesystem and a symlink keeps the
paths resolving; the numbers behind that run are in the smoke receipts, which
are tracked. The same was done, while the pilot ran, to the remaining 50-step
smoke trees and the H\* route probe (`results/artifacts/train/smoke` and
`.../probe-hstar`): each was copied, byte-size and file-count checked, then
replaced by a symlink to the archive. Nothing that a receipt cites was removed,
and the guard runs before every launch and refuses rather than filling the
disk.

### Defects found while exercising the receipts

Three defects were found by running the real receipt paths against the recorded
smoke artifacts, before any GPU time was spent on them:

- the admission record's `total_flops` is an object of three components, not a
  scalar, so `flops * tokens` would have raised on the first arm to finish;
- `loop.save_checkpoint` writes a JSON sidecar, so a bare first-token parse
  produced `{` as a checkpoint hash;
- the eval summary lacked the `revision` field the perf-numbers court requires.

All three are fixed in `scripts/t2_pilot.py`.

### A defect found and not silently fixed

`results/T2-PLAN.md` §2 tabulates "expert weight bytes (+router)" for `X`, `H`
and `H*` as `352,550,912`. That number is the specialist bank's **element
count**, not its size in bytes, and it counts all eight specialists rather than
the four the router reads: at bf16 the deployed k=4 read is `352,780,288` bytes
and the whole bank is `705,101,824`. The cell is hand-written, so the
perf-numbers court never saw it. It is not edited here, because a registered
plan is not rewritten after a measurement; the generated figures in
`results/T2-PILOT.md` are the ones to use, and the table is corrected in a
separate, dated change that cites the decision receipt.

### What is running

`scripts/run-t2-pilot.sh` was launched with `setsid nohup` at revision
`1dd7270f3959a3e0ce15f670a03997467235d04b`. It records every transition in
`results/t2/pilot-state.json`. Progress and the resume command are printed by
`scripts/poll-t2-pilot.sh`.

## 2026-11-15 — S9: the next-phase plan, the literature second pass, and a read-only pilot record

### Plan

The T2 pilot owns the GPU in this session, so nothing here may touch it. The
session delivers three documents and one tool:

1. extend `docs/PROGRAM-PLAN.md` with the workstreams that follow T2, without
   renumbering or deleting anything already registered;
2. verify every work those workstreams newly name, in the same way the earlier
   ledger was built: a real fetch, a recorded URL, a retrieval time, and a
   verdict that the fetched bytes support;
3. record the running pilot's state read-only, so the next session does not have
   to reconstruct it from container state;
4. keep the fetch tool honest about how it failed.

### What was added to the plan

WS7 (design doctrine), WS8 (complementarity and two-timescale organisation),
WS9 (deficit-directed capability construction), WS10 (eighteen candidate
mechanisms, each with its own cheap decisive experiment and kill criterion),
WS11 (twenty-one original proposals with decisive experiments), WS12 (staged
program and frontier claim), a dependency-ordered queue and a NOW that names no
GPU work. Every new item carries what / why / depends / acceptance evidence /
status / effort, and every mechanism item whose novelty is contested names the
ledger entry that contests it. Existing IDs are untouched; the new sections say
so in their own preamble.

The plan's first post-pilot step is deliberately not a training run. It is the
routed-overflow validity-mask audit and the two H-star target diagnostics,
because either can invalidate a scale-up and neither needs the GPU.

### What the literature pass changed

The Atom API answered HTTP 429 for most requests in the early rounds. That is
recorded in the receipts rather than hidden, and the tool gained two fallbacks
that read `arxiv.org/abs/<id>` and `arxiv.org/search` directly, so a claim is
still backed by a fetched page.

The consequential outcomes are in `docs/PRIOR-ART-AND-NOVELTY.md` section ten
and eleven. Summarised: a set of named works could not be located at all and are
`UNSUPPORTED`; one citation is right about the mechanism and wrong about the
paper; three names are method names rather than paper titles; and the published
analysis of manifold-constrained hyper-connections finds a low *effective*
stream count, which forbids reading that mechanism as a capacity win.

### Pilot, recorded read-only

`scripts/poll-t2-pilot.sh` was used for every reading. The dense control and the
memory arm have both training receipts and evaluation summaries; the routed
specialist arm has a training receipt and was evaluating; the two hybrids had not
started. No arm was interpreted, no container was touched, and no GPU work was
performed in this session. The per-arm state table and the "what still has to
happen" list are in `results/REPORT.md`.

### Verified courts at this revision

- `commit_identity` — OK.
- `perf_numbers` — OK, every generated region still matches its receipt after the
  documentation edits.
- the `python --net` prior-art court, one receipt per fetch round, listed in the
  ledger's method section.

### Limitations

- The ledger still does not prove absence. `UNSUPPORTED` means "not located by
  the recorded searches", and the searches are listed so a reader can repeat or
  extend them.
- The pilot's partial state is a progress record, not a result.
- The plan is a plan: no new item's acceptance evidence exists yet.

### Next experiment

None started here. The next GPU job is whatever `scripts/run-t2-pilot.sh` does
next; the next *program* step is the audit and diagnostics named above.

## 2026-11-16 — S10: the routed-overflow audit, the H*-target diagnostics and the P8 harness

The detached T2 pilot owned the GPU for this session (`X` evaluated, `H`
training, `H*` pending at the time of writing), so nothing here used the GPU,
started a container or touched the pilot. Three pieces of pre-scaling work
landed, all CPU-only and all receipted.

### P12.1 — the validity mask is applied to the wrong pairs

**The finding.** `_RoutedDispatch.dispatch` builds `valid` in capacity-sorted
pair order; `combine` restores the pair outputs to token-major order and *then*
multiplies by `valid`. Token-major pair `p` is therefore gated by the validity
of the pair whose sorted position is `p`, which is a different pair whenever an
expert overflowed. Two consequences, both reproduced: a pair that was executed
is silently dropped, and an overflowing pair is retained and reads a clamped
buffer row that can hold another token's activation. The module docstring
claimed the opposite, which is why the defect survived the T2 review.

**How it was established, not asserted.**

1. `oracle/train/reference_dispatch.py` transcribes the documented semantics
   independently (explicit pair loops, explicit buffer rows, no scatter, no
   reshape-and-reduce) and can model both mask placements.
2. `oracle/tests/test_p12_1_dispatch.py` checks the shipped code against it: the
   dispatch halves agree bitwise, `legacy` reproduces the sorted-order placement
   exactly, the opt-in `pair` placement reproduces the documented semantics
   exactly, and the case matrix (ties, duplicate expert ids, `k = 1`, `k = m`,
   one token, zero tokens, `k = 0`, capacity factors below one) agrees in both.
   The headline toy case asserts that an overflowing pair receives exactly
   `weight * E_3(hidden[other token])` under the shipped placement.
3. `oracle/eval/dispatch_audit.py` measures the incidence in the finished routed
   arm from its own training log. Overflow is routine. Sampled (forward, layer)
   pairs that overflowed:

<!-- BEGIN GENERATED: worklog-p121-overflow-samples -->
555
<!-- END GENERATED: worklog-p121-overflow-samples -->

   out of

<!-- BEGIN GENERATED: worklog-p121-layer-samples -->
1428
<!-- END GENERATED: worklog-p121-layer-samples -->

   sampled. Dropped fraction of all pairs:

<!-- BEGIN GENERATED: worklog-p121-dropped-percent -->
0.896%
<!-- END GENERATED: worklog-p121-dropped-percent -->

   The worst sampled layer lost, at training step

<!-- BEGIN GENERATED: worklog-p121-worst-step -->
660
<!-- END GENERATED: worklog-p121-worst-step -->

   this fraction of its pairs:

<!-- BEGIN GENERATED: worklog-p121-worst-percent -->
24.95%
<!-- END GENERATED: worklog-p121-worst-percent -->

   The mis-assignment counters (a *model* of the permutation, labelled as such)
   give this fraction of pairs, consistent with the analytic `2p(1-p)`:

<!-- BEGIN GENERATED: worklog-p121-misassigned-percent -->
1.653%
<!-- END GENERATED: worklog-p121-misassigned-percent -->


**The correction.** `mask_order ∈ {"legacy", "pair"}`, default `legacy`, plus an
exposed `capacity_factor` so tests can reach overflow deterministically instead
of patching a module constant. `build_arm`, `loop.py --mask-order` and
`baseline.py --mask-order` expose it, and every run receipt records it. The
registered arms, thresholds and comparison are untouched; re-training under the
corrected placement would be a different arm, so it is registered separately as
P12.1a (evaluation-only re-score of `X` first, then the T2.8 replication tier).

**What is affected.** `X`'s training and evaluation receipts, and `H`/`H*` as
they finish. `D` and `M` contain no routed dispatch. Single-token decode cannot
overflow (capacity is one row per expert); batched prefill can, and the
evaluation receipts written before this audit do not record it, so
evaluation-time exposure is now recorded going forward. `H*`'s utility labels
are unaffected because they run with `full_bank=True`, which bypasses the
router entirely.

### P12.2 — the H* target: tooling now, numbers when the artifact exists

There is no utility label artifact on disk, because `H*` has not run. Rather
than report an empty diagnostic, `oracle/eval/t2_diagnostics.py` implements the
three measurements the plan asks for and fails closed with that reason:

* what per-expert standardisation deletes (KL between the absolute and the
  standardised target, per-expert means and spreads, the first-component energy
  share, and the pairwise correlation structure);
* whether routing is expert identity or token features, using the exact split
  `z_e(t) = w_e.mean(h) + w_e.(h - mean(h))`, with top-1 and top-4 overlaps
  against the utility oracle;
* whether the four/five boundary is inside the label noise (`routing_margin` over
  the per-expert label standard deviation, with a percentile bootstrap).

The deployed-budget gap — full-bank leave-one-out versus the router's own top-4
— is declared **blocked**, with the pass recipe and the exact command written
down. `docs/T2-DIAGNOSTICS.md` fixes the decision rules before the measurement
and registers the successor target as **P12.17**: deployed-budget labels,
learned per-expert offsets retaining the absolute utility, the temperature set
to the measured label noise instead of one standardised unit, and an
interaction-aware set value whose interaction term is measured and reported.

### P8.1/P8.2 — the fresh set, three selectors, bounded composition

* `oracle/eval/fresh_tasks.py`: hand-authored items with embedded checkers and a
  reference continuation per item for the language-model instrument. Item count:

<!-- BEGIN GENERATED: worklog-p8-items -->
26
<!-- END GENERATED: worklog-p8-items -->

  The by-domain split is in the contamination receipt and in
  `docs/P8-HARNESS.md`.
* `oracle/eval/fresh_contamination.py`: disjointness checked, not asserted —
  8-word text shingles against HumanEval, GSM8K and TruthfulQA, and 13-token
  shingles against the pinned held-out windows in the model's own tokenizer.
  Verdict `disjoint`, zero offenders; it fails closed on any overlap.
* `oracle/eval/complementarity.py`: the pre-registered fixed rule, a logistic
  router on prefix features (optionally plus a hidden-state summary) trained on
  one half and scored on the other, and a non-deployable oracle; paired
  percentile bootstrap against `always-D` and `always-M`; selector cost per
  decision; and a completion-loss instrument so the same path serves the
  self-test and the checkpoint run.
* `oracle/train/composition.py`: `y = S + a r_D + b r_M` with a zero-initialised
  gate projected onto an L2 ball, a three-way interference report (accounted /
  added effect of M / deviation from D alone) because only one of those is
  bounded, and the composed admission config. `scripts/p8-composition-admission.sh`
  admitted it through the Rust ledger. Active unique parameters:

<!-- BEGIN GENERATED: worklog-p8-compose-active -->
774371384
<!-- END GENERATED: worklog-p8-compose-active -->

  of which the gate, declared as a selector:

<!-- BEGIN GENERATED: worklog-p8-compose-gate -->
57400
<!-- END GENERATED: worklog-p8-compose-gate -->

  The Python mirror agrees with the ledger.

No arm has been scored on the fresh set: that needs the GPU. `--run-composition`
is the one entry point that does not exist yet, and it needs a trained gate.

### Verified courts at this revision

- `oracle` — the full CPU suite, all passing.
- `rust` — the composed arm admitted by `selvedge admit`.
- `perf_numbers` — every generated region matches its receipt, including the
  new `p121-*` and `p8-*` blocks.
- `commit_identity` — OK.

### Limitations

- The mis-assignment rate is a *model* of the permutation (uniform random
  pairing conditional on the recorded counts), not a measurement of the actual
  permutation; the exact dropped count is measured.
- The audit covers only the routed arm that has finished. It must be re-run after
  `H` and `H*` finish so the receipt covers all three.
- P12.2 has no numbers yet, by construction, and says so in every place a reader
  might expect one.
- The Rust items are checked textually, not by invoking `rustc`.
- The fresh set is small: one set, one seed, no power analysis.

### Next experiment

None started here. The next GPU job is whatever `scripts/run-t2-pilot.sh` does
next (`H` then `H*`). The next program steps remain P12.1a (evaluation-only
re-score of `X` under the corrected placement) and P12.2 (run the diagnostics the
moment `H*` writes its labels).

## 2026-11-16 — S11: four successor courts built and CPU-tested while the pilot owns the GPU

The detached T2 pilot was running `H` throughout this session and owned the
device. Nothing here started, stopped, restarted or reconfigured a container, no
GPU job was launched, and the pilot was read only. The work is CPU-only code,
tests and pre-registration, so the moment the card is free the four courts below
launch with the commands in `results/T2-PLAN.md` §6.

### P8.2 — the missing composition entry point

`oracle/train/composition.py` now has `--run-composition` and
`--score-composition`, on top of the gate that already existed:

* a relative, per-token budget `rho_t = kappa ||S_t||_2` with `kappa = 0.05`
  (an absolute radius has no scale-free meaning across layers);
* a composed arm (`ComposedFFN`, `CompositionArm`) that installs T2 `D`'s dense
  adapter and T2 `M`'s hashed memory on **one** frozen core, maps the two T2
  checkpoints onto it with an explicit, fail-closed key remap, and freezes every
  adapter/memory parameter;
* gate-only training (`train_gate`) on the registered T2 calibration slice:
  256 steps x 1,024 positions = 262,144 tokens in stream order, AdamW lr 3e-3,
  seed 1001, the registered T1.4 distillation objective, features cached, and a
  receipt that logs `a`, `b`, projection share, budget utilisation, the three-way
  interference separation, the active union and the selector cost;
* `composition_comparison` with the four registered references and the per-item
  oracle, paired percentile bootstrap (10,000, seed 20261115).

The earlier WORK-LOG note that "`--run-composition` is the one entry point that
does not exist yet" is superseded by this entry. No gate has been trained: the
GPU was busy, so P8.2 has **no measurement**.

### P11.1 — causal bottleneck repair court

`oracle/eval/causal_repair.py`:

* a student protocol (`logits(ids, intervention)`, `hidden_state(ids, layer)`),
  with `TinyLayeredStudent` as the CPU fixture and `HFCausalStudent` as the
  frozen-0.6B adapter with a forward hook at the layer output;
* the offline bounded oracle: projected Adam on the verified loss (surrogate) and
  a verifier-driven search over the identity basis plus seeded directions, both
  inside `||delta|| <= epsilon` with `epsilon = 0.10 * median ||h_l||`;
* the bottleneck definition (the **earliest** successful layer) and the deployed
  layer (median earliest success on the training split);
* `CausalRepairPredictor` over `[h_layer, z_c, one_hot(layer)]`, where `z_c` is
  the mean of the frozen input embedding and the label/verifier/logits are not
  inputs (tested by relabelling an item and asserting the prediction is
  unchanged);
* the controls: distributed LoRA at the largest rank under the predictor's
  parameter count, all-layer adapters at rank 8, and no adaptation, all on the
  same items, tokens and steps;
* the gap metric and the registered kill rule (`capture < 0.5`, transfer
  `capture < 0.25`).

Implementation gap, stated in the contract and the receipt: the 200-item
failed-item artifact does not exist. The court runs `--mode self-test` today and
the 0.6B command is ready in `results/T2-PLAN.md` §6.2.

### T3 — latent-geometry court

`oracle/train/latent_geometry.py`:

* the matched-active-cost sweep: shared `P`/`Q`, square experts (`w = L`),
  `m = 2k`, and `k` the largest value that fits the T2 `X` per-layer active
  budget. At `L = 128/256/384/512` that is `k = 117/29/12/6` (99.3/99.8/97.1/91.7%
  utilisation) — arithmetic, not measurement;
* the deterministic, capacity-free dispatch (one write per `(token, slot)`, fixed
  slot order), zero-init parity, and the per-layer/per-expert numerical Jacobian
  rank with the analytic bounds (`L` for the shared projection, `1024` for the
  native reference — the shared bottleneck, not `k*L`, is the real cap);
* deployed-budget leave-one-selected-expert-out utility labelling, utility
  diversity (correlation, JS from uniform, effective experts, top-1 share),
  decode overhead and modelled VRAM;
* a T3 admission config that reuses T2 `X`'s core slices and the Rust ledger as
  the source of truth, plus training and measurement entry points;
* `oracle/eval/baseline.py` gained `--profile t3 --latent-spec`, so T3's quality
  is scored by the same instrument as T1/T2 rather than a new one.

The T3 registration frames this explicitly as an accounting replication of
LatentMoE / Shaping Latent MoE / Multi-View Latent MoE, not a discovery.

### P9.3 — combination utility

`oracle/eval/combination_utility.py`:

* four-condition generation (`none/m/e/both`) through a restricted `CausalView`
  whose outcome keys raise, with a manifest carrying the item-id, feature and
  input hashes and the leakage assertions that were executed;
* the interaction estimator with a paired bootstrap CI, a standard error and the
  three decision labels (a CI containing zero is "no detectable interaction",
  never "independence");
* the controller prototype (additive marginals; the interaction-aware version adds
  `I_hat(x)`), independent gates, a load-balanced control and the small
  combinatorial oracle, all choosing a subset whose reward is looked up from the
  measured table so the comparison is paired;
* the GPU path toggles the memory/expert paths with forward hooks that zero their
  outputs on a calibration slice, with one window per forward to stay inside the
  16 GiB device.

### Tests

All CPU, in the oracle court (`scripts/court oracle -- python3 -m pytest
oracle/tests`): 160 passed.

* `oracle/tests/test_p82_composition_gate.py` — zero point on a tiny end-to-end
  composition, gate reachability (D-only / M-only / both), relative budget,
  coefficients independent of the residuals (no leakage), determinism, frozen
  capabilities after training, checkpoint remap and its fail-closed path,
  paired comparison and the shared bootstrap.
* `oracle/tests/test_p11_causal_repair.py` — the planted bottleneck is the
  earliest successful layer for every failed item; the oracle stays in budget;
  the verifier-driven variant repairs without gradients; the predictor captures
  the oracle gain; the controls behave as the fixture implies; relabelling does
  not change the prediction; determinism; transfer to a different bottleneck
  depth and sequence length; the kill rule.
* `oracle/tests/test_t3_latent_geometry.py` — sweep arithmetic (square experts,
  `k` maximal, matched active cost, ceiling), the T2 `X` reference count,
  shared-projection accounting, zero-init parity, determinism, dispatch against a
  direct loop, Jacobian rank against its bound, utility diversity against known
  structure, decode-overhead decomposition and the admission mirror.
* `oracle/tests/test_p9_combination_utility.py` — artifact completeness and
  determinism, the causal view blocking outcome keys, sign recovery for synergy
  and antagonism, "no detectable interaction" under independence, CI coverage
  >= 0.85 over 40 replications, both sign conventions agreeing, the interaction
  controller beating the additive one only when an interaction exists, and the
  documented control costs.

### Registrations

New sections appended to `docs/EXPERIMENT-CONTRACT.md`, with no change to any
existing section or T2 number: **P8.2** (composition gate), **P11.1** (causal
repair), **T3** (latent geometry) and **P9.3** (combination utility). Each fixes
arms, budgets, metrics, thresholds, CIs and the kill condition, and each says what
a negative result means. `docs/PROGRAM-PLAN.md` statuses for P8.2, P9.3, P10.2,
P11.1 and P12.3 now point at them; no item was renumbered.

### Pilot progress, read-only

`scripts/poll-t2-pilot.sh` at `2026-11-16T00:29:46Z`, revision `1dd7270`:
`D`, `M` and `X` are evaluated; `H` is running at step 165/1024 (16.1%) at 4,656
tokens/s and a 10.86 GiB peak, ETA about 3.4 h by its own rate; `H*` is pending.
No ordering between arms is stated, before or after this session.

### What is not true

* No court above has a measurement. Every number in this entry is arithmetic, a
  fixture, or a read-only pilot progress figure.
* The T3 `k` values are design arithmetic at the T2 `X` active budget; they are
  not a throughput, VRAM or quality result.
* P11.1's fixture shows the oracle and predictor repairing all 12 test failures
  and the distributed LoRA repairing none, because the fixture deletes the only
  class-carrying hidden signal. That is a property of the fixture by
  construction and is not evidence about the 0.6B student.

## M3 part 2 — trace-driven execution and deadline simulator (`selvedge-sim`)

Date: 2026-11-16. Revision: `29eeac7` and its receipts. CPU only; the detached T2
pilot kept the GPU.

### What was built

A new crate, `crates/selvedge-sim`, and the CLI surface `selvedge trace
export|inspect|generate`, `selvedge simulate`, `selvedge envelope build|apply`
and `selvedge sim calibrate|validate|compare-store|pareto|placement`.

- **Trace format.** `selvedge/execution-trace/v1` JSONL plus a hashed sidecar
  manifest. The record schema has no field for an expected set, a target or a
  future token, and every struct denies unknown fields, so a trace that carries
  one fails to load. Exact mode requires a declared basis and the only accepted
  basis is a whole-bundle plan read.
- **Calibration.** Empirical distributions for transfer, SHA-256, queue wait,
  insertion, eviction and host-RAM copy, sampled from the measured per-request
  JSONL, plus a log-log size law from `io-scan`. Exponential and lognormal fits
  are recorded with quantile residuals; the exponential fit is much worse, which
  is why the empirical grid is the model of record.
- **Engine.** Coalesced runs, per-run transfer plus verification, bounded
  in-flight bytes with visible backpressure, eviction and insertion cost,
  preemption and cancellation of speculative transfers, exposed-stall
  accounting, per-record deadlines, a two-timescale phase working set, and
  execution envelopes built from a disjoint prefix. Byte conservation is a
  named invariant checked on every run and property-tested.
- **Policies and predictors.** LRU, LFU, least-stale and pinned reproduce the
  measured engine's semantics; predictors are advisory behind a trait, and a
  non-deployable oracle predictor is labelled by `uses_future()`.
- **Placement.** Sequential, frequency-sorted, co-access-blocked and
  replicated-hot layouts; the surrogate weight is `lift × marginal coalescing
  cost`, with a seeded record-level bootstrap on the strongest edges, and the
  layout decision is made by the simulator, not by the surrogate.

### What was measured or validated

- **Four held-out replays.** Calibrated on `plan-order` and `lru-1of4`, then
  replayed on `lru-1of2`, `least-stale-1of4` and `none-cold` for the 0.6B bundle
  and `lru-1of2` for the 1.7B bundle. Every hit/miss decision matched the
  engine's record and the device-byte prediction was exact; the stall
  discrepancies are in `docs/SIMULATOR.md` and the receipts.
- **Live store comparison.** Executed the same whole-bundle trace through
  `ShardStore::read_required` and through the simulator. Bytes and hit counts
  agreed exactly; the stall shape agreed within a few percent; the absolute
  latency check failed because the host was saturated by the T2 pilot during the
  run (same-window `io-scan` recorded the drift). That failure is kept in the
  receipt with the environment note and is reported as inconclusive, not as a
  simulator defect.
- **Pareto, placement and envelope receipts** on the reconstructed 1.7B M2
  trace and on synthetic traces, with generated plots.

### Tests

All CPU, in the rust court: the workspace suite passes, including 73 tests in
`selvedge-sim` (52 unit, 21 engine/property/replay). New named tests include
`replay_reproduces_hand_computed_schedule`, `coalescing_merges_adjacent_extents`,
`coalescing_never_bridges_beyond_gap`, `backpressure_wait_and_fail_modes`,
`cancellation_returns_the_server`, `prefetch_saves_stall_when_it_completes_in_time`,
`prefetch_never_changes_the_selected_set`, `deadline_misses_are_counted`,
`two_timescale_cold_change_is_charged_to_the_phase`,
`envelope_invalidation_on_excursion`, `envelope_utility_is_bytes_and_time`,
`divergence_is_reported_as_a_percentage`,
`byte_accounting_is_conserved_under_random_configurations`, and the placement
invariants (`sequential_is_a_permutation_of_extents`,
`replication_amplification_is_accounted`, `objective_prices_replication`).

### A real bug found by the tests

The inverse standard-normal quantile used for the lognormal residual table was
missing Acklam's leading sign and used `sqrt(-log p)` instead of
`sqrt(-2 log p)` for the tail region, which swapped and compressed both tails.
The residual table showed a large error for a known lognormal sample; the fix is
in `calibration::normal_quantile` and is pinned by
`normal_quantile_matches_known_values` and `normal_quantile_has_unit_moments`.

### What is not true

- No GPU, PCIe or pinned-buffer number is simulated or measured.
- The absolute simulator-vs-store latency comparison is *not* established; it is
  confounded by host contention and must be re-run on an idle host.
- No trained-router trace exists. Locality, envelope and placement conclusions
  are conditional on a trained model's routing statistics.
- `--compute-ns` is an assumption supplied by the caller, never a measurement.


## S13a — M4 part 1: a real single-device inference runtime (CPU)

Status: **implemented and measured on CPU**, GPU deferred to part 2 by design.

### What was built

- `crates/selvedge-runtime`: Qwen3 forward (tied embeddings with a verified
  `lm_head`, RMSNorm, RoPE, GQA with per-head q/k norm, SwiGLU MLP read from the
  bundle, tied output projection), an in-repo f32 kernel backend, the two
  execution modes as types, the budget gate with the within-budget fallback, and
  per-token accounting written as JSONL.
- `selvedge run` in `crates/selvedge-cli`: typed flags, `--json`, nonzero exit on
  any validation, budget, witness or parity failure.
- `selvedge-convert`: a whole-tensor `read_tensor_f32` on the safetensors reader
  plus the rank-any `decode_tensor_f32` it needs, so the runtime can load rank-1
  normalization gains through the same validated reader.
- Fixture and oracle: `fixtures/tiny-qwen3` (deterministic tiny Qwen3, committed)
  and `oracle/qwen3_numpy_reference.py` (independent f32 NumPy forward, used for
  both the tiny fixture and the pinned 0.6B).

### Backend decision (mandate requirement)

Candle was measured in the rust court and rejected for this phase: its dependency
edges, cold-build time and peak RSS are recorded in the generated block in
`docs/RUNTIME.md` §2 and in
`results/runtime/backend-decision-20261116T012956Z.json`. The reasons are bitwise
parity (we must own the reduction order), `unsafe`/allocation shape under the OOM
and no-unsafe rules, seven required operations, and build weight. `tokenizers`
0.23.2 is pinned because the tokenizer must be the teacher's engine.

### Evidence

- Tiny parity against NumPy: `cargo test -p selvedge-runtime --test tiny_parity`
  (six positions including two greedy steps; the test asserts a tolerance and a
  tighter observed bound).
- Court E, tiny fixture: `tests/court_e.rs` — all cache policies, capacities,
  prefetch settings, fetch orders, injected delays and thread counts produce
  bitwise identical logits, and the cold run reads strictly more device bytes.
- Court E, real checkpoint: `results/runtime/court-e-0.6b-*.json`, produced by
  `scripts/runtime-courts.sh court-e`; all variants share one sequence digest.
- Real-model parity: `results/runtime/court-e-parity-0.6b-*.json`; the observed
  maximum absolute logit difference and the top-1 agreement are generated into
  `docs/RUNTIME.md` §1.
- Court F, CPU baseline: `results/runtime/court-f-cpu-*.json` and the generated
  table in `docs/RUNTIME.md` §10.

### What is not true

- The native-set declaration is a trust input derived from the validated
  manifest; the witness proves equality with it, not that it is honest. Stated in
  `docs/RUNTIME.md` §1 and §5, and in `docs/ARCHITECTURE.md`.
- Bitwise identity is claimed within one build across streaming variants, not
  against NumPy or PyTorch.
- Approximate mode's selector is a control with no learned parameters; no quality
  claim is made for it.
- No GPU number exists. No GPU kernel exists. GPU integration is part 2.
- Eight kernel threads are slower than one at batch 1 in this implementation; the
  numbers are measured, the explanation (per-call scoped-thread spawn) is a
  hypothesis, and a persistent worker pool is not implemented.

### Pilot (read-only observation)

The detached T2 pilot (`selvedge-oracle-court-20261115T234515Z-b23ea1`) was polled
read-only at 2026-11-16T02:02:17Z: arm H at step 540/1024, loss 1.8003, about
4700 tokens/s, 10.86 GiB VRAM in use, and host memory with ample headroom. No
container was touched and no GPU job was started from this subphase.

### Pre-existing clippy lints fixed

`cargo clippy --workspace --all-targets -- -D warnings` failed on
`selvedge-sim`'s own test code (`useless_vec` in `tests/engine.rs`,
`field_reassign_with_default` in `src/envelope.rs` and `src/synthetic.rs`) once
the workspace was relinted. The fixes are mechanical and behaviour-preserving;
they are listed here so the change is not mistaken for part of the runtime work.

## S14 — the capability court and the fresh Rust suite (CPU)

### Plan

P12.8–P12.11 need one instrument that every frontier claim will use, and P12.9
needs a verifiable domain that is not a public benchmark. This session builds
both, plus the pre-registration that has to exist before any arm runs. The GPU
belongs to the detached T2 pilot, so nothing here evaluates a model; the harness
is tested end to end on reference solutions, declared wrong solutions and
scripted fake model outputs.

### What was built

- **A fresh executable Rust suite.** Items are authored as failing crates with a
  harness-owned hidden test, a reference solution and a declared wrong mutation,
  across borrow/lifetime repairs, failing-test debugging, API migration,
  dependency/edition/feature compatibility, concurrency defects, trait/impl
  errors and multi-file repairs. The tracked tree is materialised from an
  authoring source whose `--check` fails on drift.
- **An anti-superficial-specialisation trio** (P11.21): one lifetime, one
  numeric-dependency and one quantifier-chain task, each annotated per hidden
  test with the shared dependency-closure repair and judged per surface, never
  pooled.
- **A bounded checker sandbox.** Wall clock, address space, CPU, file size and
  open-file limits; a new session and a killed process group; a deterministic
  environment built from scratch; a measured no-default-route network check; a
  bounded workdir; per-run cgroup OOM counters and output digests.
- **Five sandbox probes** that prove a pass, an assertion failure, a compile
  error, a timeout and an allocation failure are each recorded as an outcome.
- **The staged court.** `generate` in the model court, `check` in the rust court,
  category-balanced paired bootstrap, per-surface trio scoring, failure
  accounting, the ten-field control matrix, capacity scaling, Pareto,
  no-hidden-teacher, prompt-order invariance, few-shot shingle disjointness,
  tool and feedback parity.
- **The pre-registration**, bound to the suite hash before any arm run.
- **The contamination receipt**, in both required units, fail-closed.

### Evidence

- Checker self-test: `results/capability/selftest-20261116T022833Z.json` — every
  failing state fails as declared, every reference passes, every declared
  mutation fails, every probe lands on its expected outcome.
- Contamination: `results/capability/contamination-20261116T022822Z.json` —
  disjoint against HumanEval, GSM8K, TruthfulQA, the T2 evaluation windows and a
  bounded T1/T2 corpus sample.
- Pre-registration: `docs/capability-preregistration.json`.
- Instrument documentation: `docs/CAPABILITY-COURT.md`.

### A real finding from the probe self-test

On the first clean-tree rerun the timeout probe *passed*. The cause was the
binary picker: `cargo test --no-run` writes one executable per test target, and
the harness's restored `tests/check.rs` is an empty integration test that exits
successfully, so "newest executable" could select it and run zero tests. The
picker now selects the crate's unit-test binary by package name. This is exactly
the failure mode a probe exists to catch — a checker that reports a pass without
running the check — and it was found because the probes are part of the
self-test, not because of inspection.

### A real finding from the contamination gate

The first contamination run was not clean: one 13-token shingle in a debugging
fixture also occurred in a held-out GSM8K question — a sequence of decimal
literals used in an assertion. This was a coincidence rather than copied text,
but the gate is fail-closed, so the fixture's numbers were changed and the run
re-passed. No exception list was added. This is what the gate is for; a
contamination check that cannot fail is decoration.

### Tests

- `oracle/tests/test_capability_suite.py` — suite shape, manifest binding and
  drift detection, prompt determinism, hidden-test non-leakage, extraction
  rules and harness-owned path rejection.
- `oracle/tests/test_capability_stats.py` — the stdlib percentile rule against
  numpy, paired and category-stratified bootstrap, category balance.
- `oracle/tests/test_capability_court.py` — fake-adapter end-to-end runs,
  composite arithmetic, threshold decision rule, empty-answer accounting,
  control-matrix validation, complementarity, capacity slope, Pareto,
  no-hidden-teacher, parity checks, prompt-order invariance, few-shot leakage.
- `oracle/tests/test_capability_sandbox.py` — the child's own view of every
  limit, deterministic environment, timeout, allocation bound, digest stability,
  fail-closed behaviour outside a court and with a default route.
- The suites run in the python court; the real checker runs and the probes run
  in the rust court, which now carries a standard-library Python beside cargo so
  one process can drive the checkers and score their outcomes.
- The staged path was also exercised end to end through the CLI with a scripted
  adapter that emits the reference solutions
  (`results/capability/instrument-smoke-20261116T023255Z/`). That receipt proves
  the pipeline and the pre-registration binding; it is not an arm result, and its
  adapter kind is `fake`.

### What is not true

- No arm has been evaluated. Every number describes the instrument.
- The trio is below P11.21's item count and has author-only annotations; it is
  a transfer probe, not the published anti-specialisation suite.
- Feedback repair loops are recorded and parity-enforced but not implemented.
- The corpus contamination check is a bounded deterministic sample; its coverage
  is in the receipt.
- Mutation testing is one authored wrong answer per task, not a campaign.
- The Rust-runtime adapter has no end-to-end generation yet, and the PyTorch
  adapter reports residency and I/O as null with reasons.

### Pilot (read-only observation)

The detached T2 pilot (`selvedge-oracle-court-20261115T234515Z-b23ea1`) was
polled read-only: arm H at step 560/1024, loss 1.8106, about 4620 tokens/s,
10.86 GiB VRAM in use, ample host memory. No container was touched and no GPU
job was started from this subphase.

### Next

When the pilot finishes: run the pre-registered suite on the T2 checkpoints (the
exact commands are in `docs/CAPABILITY-COURT.md` §10), then the Rust-runtime
subset on CPU, then the D/M complementarity comparison through the same
verifier.

## S16 — the evaluation instrument was wrong for a hybrid, and H is evaluated

H's training container finished (exit 0, its artifacts intact) but the
orchestrator that launched it had been killed with its editor unit. The first
resume harvested the checkpoint and started the evaluation, which died with a
CUDA out-of-memory inside `harness.window_stats`.

### The failure, read from its own evidence

The court log and receipt are
`results/courts/logs/oracle-20261116T042635Z-8cc96b.log` and
`results/courts/oracle-20261116T042635Z-8cc96b.json` (stamp
`20261116T042635Z`, revision `5ede311`). In that run PyTorch had 13.34 GiB
allocated when a 298 MiB allocation failed on a 15.61 GiB card. Three things
were resident at once:

* the 1.7B teacher and the hybrid student (3.20 GiB + 2.27 GiB of weights);
* two full `positions x 151,936` f32 logits tensors plus their log-softmax and
  `p_t` temporaries;
* a 5.9 GB checkpoint loaded straight onto the device and never dropped — a
  second copy of every parameter, held for the whole run.

The checkpoint copy was the largest single avoidable term, and it is why the
same instrument survived on D, M and X: only the hybrid's checkpoint is large
enough to matter.

### The instrument

`oracle/eval/harness.py` now runs each model once for its final hidden states —
with the vocabulary head stubbed for that pass, so every arm hook still fires
(the memory arms set their token context from a forward pre-hook; calling the
body directly would leave it empty) — and applies the head to a bounded number
of positions at a time, reducing in f32 and freeing each chunk.
`oracle/eval/baseline.py` loads the checkpoint on the host, drops it after the
state-dict copy, frees the teacher before generation, and records per-stage
`max_memory_allocated`/`max_memory_reserved`. `PYTORCH_CUDA_ALLOC_CONF` is now
forwarded into every court container and recorded in the court receipt.

The registered metric is untouched: same KL(teacher||student) in nats/token,
same NLL/PPL/top-1, every position scored against the full vocabulary. This is
a storage change, not a definition change.

### Parity evidence

On the real H checkpoint, over the same held-out windows, the streamed
instrument and a full-width head produce identical NLL and top-1 and a worst
per-window KL relative difference below one part in ten million — the bf16
rounding of the head, nothing more. Receipts:
`results/eval/s16-parity-chunked.json`, `results/eval/s16-parity-full.json`.

### Named tests

`oracle/tests/test_eval_harness_memory.py`:

* `test_chunked_window_stats_match_the_full_logits_reference`
* `test_the_metric_is_independent_of_the_chunk_size` (chunk sizes from one
  position to larger than the window)
* `test_every_position_is_scored_and_none_is_dropped`
* `test_a_two_token_window_scores_exactly_one_position`
* `test_the_captured_hidden_states_are_the_model_forwards`
* `test_a_full_width_streamed_head_is_bitwise_the_model_logits`
* `test_chunked_logits_are_within_one_step_of_the_full_forward`
* `test_no_head_call_exceeds_the_chunk`
* `test_the_model_forward_still_runs_so_the_arm_hooks_fire`
* `test_a_model_without_an_output_head_fails_closed`
* `test_answer_loglikelihood_matches_the_full_logits_reference`

They run on CPU and CUDA in the oracle court (receipt
`results/courts/oracle-20261116T043850Z-fca7be.json`); the whole oracle suite
passes in the GPU court (receipt
`results/courts/oracle-20261116T043904Z-90c37a.json`).

### Orchestrator resume

`scripts/run-t2-pilot.sh` now archives a previous failure instead of erasing it
(`state-archive-failure`), refuses to compose an arm receipt silently, re-verifies
the final checkpoint before every evaluation, reuses a completed evaluation
receipt that names the same checkpoint, and gained `--from-eval` ("train
nothing; a verified checkpoint is the only way to reach the harness").
`scripts/t2_launch_record.py` writes each GPU stage's VRAM arithmetic from the
accounting receipts and the admission config, and finalises it with the court
receipt, the measured per-stage peaks and the unit's own journal accounting.

The resume path was verified on the stranded H arm. Its log line:

```
[2026-11-16T04:46:02Z] [H] recovered a verified final checkpoint; skipping training
```

### H's evaluation

The arm was evaluated on all held-out windows with no deviation: mask order
`legacy` (the registered placement), the admitted config and admission record
hashed into the receipt, chunk size recorded, and the previous failure kept in
the pilot state's `failures` history. Per-domain KL, per-window values and the
per-item outputs are in `results/eval/t2-H.json`; the summary is
`results/t2/H/eval.json`.

Mixed-domain mean KL(teacher||student), nats/token:

<!-- BEGIN GENERATED: worklog-s16-h-kl-mixed -->
0.5110
<!-- END GENERATED: worklog-s16-h-kl-mixed -->

Held-out windows pooled per domain (general is the largest domain):

<!-- BEGIN GENERATED: worklog-s16-h-windows -->
80 windows
<!-- END GENERATED: worklog-s16-h-windows -->

Task metrics, on the registered instruments:

- HumanEval pass@1 (greedy):

  <!-- BEGIN GENERATED: worklog-s16-h-humaneval -->
0.2073
  <!-- END GENERATED: worklog-s16-h-humaneval -->

- GSM8K exact match:

  <!-- BEGIN GENERATED: worklog-s16-h-gsm8k -->
0.0967
  <!-- END GENERATED: worklog-s16-h-gsm8k -->

- TruthfulQA MC1:

  <!-- BEGIN GENERATED: worklog-s16-h-truthfulqa -->
0.2766
  <!-- END GENERATED: worklog-s16-h-truthfulqa -->

Peak VRAM for the whole evaluation:

<!-- BEGIN GENERATED: worklog-s16-h-vram -->
6.39 GiB
<!-- END GENERATED: worklog-s16-h-vram -->

**These are one arm's numbers.** No comparison between arms, no ordering and no
K15-K19 decision is made in this entry; that is S17's job on five evaluated
arms. `H*` had not run when this entry was written.

### What this entry does not claim

- No T2 verdict, and no cross-arm ordering.
- Nothing about P8.1/P8.2; neither was run in this session.
- The capability court was not re-run.

## S16C — the plan is tiered and re-ordered, WS13/WS14 registered, the court made honest

No GPU work. `H*` ran untouched for the whole session and was polled read-only.
Everything below is CPU, documentation, or network fetches in a limited court.

### The ordering inversion (P-correction)

The §16 queue ended with `P12.8 → P12.10/P12.11`, i.e. the control matrix and the
frontier thresholds were scheduled **after** the experiments they govern. The
`Depends.` fields were already correct (P12.10 depends on P12.9 and P7.7; P12.11
depends on P12.10); the diagram had been drawn from an "experiments, then claims"
narrative instead. This is the post-hoc-threshold failure P6.5 exists to prevent,
inside the plan that forbids it. §0.2 records the inversion, why it is a defect,
and the rule it leaves behind: *a threshold, a row set, a control or a kill
criterion may never be written after the numbers it judges; an engineer, a kernel,
a residency tier or an additional claim may be.* The queue now puts P12.9,
P12.10, P12.11 at step 4, before every governed experiment.

### Tiers

`docs/PROGRAM-PLAN.md` §0.1 defines ACTIVE / CANDIDATE / HYPOTHESIS and indexes
all existing items by tier, with the assignments that need a reason named. No P-ID
was renumbered, retitled or deleted. ACTIVE is the T2 pilot, the one next decisive
test and the criteria and blockers that bind it; CANDIDATE is the defined mechanism
space with a cheap experiment and a kill criterion each; HYPOTHESIS is retained
with no implementation commitment.

### WS13 and WS14

- **WS13 (§18), 21 items.** Four published mechanisms staged as competing
  alternatives (adaptive expert widths, cross-layer expert reuse, global experts
  with a recurrent router, typed latent reasoning) plus the sixteen inventions, each
  with what / why / decisive experiment / falsifying control / kill criterion /
  depends / acceptance / status / effort / tier. The workstream accounting rule is
  stated once: reused parameters count **once per output token**, repeated FLOPs
  count **per use**.
- **WS14 (§19), 7 items.** Three runtime bottlenecks registered as execution-ready
  work — decoded-weight residency (`read_group` re-decodes every read, cache hits
  included), synchronous prefetch (no fetch/compute timelines, no bounded staging),
  token-at-a-time prefill (keep the scalar oracle, add a batched path) — and four
  evaluation corrections (suite scale toward hundreds of tasks, executed Rust
  verifiers, correctness separated from reference-continuation likelihood, a sealed
  evaluation set).

### Literature, third pass

Rounds 18–18d added twenty named works: the four WS13 mechanisms and the nearest
prior art for each of the sixteen inventions. Verdicts: 9 VERIFIED, 4 PARTIAL, 7
UNSUPPORTED, plus PonderNet as a supplementary VERIFIED. Consequential findings:

- **MoSE could not be located.** Three searches; the only MoSE in the corpus is
  "Mixture-of-Shape-Experts", an image-segmentation prompt method. The *mechanism*
  (nested slimmable widths) is published as MatFormer, so P13.1 is framed against
  MatFormer, not as a MoSE reproduction.
- **GMoE and TYLER are title-only.** Their ACL/EMNLP landing pages serve no
  abstract, so "recurrent routing state" and "learned typed-operator budget" are
  PROPOSED parts of P13.3/P13.4, not reproductions.
- **SMEAR's identity in the strategy note was wrong** (2306.03745 is Soft Merging of
  Experts; 2310.17680 is CodeFusion), and its abstract is about soft expert merging,
  not sequence-level routing — so P13.18's phase-boundary switching has no located
  prior art at all.
- Six of sixteen nearest-art citations could not be located (CROWN, CoLT5, function
  vectors, HAWQ, DEER, robustness ReCode), and two live homonyms are recorded
  (MoSE, ReCode).
- Six arXiv ids recalled from memory resolved to different papers (a dilepton
  physics paper, a Heun-algebra paper among them). Those records are kept: an id
  that resolves elsewhere is evidence about the id.

### Court corrections (all CPU-tested, before any arm was scored)

1. `compare()` computed the category-regression limit and ignored it in the
   verdict. The verdict is now conjunctive; `improvement_met` and `threshold_met`
   are separate; the verdict text names the regressing categories.
2. A missing recorded pre-registration digest used to skip the binding entirely.
   A registered run now requires a nonempty matching digest, the contract is
   verified against the suite hash and its threshold before the digest, and
   `compare()` takes the threshold from the receipt — refusing an external value, a
   tampered recorded value, and any registered/unregistered mix.
3. `T2ArmAdapter` no longer loads with an unchecked `strict=False`. The allowlist is
   exact and derived (trainable tensors present, frozen base omitted, nothing else),
   shapes are checked, and a supplied training receipt must agree on arm, config
   digest, admission identity, checkpoint name/digest/step and — for routed arms —
   the recorded dispatch mask placement. The checkpoint is staged on the host
   instead of being mapped onto the device.

Recorded consequence: the four finished T2 `run.json` receipts do not record
`dispatch_mask_order`, so the routed arms (`X`, `H`) cannot yet be scored through
the capability court with a verified dispatch identity, and the adapter refuses
them rather than guessing. `D` and `M` have no dispatch and are unaffected.

### Documents

- `results/REPORT.md`: the stale "no trained model is evaluated" claim is replaced.
  `D`, `M`, `X`, `H` are trained **and** evaluated on the likelihood instrument;
  what does not exist is a T2 verdict or any capability/frontier advantage. The
  M1/M2 negatives stay in the "Established" list and stay first.
- `README.md`: rewritten around four questions — what SELVEDGE is attempting, what
  has been measured, what is running, what blocks a frontier claim — with links and
  no new numbers.
- `docs/CAPABILITY-COURT.md`: the pre-registration section now records that the code
  enforces what the text always said, and the first-real-run commands carry the new
  receipt and mask-order arguments.

### What this entry does not claim

- No T2 verdict and no cross-arm ordering. `H*` was training throughout; its
  progress is reported to the operator, not interpreted here.
- No capability result: no arm has been scored on the suite.
## 2026-11-16 — S20: prior-art false negatives, dispatch provenance, critical paths, partial T2

Scope: operator corrections after the round-18 ledger wrongly concluded that MoSE
could not be located and that GMoE/TYLER were title-only. Docs and CPU/network
only; `H*` training was left untouched and polled read-only throughout.

### Prior art — three retrieval false negatives corrected

All three works were fetched directly at their true venues on 2026-11-16 UTC and
are now `VERIFIED` with the exact supported claim. The round-18 failed-fetch
receipts are kept; the old verdicts are retained as superseded history.

- **MoSE** — *Mixture of Slimmable Experts*, Tastan et al., ICML 2026, PMLR
  306:119816–119836 (`https://proceedings.mlr.press/v306/tastan26a.html`). Nested,
  slimmable experts executed at variable widths under a fixed budget. Row 115
  `UNSUPPORTED` → `VERIFIED`.
- **GMoE** — Hong & Kim, ACL 2026 long 2065
  (`https://aclanthology.org/2026.acl-long.2065/`). GRU-based recurrent router
  shared across layers, routing-logit propagation, global experts plus a local
  expert per layer. Row 117 `PARTIAL` → `VERIFIED`.
- **TYLER** — Lin et al., EMNLP 2026 main 2004
  (`https://aclanthology.org/2026.emnlp-main.2004/`). A learned policy over typed
  latent operators with per-step allocation. Row 118 `PARTIAL` → `VERIFIED`.

`docs/PRIOR-ART-AND-NOVELTY.md` gains §12.1b “Retrieval false negatives”, which
states the rule plainly: a failed fetch is not evidence of non-existence. The
failure modes were arXiv-only search (MoSE is in PMLR), landing pages that had
not yet served an abstract (GMoE, TYLER), unsupported `ti:` queries, and homonym
capture. `scripts/prior_art_audit.py` is new and audits the §12 row tally; it
passes on the corrected ledger. `docs/PROGRAM-PLAN.md` P13.1/P13.3/P13.4 are
reframed as **published baselines to reproduce and extend**, not proposed
mechanisms.

### Legacy dispatch provenance for X and H — CERTIFIED as `legacy`

Neither original receipt was edited. The evidence chain:

- both arms trained on the same `oracle/train/modeling.py` blob
  (`e61ca0d`, sha256 `9ccf56e8…`), whose `combine()` has no `mask_order` and
  restores, weights, then applies the sorted-position mask — byte for byte the
  legacy placement;
- `f13581f` (which introduced `mask_order`) is the direct child of `H`'s training
  commit `3e5454`, and `X` trained earlier still;
- both ran on image digest `sha256:a754d3e1…`;
- a fixed, deterministic batch replay (`oracle/eval/dispatch_provenance_replay.py`,
  court receipt `results/courts/oracle-20261116T054753Z-7abdd2.json`) shows the
  historical combine bitwise equal to the current `legacy` path and different
  from `pair` whenever overflow occurs.

`results/t2/PROVENANCE-AMENDMENT-X-and-H.json` binds the receipt hashes, training
commits, code digest, image digest, replay evidence, inferred mask order
(`legacy`), and an explicit confidence statement. The capability adapter accepts
it only under `--provenance-amendment`, validates arm, inferred order, receipt
digest, and replay digest, and records the amendment digest in downstream
provenance. Tests cover acceptance and four refusal paths
(`oracle/tests/test_capability_court.py`, 46 passed in court). A
corrected-dispatch sensitivity re-evaluation from the same checkpoints is
registered in the amendment and **not launched**.

Impact is quantified from existing receipts, not re-measured: `X`'s audit block
and `H`'s own `train.jsonl` routing blocks give each arm's dropped fraction and
worst sampled layer; the values live in the amendment and in
`results/t2/dispatch-audit-20261115T234953Z.json`, never hand-copied here.

### Plan — critical paths and tier gate

`docs/PROGRAM-PLAN.md` §0.0 now separates **(a) neural capability** (finish H*,
certify historical checkpoints, run the fresh Rust suite) from **(b) physical
execution** (runtime optimisation, trained-module integration, real transfers and
latency). It states explicitly that **P14.1–P14.3 do not gate the T2 capability
run**, because that court executes through the PyTorch adapter. The WS14 headers
and tier index are corrected to match. P14.5 is corrected: Rust cases already
execute through `cargo test`; the gap is checker coverage and scale, not a
missing executable path. A dated tier-enforcement gate (2026-11-16) says
MoSE/GMoE/TYLER reproductions and the sixteen inventions do not start until `H*`
reports and the first defensible capability measurement exists; exceptions need a
written, registered justification. `README.md` now says plainly that there is no
competitive or GPU performance claim while the CPU runtime has measured latency
receipts.

### Partial T2 record (exploratory)

`results/T2-PARTIAL.md` and its JSON companion are generated by
`scripts/t2_partial.py` from the stored per-item evaluation receipts. They give
the paired X-vs-H comparison per task (outcomes, discordant counts, percentile
paired bootstrap) and the per-arm composite. Recomputed task means match the
stored summaries exactly. The document states in its first paragraph that it is
**not** the registered five-arm verdict, that `H*` is outstanding, and that the
corrected-dispatch runs are outstanding. Direction: `H` is below `X` on the
composite with an interval excluding zero; the exact values are in the generated
file, not restated here.

### What this entry does not claim

- No T2 verdict and no cross-arm ordering beyond the exploratory partial above.
- No capability result: no arm has been scored on the suite.
- No competitive or GPU performance claim. `H*` was not touched.
- The provenance amendment certifies what the legacy code *did*, not that the
  legacy dispatch is correct.

## 2026-11-16 — S22: the post-`H*` pipeline is built and CPU-tested before the GPU frees

The GPU belongs to `H*` (training, detached) and was polled read-only. Everything
below is CPU work: the commands that will spend the GPU window were written,
gated and tested now, so the window is used only for measurements.

### What was built

- **`scripts/post-hstar.sh`** / **`scripts/post_hstar.py`** — one resumable
  orchestrator with eight ordered stages: `preflight`, `verify`, `eval`,
  `instrument`, `capability`, `sensitivity`, `verdict`, `compare`. State lives in
  `results/t2/post-hstar-state.json` and is written after every transition; a
  resume re-runs only stages that are not `done`, and archives the previous
  failure instead of deleting it.
- **`scripts/t2_verdict.py`** — the guarded verdict reader. The rule itself is
  imported from `oracle/eval/t2_decision.py` (one implementation of the
  registered statistic), and the reader adds the gate: every required receipt
  exists for the registered arm, seed and budget; each checkpoint hashes to its
  own sidecar; config and admission digests match the files *and* the training
  receipt; a routed arm without a recorded dispatch mask must be covered by a
  certified provenance amendment whose digest is carried into the verdict; and a
  sensitivity receipt is accepted only at `mask_order=pair`. It emits
  `results/t2-verdict.json` and renders `results/T2-VERDICT.md`, with the
  decision-bearing scalars published as generated regions.
- **`scripts/capability_compare.py`** — the registered Level-1 composite across
  the five arms, with per-category intervals and the conjunctive verdict from
  `oracle/capability/court.compare`. It refuses an unregistered result, a
  mismatched pre-registration digest, a different suite or threshold, a result
  below the category minimum, and a routed arm with no provenance amendment.
  `M`'s registered comparator `D0` has no verified-task row, so that comparison
  is recorded as not measured rather than silently substituted with `D`.
- **Contract T2.12 A10** — the corrected-dispatch sensitivity is registered now,
  before either receipt exists: arms `X` and `H`, same checkpoints, evaluation
  only, `mask_order=pair`, label `S1`, point-only reporting, and an explicit rule
  that the registered K15–K19 verdict stands whatever the sensitivity shows.
- **`results/T2-PLAN.md` §7** — the one command, the stage/guard table, the exact
  per-stage commands, the limit arithmetic, the resume semantics and the test
  command.

### Named CPU tests (all passing before any GPU stage ran)

`oracle/tests/test_t2_verdict.py` (12): missing receipt, config-hash mismatch,
admission-hash mismatch, checkpoint-sidecar mismatch, missing provenance
amendment, amendment that does not bind the receipt, a genuine `K15`/`K16` win,
an arm with lower compute and worse quality reported as a `K17` failure,
cross-run determinism, a sensitivity receipt that is not `pair`, a sensitivity
that is reported without re-thresholding, and thresholds that do not move.

`oracle/tests/test_capability_compare.py` (13): a genuine gate win, an aggregate
win that fails a category regression, missing arm, unregistered result,
mismatched digest, missing amendment, amendment that does not cover the arm,
`H*` without a recorded mask, wrong suite, wrong threshold, missing category
minimum, the `M-D0` not-measured record, and exit `3` on refusal.

`oracle/tests/test_post_hstar.py` (11): a full run in stage order with every
limit recorded, a resumable mid-stage failure that repeats no earlier work,
GPU-busy / low-memory / live-container preflight refusals, a missing checkpoint,
a missing training receipt with no pending state, a verdict that stops on a
missing receipt, a capability `generate` failure, a dry run that launches
nothing, and the sizing arithmetic.

### What this entry does not claim

- No T2 verdict: `H*` is still training and the verdict reader refuses to emit
  anything without its receipts (that refusal is itself one of the tests).
- No capability result: no arm has been scored on the fresh suite.
- No performance number. The limits in §7.4 are configuration derived from
  already-recorded VRAM peaks, not new measurements.
- The scripted court in `oracle/tests/fake_court.py` is test infrastructure; the
  production path still runs every stage through `scripts/court` inside Docker.

## 2026-11-16 — S23: the hybrid's interaction diagnostic, and P13.14 prepared

### Context

`H*` owned the GPU for the whole session and was not touched. Everything below
is CPU work inside the oracle court, with an explicit memory limit on every
command, over artifacts that already existed. No training, no forward pass.

### Plan

1. Answer "why did `H` lose `M`'s and `X`'s gains" from stored artifacts, and say
   precisely what those artifacts cannot answer.
2. Prepare plan item P13.14 (omission-conditioned recovery) properly:
   compensator, the strong alignment baseline, a pre-registration, and CPU tests.
3. Keep the tier gate closed and record that this session executed nothing.

### What was measured (receipt-backed)

- `oracle/eval/h_diagnostics.py`, 28 CPU tests in
  `oracle/tests/test_h_diagnostics.py`. It measures the item-level outcome
  partition from the stored T2 evaluation receipts; router geometry and
  displacement from the *reconstructed deterministic initialiser* (with a
  wrong-seed control in every row); `H*`'s usage, dead experts and top-`k` set
  diversity from the stored pre-expert states; memory gate values, path
  magnitudes and row reads from the stored weights and the exact calibration
  tokens; and the expert/memory interaction proxies with permutation nulls.
  Output: `results/H-DIAGNOSTICS.md`, rendered from
  `results/t2/H-DIAGNOSTICS-20261116T065442Z.json` by the tool, so no number in
  it is hand-written.
- The registered P12.2 diagnostic also became measurable this session, because
  `H*`'s Phase U artifact now exists. `oracle/eval/t2_diagnostics.py` ran on it
  and wrote `results/t2/diagnostics-20261116T070700Z.json`. `docs/T2-DIAGNOSTICS.md`
  records the two decision rules that receipt fires. The deployed-budget utility
  and the interaction term remain unmeasured.

What these two receipts do **not** establish: they are not the registered T2
verdict, they do not re-score anything, and every statistic that needs `H`'s or
`X`'s own hidden states is deferred with its exact command (the diagnostic's
`--deferred-commands`), because producing those states needs a forward pass and
the GPU was not free.

### What was prepared (no run)

- `oracle/train/recovery.py`: `C_theta(h, S)` as a rank-`r` bottleneck over the
  layer input and a learned set code, zero-initialised on its last layer, with
  only `(hidden_states, selected)` as inputs; the set-independent affine
  low-rank alignment baseline fitted in closed form and truncated to the rank
  the compensator's own element count allows; a shuffled-condition control; the
  arm matrix and the M2-style ledger.
- `oracle/eval/recovery_court.py`: `--plan` (arm matrix and ledger from the
  registered constants, no model), `--smoke` (tiny model, CPU, exercises
  install → capture → fit → gradient), and the registered run.
- `docs/EXPERIMENT-CONTRACT.md` P13.14: five arms, matched-byte rule, data,
  metrics, decision order, the task-relevance clause and the kill condition,
  all fixed before any measurement, plus the exact GPU command and its memory
  arithmetic.
- `oracle/tests/test_p13_recovery.py` (24 tests): the zero point (bit-exact
  reproduction of the sparse layer), the accounting, the leakage boundary,
  determinism, installation into the model, and a synthetic case where the
  omitted groups' function is known — where the conditioned compensator learns
  it and both the alignment baseline and the shuffled control do not.

### Courts

`oracle/tests` — the whole directory, inside the oracle court; `perf_numbers`;
`commit_identity`; `oracle` (the diagnostics runs). Receipts are under
`results/courts/`.

### What this entry does not claim

- No P13.14 measurement: no arm has been trained, and the tier gate of
  `docs/PROGRAM-PLAN.md` §18 is still closed (`H*` training, no capability
  measurement). No exception was requested.
- No capability, performance or frontier claim, and no revision of any T2 arm,
  threshold or kill condition.

## 2026-11-16 — S25: P18.6 cut-set discovery from stored omission artifacts

### Plan

P18.6 is the only `ACTIVE` item in WS15–WS18 and the only diagnostic the tier
gate permits before it opens. This session executes it on CPU over the stored
M1/M2 artifacts, with no model execution and no GPU work; `H*` owns the GPU and
was polled read-only. The analysis was pre-registered in a dated section of
`docs/EXPERIMENT-CONTRACT.md` and committed before any headline statistic was
computed (commit `1a8ad21`).

### What was built

- `docs/EXPERIMENT-CONTRACT.md` P18.6.1–P18.6.8: data sources with stamps,
  revisions and hashes; an internal discovery/replication split by document;
  the exact statistics; the decision thresholds D1–D4; the P18.6 kill
  condition; the GPU experiment that would follow the gate; and the reproduction
  command.
- `scripts/p18_6_cutset.py`: a deterministic, standard-library-only analysis.
  It verifies the registered input hashes and refuses to run on a mismatch. It
  writes one JSON receipt and one generated Markdown report.
- `results/P18.6-CUTSET.md` and its stamped JSON receipt under `results/p18.6/`.

### What was measured

The primary artifact is the stored M2 interaction receipt: 160 calibration
contexts, each measuring the retained masks `S`, `S+g`, `S+j`, `S+g+j` and the
exact teacher KL. The analysis reports the pairwise non-additivity distribution,
the marginal-magnitude asymmetry and cross-window sign instability, the
unary-only versus pairwise-additive explanation in sample and out of sample, the
monotonicity/submodularity violation counts, the measured-minimal destructive
set test, a covering-array-versus-random coverage simulation over the stored
design, and a secondary layer-omission additivity check from the M1 frontier.
The numbers are in the receipt and the generated report; this log does not
repeat them.

### Decisions

- **D1 passed:** the stored interactions are large, not second-order.
- **D2 failed:** the sign of the pair interaction does not replicate across the
discovery/replication split, so a cell-level interaction estimate is worse than
the unary-only model out of sample.
- **D3 failed, and the test is structurally degenerate on this instrument:**
every measured KL value is far above the usable-tier bound, because every
non-probed layer keeps the static top-5 magnitude mask. No measured immediate
subset can be safe at the registered `epsilon`, so the zero minimal count says
nothing about trained models. P18.6's kill condition therefore fires and
nothing is fed to P16.12.
- **D4 failed:** on this population a strength-2 covering array over the stored
design factors shows no measured recall advantage over random sampling.

### Instrument gap and recommendation

The trained banks `X` and `H` have no combination measurements at all, and `H*`
has only single-expert full-bank leave-one-out labels. The stored M2 instrument
is a degraded, non-deployed selection regime on eight calibration windows, five
layers and four fixed pairs; it cannot supply a cut set for a trained model.
Because D1 or D2 failed, the registered rule does not justify a GPU
covering-array experiment from this evidence. The receipt records the minimal
pass that would close the gap (full-bank, pairwise and triple omissions on the
trained banks, with a discovery/replication split and random plus additive
controls) and its cost estimate from the measured forward time; that pass may
not run before the §20.0 gate opens.

### Verification

- The analysis ran in the Python court with `SELVEDGE_MEM=8g`,
  `SELVEDGE_CPUS=8`, `SELVEDGE_PIDS=1024`, `SELVEDGE_TMP=2g`; the court
  re-inspected the applied cgroup limits and recorded an enforcement match. No
  OOM, no new model forwards, no GPU.
- The first run failed closed on a hash-check error in the script (it hashed the
  label sidecar file instead of the artifact hash the sidecar declares); the
  failure is retained and the fix is a separate commit. Every subsequent run
  reproduces the same decisions.
- A second full run at the reporting revision produced an identical statistical
  payload (all fields except the clock stamp, receipt path and revision); its
  court receipt is `results/courts/python-20261116T073321Z-1f3230.json`.
- `commit_identity` and `perf_numbers` are run through the court; the receipts
  are under `results/courts/`.

### What this entry does not claim

- No cut set, capability result or trained-model result. The report is a
  negative result about the stored instrument, not about SELVEDGE's trained
  models.
- No new model pass was run, no threshold was changed, and no M1/M2 result was
  revised.
- The discovery/replication split is internal to calibration text; no result
  replicated on the held-out corpus.

## 2026-11-16 — S27: post-`H*` bookkeeping, the v3→v4 binding defect, and the T2 verdict run

### Plan

`H*` training was still running detached when this session opened (step 867 of
1024, GPU 93%), so the GPU window was used only for read-only polling. The
session's CPU work, in order: (1) record the P18.6 negative and the measured
power gap in the plan and the report; (2) append a numbered item for closing the
power gap without renumbering anything; (3) fix the post-`H*` pipeline's
pre-registration binding, which was stale against the 144-item suite; and
(4) wait for `H*` to exit, then run `scripts/post-hstar.sh --resume` supervised,
stage by stage.

### What was changed before any GPU work

**`docs/PROGRAM-PLAN.md`.**

- **P18.6 → executed, negative, kill criterion fired.** The status now names the
  receipt (`results/p18.6/cutset-20261116T073406Z.json`, stamp
  `20261116T073414Z`, revision `03a72dc3b4a09014f4ae6e84982b62f9b1d64d50`), the
  report (`results/P18.6-CUTSET.md`), the pre-registration commit
  (`1a8ad21`), the four decisions (D1 yes, D2 no, D3 no, D4 no) and the fact
  that the ε test is structurally degenerate on the stored instrument rather
  than merely empty. `feed_to_P16.12`: no. The tier-index row and the
  "only `ACTIVE` item" note now record that the item has been discharged, so
  nothing in WS15–WS18 is `ACTIVE`.
- **P14.4 → implemented at 144 scored tasks**, with the registered power gap
  stated in the item itself rather than left to the receipt, and the pointer to
  the generated values (`docs/CAPABILITY-COURT.md` §7.1).
- **P14.7 → implemented, frozen and unexercised**: the 24 sealed tasks are
  hash-bound in `fixtures/capability/SEALED.json` and in the v4 registration,
  no arm has been scored against them, and the set's own size cannot carry a
  10% claim.
- **P14.8 appended** (WS14 goes from 7 items to 8; the tier index and the
  CANDIDATE count are updated, and **no existing P-ID is renumbered, retitled
  or removed**). Its deliverable is a suite and design that reach ≥80% power at
  a 10% relative effect; its acceptance evidence is a power curve **simulated
  from the real per-item outcome matrix**, not a formula; the 10% threshold,
  the conjunctive category-regression limit, the bootstrap seed and the
  resample count are explicitly not movable by it.

**`results/REPORT.md`.** Records, as a quotation naming the receipt stamp and
revision, that the scored suite is **under-powered for a 10% claim at n = 144**:
the analytic MDE sits almost on the threshold while the simulated decision power
of the registered conjunctive rule at that effect is about half, so the binding
constraint is the rule and not the interval. It also records that the sealed
holdback is frozen and unscored and that no under-threshold null in the file is
read as evidence of no effect.

### The defect found and fixed: the pipeline bound a superseded registration

The post-`H*` pipeline was written (S22) while the registered suite was 72 items
and bound `docs/capability-preregistration-v3.json`. S26 then doubled the suite
to 144 and pre-registered it in `v4`, an append-only supersession that keeps the
threshold, the decision rule, the bootstrap seed and the resample count
identical and changes only the suite hash, the item count and the per-category
minimum.

Because `oracle/capability/court.py::verify_preregistration` refuses a contract
whose `suite_sha256` or `suite_items` disagrees with the suite manifest, the
capability stage would have failed closed on its **first** generate call for
**every** arm — before any GPU time was spent, but after a needless `H*`
evaluation cycle had queued behind it. The defect is a stale pointer, not a
threshold: nothing registered changed.

Fixed in four places, all pointer-only:

- `scripts/post_hstar.py`: `PREREG` → `v4`. The item floor is now **read from
the registration** (`primary.suite_items`) instead of the hard-coded 72, and a
registration that declares no suite size is refused rather than passing by
default — so a future expansion supersedes the registration and the check moves
with it.
- `scripts/capability_compare.py`: the default `--preregistration` and its usage
text → `v4`.
- `scripts/t2_verdict.py`: the (unused) `PREREG_REL` constant → `v4`.
- `oracle/tests/fake_court.py`: the scripted court now serves `v4` and its
fixture result carries 18 items per category and 144 items, so the fixture
cannot agree with a pipeline that reads the old size.

Validation, in the python court with `SELVEDGE_MEM=12g`, `SELVEDGE_CPUS=8`,
`SELVEDGE_PIDS=1024`, `SELVEDGE_TMP=8g`: 143 tests passed across the nine
torch-free capability / post-`H*` / verdict modules; the court re-inspected the
applied cgroup limits and recorded an enforcement match. The torch-dependent
modules were not collected in this image (no `torch`), which is the same split
earlier sessions used; no GPU was touched.

### Failures on the real run, and the instrument fixes

Four defects surfaced in the first real execution of the pipeline. None of them
is a threshold, a metric or a registered rule: each is launch arithmetic or an
import/compute-path bug that the CPU tests could not reach, and each is now
fixed with a regression test where a CPU test can reach it.

**F1 — the container was handed the host's interpreter path (exit 127).** The
instrument stage passed `sys.executable` to the command that runs *inside* the
court image, which resolved to the host's `/usr/bin/python3`; the image has its
interpreter at `/usr/local/bin/python3`, so the stage died instantly. Fixed by
naming the interpreter as `python3` and letting the image resolve it on `PATH` —
the same convention `scripts/run-t2-pilot.sh` uses. Regression test:
`test_every_container_stage_uses_the_image_interpreter` asserts, over the
recorded argv of a full run, that every container stage starts with a bare
`python3` and never an absolute path. The scripted court had hidden this because
it reads the script name and ignores the interpreter.

**F2 — `ModuleNotFoundError: No module named 'harness'` in `generate`.**
`oracle/capability/adapters.py` inserted `oracle/train` on `sys.path` but
imported `harness`, which lives in `oracle/eval`; every other module in the tree
inserts the eval directory. The path setup is now one named function
(`model_path_dirs`) that returns both, and
`test_the_t2_adapter_can_import_both_model_modules` asserts that each directory
contains a module the adapter imports. The CPU tests had not reached it because
they use the scripted fake adapter, not the real `T2CheckpointAdapter`.

**F3 — mixed-dtype matmul in greedy decoding.** With the import fixed, `generate`
died on `expected mat1 and mat2 to have the same dtype, but got:
c10::BFloat16 != float`. The arms' extra modules hold fp32 parameters while the
frozen base is bf16, so those matmuls are only defined inside autocast. The
training loop (`oracle/train/loop.py`) and the registered evaluation instrument
(`oracle/eval/harness.py`) both run under
`torch.autocast("cuda", dtype=torch.bfloat16)`; the newly written generation path
was the only forward pass in the tree that did not. It now does, so generation
uses the *same* registered compute path as every other measurement of these
arms — this is not a new numeric path and no metric changes. No CPU test can
exercise this (it needs torch and a GPU), so the evidence is the stage itself:
the run fails closed with the dtype error rather than producing a number.

**F4 — the declared memory limit never reached the container.** This one is the
serious one, and it is a directive-2 defect. `systemd-run` does **not** inherit
the caller's environment, so the `SELVEDGE_MEM`/`SELVEDGE_CPUS`/`SELVEDGE_PIDS`/
`SELVEDGE_TMP` values the pipeline set on the subprocess were dropped; the court
fell back to its built-in `8g` default while the stage state recorded `16g`
(D) and `24g` (the hybrids). `docker inspect` on the running container showed
`Memory=8589934592` against a declared 16 GiB. The limits are now passed with
`--setenv=` on the unit as well as `--property=MemoryMax=`, and after every stage
the runner re-reads the court's own `memory=<n>g` report of the limit it applied
to the container's cgroup and **fails closed** on a mismatch or on a missing
report in systemd mode. Verified on the re-run: `docker inspect` reports
`Memory=17179869184` and `MemorySwap=17179869184` with `PidsLimit=1024`, and the
court line reads `memory=16g`. Tests:
`test_the_systemd_unit_receives_the_limit_in_its_environment`,
`test_a_mismatched_applied_limit_fails_closed`,
`test_a_matching_applied_limit_is_recorded`,
`test_a_systemd_stage_that_reports_no_limit_fails_closed`.

The same run also exposed a resumability gap that is now closed: a `generate`
that exits zero without writing its receipt was only discovered by the *check*
stage failing on a missing file, which names the wrong stage. The capability
stage now verifies the receipt exists immediately after `generate` and fails with
the arm, the exit code and the missing path named.

One honest note on the sequence: the first `generate` for `D` was stopped by the
operator **after** F4 was detected and before it wrote its receipt, which is why
the state records a `check` failure with a `FileNotFoundError` for that attempt.
It was a deliberate stop to avoid running four arms under a limit the state
misreported, it is retained in `results/t2/post-hstar-state.json`, and the arm
was re-run from scratch under the corrected arithmetic.

**F5 — `H*`'s arm receipt does not carry the dispatch mask order, so the adapter
refused it.** `generate` for `H*` failed closed with "the training receipt
records no dispatch mask placement (P12.1)". The cause is a composition defect,
not a missing measurement: `oracle/train/loop.py` writes
`"dispatch_mask_order": arm_model.mask_order()` — read back from the
*constructed* model — into both `run-meta.json` and `summary.json`, and both
retain `legacy` for this run; `scripts/t2_pilot.py arm-receipt` simply never
copied the field into `results/t2/<arm>/run.json`. The composer now copies it
(with the summary as a fallback) plus the dispatch-exposure block, so later arms
carry it.

`H*`'s receipt is preserved byte-for-byte rather than recomposed, because
recomposing it would have dropped `run.argv` (the training state that supplied it
was consumed) and changed a receipt the verify stage already bound. Instead the
arm is certified the way the program already certifies provenance:

- `scripts/provenance_audit.py` (new, CPU only, standard library) writes
  `results/t2/provenance-replay-Hstar-20261116T122413Z.json`. It reads the order
the loop recorded in both artifacts, checks that they agree with each other and
  with the dispatch-exposure block, checks that `oracle/train/loop.py` and
  `oracle/train/modeling.py` are **byte-identical to the blobs at the launch
  revision** (`8ac067f`, git blob ids recomputed over the file bytes), and checks
  that the record is *informative*: the placements differ only on overflow and
  this run dropped 251,785,734 of 8,408,236,032 pairs;
- `results/t2/PROVENANCE-AMENDMENT-Hstar.json` certifies `H*` as `legacy`, binds
  `results/t2/H*/run.json` by sha256 and names that audit as its replay receipt.
  Its own text states that the value is **recorded, not inferred** — unlike `X`
  and `H`, whose order had to be inferred from a pre-correction code blob.

The pipeline now passes the right amendment per arm (`X`/`H` → the S20
amendment, `H*` → this one) and its freshness check requires the matching
amendment digest for whichever arm is routed. The verify stage's recorded
`dispatch_mask_order: null` for `H*` is left as it was: it is an accurate record
of what that receipt said at the time, and the amendment is what supplies the
missing field. This is instrument/provenance work; no threshold, arm or metric
changed, and no measurement was redone.

### Three more reader defects, all the same class (host paths, stale names)

**F6 — the verdict reader was handed the host repository path.**
`scripts/t2_verdict.py --repo-root /mnt/...` runs *inside* the court, where the
repository is the working directory; every required receipt was reported missing
and the reader refused. The verdict and compare stages now pass `--repo-root .`,
which resolves correctly in the container and in the scripted-court fixture. The
argv regression test now also asserts that no stage hands the container the host
repository path.

**F7 — the verdict reader required a schema the instrument never emits.** It
gated on `selvedge/t2-eval/v1`; `oracle/eval/baseline.py` writes
`selvedge/eval-receipt/v1`, and every real receipt — including the frozen `D0`
reference — carries the latter. The test fixture carried the same wrong string,
so the test agreed with the bug. Both names are now accepted, anything else is
still refused, and two tests pin it: the registered-instrument schema is
accepted and an unknown schema is refused.

**F8 — three receipts predate the instrument recording the admission digest.**
`D`, `M` and `X` were evaluated before `baseline.py` started writing
`model.admission_sha256`; `H` and `H*` have it. The reader refused the arms
whose digest was *missing*, which conflates "not recorded" with "mismatched".
A missing digest is no longer treated as a mismatch: a recorded digest must
still match (the mismatch test still fails closed), a missing one is reported as
`digest_recorded_by_instrument: false`, and the identity is established by the
path, the admitted flag and the active-parameter count agreeing with the ledger
and the training receipt. The verdict document says so rather than implying a
binding it does not have.

**F9 — the comparison reader crashed on a zero baseline.** With every arm at
zero the relative difference is undefined and the JSON is correct, but the
summary print used a float format on `None` and raised `TypeError` *after*
writing the receipt. It now prints `undefined (baseline is zero)`, the gate
fails, and a test pins the all-zero case end to end.

**F10 — the verdict document emitted sentinels the perf-numbers court cannot
see.** Five kill-condition verdicts and six per-arm composites were emitted
*inside markdown table cells*, and the court requires a sentinel to occupy its
own line, so those eleven numbers could never be verified. The tables now carry
plain rendered values (covered by the document's byte-for-byte `--check`) and
the same numbers are repeated as standalone generated blocks below each table.
The verdict stage was re-run with `--force`; `perf_numbers` now verifies all 660
blocks.

### The run

`H*` finished its registered budget and the orchestrator composed its receipt and
ran its evaluation; the post-`H*` pipeline then ran all eight stages to `done`
under `systemd-run --user --collect --unit=selvedge-post-hstar` with a 4 GiB
outer limit (the earlier T2 orchestrator unit's measured peak was 74 MiB) and
per-stage `SELVEDGE_MEM` values re-derived from each workload's measured VRAM
peak. The post-`H*` state records **ten failed stage attempts before success** —
the ten are the ten defects above, each named, and nothing after a failure ran.

`H*` training, measured. Tokens:

<!-- BEGIN GENERATED: worklog-hstar-tokens -->
67108864
<!-- END GENERATED: worklog-hstar-tokens -->

Throughput, tokens per second:

<!-- BEGIN GENERATED: worklog-hstar-tok-per-s -->
5031
<!-- END GENERATED: worklog-hstar-tok-per-s -->

Peak VRAM in GiB (allocated, the loop's own counter):

<!-- BEGIN GENERATED: worklog-hstar-vram-gib -->
10.81
<!-- END GENERATED: worklog-hstar-vram-gib -->

GPU-hours of wall-clock:

<!-- BEGIN GENERATED: worklog-hstar-gpu-hours -->
4.05
<!-- END GENERATED: worklog-hstar-gpu-hours -->

Each value above is a generated block on its own line, so the perf-numbers court
re-resolves it from `results/t2/H*/run.json`.

### What the runs decided

**T2's registered verdict stands: `error_direction_not_supported`.** `K15` fails
(`H*` vs `H` is not decisive on the primary composite and the KL difference is
inside the registered margin), `K16` fails (`H*` is worse than `D` on the primary
composite), `K18` fails (`X` beats `D` on the composite but regresses pooled KL
beyond the margin), and `K17` passes (`M` vs `D0`). The negative is kept as the
result: no bank scaling, and the mechanism does not move into WS4/WS5 without a
new pre-registration. Numbers: `results/T2-VERDICT.md` and `results/T2-PILOT.md`,
both generated.

**The corrected-dispatch sensitivity `S1` changes nothing.** `X` and `H`
re-evaluated with `mask_order=pair`: both move by less than one percentage point
of the primary composite, in the same (negative) direction, and the registered
verdict is untouched. The V2 retraining tier is **not** run.

**The verified-task comparison is a floor effect, not a ranking.** All five arms
scored zero verified tasks on the 144-task suite plus trio. Outputs were
unparseable or failed to compile, except two items (one `H`, one `H*`) that
compiled and then failed a hidden test. The reference solutions pass the same
checkers and the self-test proves the court reaches them, so this is not an
instrument failure; it is a 0.6B distilled model outside the instrument's reach.
The comparison's registered outcome is
`conditional_capacity_not_supported_on_verified_tasks` with every relative
difference undefined.

**This is the honest reading, and it is not a capability claim either way.** A
floor cannot rank arms, so the null says nothing about `H*` versus `H` on Rust
tasks; it says the suite does not discriminate at this scale. It also breaks an
assumption of the P14.4 power analysis, which models a 0.5 baseline: the real
baseline is zero, so the effective power is *lower* than the reported figure.
That is recorded in `results/REPORT.md` and is P14.8's first job (estimate item
difficulty from real solver runs rather than assume a baseline). Nothing here is
presented as a systems or cache result.

### Verification performed (receipts, not prose)

- The python court ran the torch-free test modules after each fix; the final
  counts are 16 post-`H*` tests, 50 capability-court tests, 14 comparison tests
  and 15 verdict tests, all passing, each run with an explicit court limit and a
  recorded enforcement match.
- Every GPU stage ran in the oracle court with a container limit re-checked
  after start (`docker inspect`: `Memory=17179869184` and swap equal for the
  16 GiB arms; `25769803776` for the 24 GiB hybrid arms) and the runner re-read
  the court's own `memory=<n>g` line, failing closed on a mismatch.
- `commit_identity` and `perf_numbers` pass before every push; `perf_numbers`
  verifies 660 blocks, including the eleven verdict numbers that were previously
  unverifiable.

### What this entry does not claim

- No capability win, and no capability loss: the verified-task comparison is
  uninformative at this scale and is reported as such.
- No seed-level result: one seed per arm, `D_M` and `HL` registered and unrun.
- No sealed-set result: the 24 sealed tasks were never scored by any arm.
- No claim that the corrected dispatch is irrelevant in general: only that it
  moves neither `X` nor `H` past the registered thresholds at this checkpoint.

## 2026-11-16 — S29: the registered calibration pass (GPU), and its refusal

### What ran

The GPU was free. The registered pass ran in the order the protocol fixes: smoke
one arm to size the limit, generate the calibration split on `D`, `M`, `X`, `H`
and `H*`, check on CPU with partial credit, then `calibrate` against
`docs/capability-calibration-preregistration-v1.json`. The scored split was not
derived, no comparison between arms was run, and the sealed holdback was never
named by any command.

Every GPU stage ran under `systemd-run --user --collect --pipe` with
`MemoryMax=MemorySwapMax=12G`, `TasksMax=1024`, `CPUQuota=800%`, a 4 GiB tmpfs
and the matching `SELVEDGE_*` variables passed with `--setenv` (the S27 lesson:
the unit does not inherit the caller's environment). Host `MemAvailable` was read
immediately before each launch; no second heavy job ran. Every court receipt
reports `enforcement=ok`, exit 0 and `oom_killed=false`.

The 12g limit is derived, not inherited. The first smoke (three items, `H*`) was
launched at the §15 24g for one run only; its measured peaks and the arithmetic
are in `docs/CAPABILITY-COURT.md` §17.1. The full pass peaked at:

<!-- BEGIN GENERATED: worklog-cal-peak-arms -->
7.358 GiB process peak, 8.616 GiB cgroup peak, 4.668 GiB VRAM allocated
<!-- END GENERATED: worklog-cal-peak-arms -->

### What it decided

The protocol **refused a suite version**:

<!-- BEGIN GENERATED: worklog-cal-window -->
1 accepted, 69 retired (68 zero pass, 1 does not discriminate), 0 saturated, 0 with a partial-credit spread
<!-- END GENERATED: worklog-cal-window -->

Every category had a reformatted cell, so no category
was scorable and no ladder could authorise a claim. The ladder was also not
monotone: only the `dependency-compat` patch cell was off the floor, which made
the declared easiest tier look hardest. `VERSION.json` was not written; the receipt
records the refusal and `calibrate --check` recomputes it.

The measured cause is stronger than a floor. Reading the raw generations, every
unambiguous patch letter any arm produced was the same letter, and the
diagnostic arms likewise; the body and file formats emit a constant markdown
header instead of code. The arms do not condition on the item. A tolerant parser
would therefore score a constant, and the only "passes" are the base rate of the
reference letter. `docs/CAPABILITY-COURT.md` §17.2 and §17.6 carry the numbers
and the changes that could yield a discriminative instrument (a solver that
answers, or a re-registered single-letter instrument with a response-entropy
screen). Neither was implemented, because doing so without a fresh calibration
pass is the defect the registration exists to prevent.

Power was computed from the measured baseline and paired variance, not from 0.5,
and the instrument is far from the registered target. The smallest effect it
could detect at the items available:

<!-- BEGIN GENERATED: worklog-cal-mde-planning -->
0.1175 absolute, 20.57x baseline
<!-- END GENERATED: worklog-cal-mde-planning -->

P14.8 remains open with that shortfall published, not absorbed.

### Receipts

- `results/capability/calibration-20261116T164100Z.json` — the window, cells,
  ladder, power and the recorded refusal (stamp `20261116T162758Z`).
- `results/capability/calibration-analysis-20261116T165500Z.json` — the
  per-arm/per-format tables, the answer-shape projection and the resource peaks
  the limits were sized from (stamp `20261116T163243Z`).
- `results/capability/calibration/<arm>/{generations,result}.json` — the five
  arms' generation and check receipts, plus `generate.log`.
- `results/capability/calibration-smoke/` — the two smoke runs and the failed
  partial-category check.
- `results/courts/oracle-*.json` and `results/courts/rust-*.json` — limits,
  enforcement and the OOM verdict for every stage.

### Failures and fixes

**F1 — the smoke check refused a partial generation.** `check` requires every
item of the selected category, so the three-item smoke could not be checked. The
second smoke generated the whole `lifetime-borrow` category and used
`--category`; the full pass generated the complete split before any check. No
code change.

**F2 — a refused calibration could not be verified.** `calibrate --check` with
`--suite-version-out` demanded a `VERSION.json` and exited 3 when the protocol
had refused one, which made the refusal itself unverifiable. The receipt now
stores `suite_version` (`emitted`, `revision`, `reason`); `--check` recomputes
the decision and fails closed if a version file exists, if the reason no longer
recomputes, or if the records now support a version. `oracle/tests/
test_capability_formats.py` covers those cases, and the full capability test
selection passes.

**F3 — the receipts could not size a limit.** Neither generate nor check recorded
a measured peak, so directive 2's arithmetic had no input. Both receipts now
carry `resources`: process `ru_maxrss`, cgroup `memory.peak`, and CUDA
`max_memory_allocated`/`reserved`. The 12g limit in this session is the first
one derived from a receipt this court wrote.

**No OOM, no CUDA failure, no retry caused by a killed job.** The one failed
check invocation is F1, and it failed closed before scoring anything.

### What this entry does not claim

- No capability comparison: the calibration measures the instrument, not the
  arms, and the protocol forbids reporting one.
- No calibrated suite version, and no scored split derived from one.
- No change to the registered T2 verdict, its thresholds, or any historical
  receipt.

## 2026-11-16 — S30: the response collapse, the instrument fix and the D/X pilot

### What ran

S30 took the S29 refusal as a hypothesis about the *instrument* and tested it,
then built and measured the replacement. Four stages, each an explicit court run
with a measured OOM limit:

1. **Prompt forensics** (oracle court, CPU): the exact rendered prompt for a
   sample of patch and diagnostic items, dumped verbatim with the tokenised form
   the adapter feeds the model, the prompt audit (candidates present, labels
   unique, reference position, example letter) and the stored answers' letter
   distribution.
2. **Sanity ladder** (`D`, GPU, oracle court): a trivial MCQ at three positions,
   the same MCQ with the example letter set to the correct and to a wrong letter,
   and a real patch item under the registered prompt, without the example, with an
   explicit one-character instruction, under three candidate permutations and
   with two few-shot examples.
3. **Implementation**: likelihood selection for the selection formats, a new
   stage (`select`), a `--format` filter on `check`, the response-entropy screen,
   the v2 registration, and CPU tests for all of it.
4. **Pilot** (`D`, `X`, GPU select + rust check): the whole calibration split's
   selection items, then the executable checker, then the power inputs from the
   real item matrix.

`MemAvailable` was read before every stage (≈89 GiB with no other heavy job), each
limit was sized from the stage's own measured peak with the registered headroom
factor, and every court receipt reports `enforcement=ok` with no OOM kill. The
ladder ran in a 6g court at a measured 3.452 GiB process / 3.112 GiB cgroup /
1.843 GiB VRAM peak; `D`'s selection ran in a 6g court at 3.357 / 3.020 / 1.895;
`X` was first run at the same 6g and its cgroup peak landed exactly on the limit
(5.459 GiB process), so it was re-run at 9g (5.459 × 1.5 = 8.19, rounded up) with
byte-identical choices — both receipts are kept.

### What the forensics found

The prompt is clean in every respect the audit can check: every candidate's
content is present, each label appears once, nothing is truncated, and the
tokenizer is applied with no chat template (the receipt records the template that
*would* have been applied, and that the arm is a raw-text next-token model makes
raw text the right choice). The defect is one line: the answer-format section
shows a concrete example letter, `C` for patch-selection and `B` for
diagnostic-identification. The stored answers are that letter:

<!-- BEGIN GENERATED: worklog-s30-d-patch-letters -->
22 recoverable letters
<!-- END GENERATED: worklog-s30-d-patch-letters -->
over
<!-- BEGIN GENERATED: worklog-s30-d-patch-items -->
23 patch items
<!-- END GENERATED: worklog-s30-d-patch-items -->
on `D`, and every recoverable one is
<!-- BEGIN GENERATED: worklog-s30-d-patch-constant -->
C
<!-- END GENERATED: worklog-s30-d-patch-constant -->

The ladder then separated the two possible mechanisms. With no example letter the
arm still answers, but with a fixed position prior; with an example letter it
copies it even when the example is the wrong answer. Of the trivial MCQs at three
positions it answered
<!-- BEGIN GENERATED: worklog-s30-mcq-correct -->
1
<!-- END GENERATED: worklog-s30-mcq-correct -->
of
<!-- BEGIN GENERATED: worklog-s30-mcq-items -->
3
<!-- END GENERATED: worklog-s30-mcq-items -->
correctly, and under permutation its chosen *content* was invariant on
<!-- BEGIN GENERATED: worklog-s30-content-stable -->
0
<!-- END GENERATED: worklog-s30-content-stable -->
of
<!-- BEGIN GENERATED: worklog-s30-ladder-items -->
2
<!-- END GENERATED: worklog-s30-ladder-items -->
The completion shows why: asked for a letter, the arm continues the instruction
("For example, if the correct answer is `A`, then return `A`.") and degenerates
into repeated fences. It is completing a document, not answering a question.

**So the collapse was a prompt-conditioning defect, not evidence about the
tasks**, and the S29 refusal was correct about the instrument.

### What was built

`oracle/capability/scoring.py` scores each candidate's *content* as a
continuation of the prompt and takes the arg-max — no generation, so nothing can
collapse. The sum and the length-normalised mean of token log-probabilities are
both recorded (the mean is primary), the letter-score ranking is kept as an
unscored position-bias diagnostic, the label and reference flag never enter the
scored text, and a scored sequence past the exactness bound refuses rather than
truncating. `court select` emits a generations-schema receipt so the registered
`check` still compiles the chosen candidate; the selection prompt is a named
variant whose hash `check` recomputes. `check --format` scores exactly the
formats a run asked. The response-entropy screen marks a near-constant format
non-discriminating. All of it is registered in
`docs/capability-calibration-preregistration-v2.json`, which appends to v1; v1,
v4 and the T2 verdict are untouched. CPU tests cover the ranking algebra, the tie
rule, the leakage guards, the prompt variant, the registration check, the screen
and the two stages end to end.

### The pilot

Two arms, the selection formats, one arm at a time. The instrument now varies its
answers and produces non-zero, non-saturated cells:

<!-- BEGIN GENERATED: worklog-s30-exec-items -->
23 executable items
<!-- END GENERATED: worklog-s30-exec-items -->
of which
<!-- BEGIN GENERATED: worklog-s30-both -->
10 are passed by both arms
<!-- END GENERATED: worklog-s30-both -->
,
<!-- BEGIN GENERATED: worklog-s30-neither -->
11 by neither
<!-- END GENERATED: worklog-s30-neither -->
and
<!-- BEGIN GENERATED: worklog-s30-discordant -->
2 by exactly one
<!-- END GENERATED: worklog-s30-discordant -->
. `D` alone passes
<!-- BEGIN GENERATED: worklog-s30-d-patch-passes -->
11
<!-- END GENERATED: worklog-s30-d-patch-passes -->
of
<!-- BEGIN GENERATED: worklog-s30-patch-items -->
23
<!-- END GENERATED: worklog-s30-patch-items -->
patch items through `cargo`, and both arms verify the same diagnostics; the
answer entropy is far above the screen's minimum, so both selection formats are
scored rather than excluded.

**It does not separate the arms**, and it does not rescue the free-form formats:
`body-completion` is excluded by the screen, `file-replacement` passes the screen
and still passes nothing. That distinction is recorded in
`docs/CAPABILITY-COURT.md` §18.5 — the screen excludes, it never accepts.

### Power

The measured inputs, from the real item matrix:

<!-- BEGIN GENERATED: worklog-s30-baseline -->
0.478261
<!-- END GENERATED: worklog-s30-baseline -->
baseline and
<!-- BEGIN GENERATED: worklog-s30-variance -->
0.086957
<!-- END GENERATED: worklog-s30-variance -->
paired-difference variance, so at 80% power for a 10% relative effect the
registered formula needs
<!-- BEGIN GENERATED: worklog-s30-required-n -->
38 items per category
<!-- END GENERATED: worklog-s30-required-n -->
against
<!-- BEGIN GENERATED: worklog-s30-available -->
3 available per category
<!-- END GENERATED: worklog-s30-available -->
The smallest detectable effect at the available n is
<!-- BEGIN GENERATED: worklog-s30-mde -->
0.1686 absolute
<!-- END GENERATED: worklog-s30-mde -->
**The five-arm comparison was not run**, because at this n it could not decide the
registered effect. The shortfall is published, not absorbed.

### Receipts

- `results/capability/forensics/forensics-20261116T173500Z.json`
- `results/capability/ladder/D-20261116T173600Z.json`
- `results/capability/pilot/{D,X}/{selection,result}.json`
- `results/capability/pilot/pilot-analysis-20261116T173200Z.json`
- `docs/capability-calibration-preregistration-v2.json`
- `results/courts/{oracle,rust,python}-2026*.json` for every stage

### Failures and fixes

**F1 — the selection stage rejected its own registration.** `load_preregistration`
is the v1 endpoint contract and refuses the v2 schema; the select stage now loads
the v2 document as a document and verifies it with the module that applies it,
and `check` gained a v2 branch that calls the same verifier. **F2 — a partial
selection run could not be checked.** `check` required every item of the suite;
`--format` restricts both the missing-item check and the scoring to the formats a
run asked, so an unasked item is neither scored nor counted as a failure. **F3 —
`X` ran at `D`'s limit.** Its cgroup peak reached the limit, so it was re-run at
the limit its own measured peak implies; both receipts are kept. **F4 — a
precedence bug in the pilot's paired differences.** The first analysis receipt
computed `a if pass else b - c` instead of `(a if pass else b) - c`, inflating the
variance; the corrected analysis is the one cited and the bug is visible in the
diff. **F5 — the entropy screen is not a correctness screen.** The first draft of
§18.5 claimed `file-replacement` was excluded by the screen; the generated numbers
showed it passes the screen and passes no item, and the text was corrected to say
so.

### What this entry does not claim

- No capability comparison between arms, and no five-arm comparison.
- No calibrated suite version; the scored split is still not derived.
- No claim about the arms' ability to write code: the selection formats measure
  ranking, not generation.
- No change to the T2 verdict, the v1 window, v4, or any historical receipt.

## 2026-11-16 — S32: the frontier target verified, the literature re-read, the program registered

### What ran

Verification only: no training, no model execution beyond the one refused
accounting call, and no change to any registered T2 or verdict receipt.

1. **Model identity, over the network, in the oracle court** (`--net`,
   `SELVEDGE_MEM=4g`): `scripts/s32_frontier_models.py probe` resolved twelve
   candidate ids through the Hub API, pinned each to the returned commit SHA,
   listed every file at that SHA with its byte count and LFS oid, fetched and
   hashed every small file (config, tokenizer, generation config, LICENSE, README,
   preprocessor), and accounted **every tensor from the safetensors header by
   HTTP range read** — no checkpoint was downloaded to count it. Receipts:
   `results/frontier-models/20261116-probe-p{1,2}.json`.
2. **One checkpoint fetched and hash-verified**
   (`20261116-fetch-qwen35-08b.json`): the single safetensors shard's SHA-256
   equals the Hub's LFS oid, and the config/tokenizer/vocab files were verified
   the same way.
3. **Accounting and ceiling arithmetic** (`20261116-ceiling.json`): per-model
   text/vision/MTP split, FFN vs always-active, layer-type counts, and the
   headroom to the 1e9 ceiling.
4. **Disk arithmetic** (`20261116-disk.json`): filesystem free bytes, the
   teacher-cache formula at every requested token/k pair, the largest fitting
   configuration, and the checkpoint byte counts.
5. **`selvedge inspect-model` on the real checkpoint**, in the rust court. It
   failed closed: the auditor supports `model_type == "qwen3"` and a top-level
   `hidden_size`; Qwen3.5 nests its language config under `text_config`.
   Receipt `results/courts/rust-20261116T181516Z-5237e7.json`.
6. **Literature, four fetch rounds** (`--net`, `SELVEDGE_MEM=2g`):
   21 of the first round's 30 Semantic Scholar queries returned 429, so the
   documented arXiv fallbacks were used in rounds 26–28. Receipts:
   `results/prior-art/prior-art-fetch-20261116-round2{5,6,7,8}-ws19-*.jsonl`.

### What it decided

- **The anchor exists and is a hybrid, not a dense transformer.**
  `Qwen/Qwen3.5-0.8B` at
  `2fc06364715b967f1860aea9cf38778875588b17`, Apache-2.0. Its configuration and
  its tensor names agree with the vendor page on every checkable fact: 24 text
  layers, Gated DeltaNet on eighteen of them, gated attention on the other six at
  the stated interval, hidden 1024, vocab 248320, tied embeddings (no
  `lm_head.weight` exists), and one MTP layer. **Correction (S38, 2026-11-17):**
  this entry audited the **chat** revision; the WS19 anchor is the `-Base`
  revision, which is the same graph over different weights and four fewer added
  tokens (`docs/FRONTIER-TARGET.md` §10). Every architecture fact above holds for
  both, because the safetensors header is byte-identical.  The vendor's "sparse
  Mixture-of-Experts" sentence is **contradicted** by the 0.8B checkpoint: every
  layer is a dense SwiGLU MLP.
- **The "~752M" claim is right about the text tower and wrong about the model.**
  Text tower 752,393,024 elements; the checkpoint is 873,438,784 because it also
  carries a vision tower (100,592,896) and an MTP head (20,452,864). The vendor
  page prints neither — it says "0.8B". Both numbers are generated in
  `docs/FRONTIER-TARGET.md` §4.
- **One named model does not exist.** `LiquidAI/LFM2.5-1.2B` returns HTTP 401 with
  the Hub's "Invalid username or password"; the search recorded in the receipt
  lists the closest verified alternatives, and the program's stretch model is
  `LiquidAI/LFM2.5-1.2B-Instruct` (verified, **LFM Open License v1.0**, not
  Apache).
- **`google/gemma-3-1b-it` cannot be audited without accepting Google's terms**
  (gated, weight range read returns 401). It stays a reference.
- **The teacher-cache budget the plan proposed does not fit this machine.** The
  proposal's 190 GB of top-k logits exceeds the free space the disk receipt
  records, by a factor of roughly two and a half. The largest configuration that
  fits with the program's own headroom is 0.25B tokens at k = 32; 0.1B tokens at
  k = 64 is the fidelity arm. Both byte counts are generated in
  `docs/FRONTIER-TARGET.md` §7.
- **The loop mechanism is published, with a number.** SMELT loops the middle half
  of layers twice under matched per-token FLOPs, parameters and KV cache, up to
  54B, and reports 6.8–18.0% training-FLOP savings on the compute-optimal
  frontier; Retrofitted Recurrence installs recurrence into Qwen2.5-0.5B-Instruct
  at two budgets with loop-1 non-inferiority. The registered program treats its
  loop stage as a reproduction.
- **The per-pass-memory novelty survives only in a narrow form.** The closest
  published factorial (2606.20737) crosses loop recurrence with shared-memory
  access and reports that *repeated memory rereading alone* crosses the grokking
  barrier; the shared-memory-across-loops work (2610.02383) shares a KV cache;
  RecurTrace (2609.03379) attends over loop time. None of them reads a
  **token-addressed learned lookup table inside each pass**. That, and only that,
  is the residual claim.
- **Three names in the plan are homonyms or absent.** `MoLE (ICML 2025)` is not in
  the ICML 2025 PMLR index under either plausible title; `MeSH` and `MeSA` return
  only unrelated homonyms; `LongCat-Flash-Lite` is not a paper.

### What was registered

- `docs/PROGRAM-PLAN.md` §24 (WS19: six stages plus a blocking prerequisite, each
  with what/why/deps/acceptance/win/kill/effort/tier), §25 (WS20: the stop and
  keep lists), and the S32 tier-index row in §0.1.
- The win condition as stated in §24: audited ≤ 1e9 unique active on every token,
  beat the base model on one pinned harness, close at least half of the measured
  0.8B→2B gap on one family, stretch past the 1.2B model, three seeds at the claim
  stage only.
- Risks, explicitly: the anchor's hybrid needs kernels the runtime does not have;
  per-pass memory may tie shared-key memory; n-gram memory may help less at this
  depth; the ceiling must be re-derived per token; and two of the stage
  mechanisms are reproductions.
- `docs/FRONTIER-TARGET.md` (new): the verified identity table, the architecture
  facts, the auditor's refusal, the ceiling arithmetic and the disk arithmetic,
  with every number generated from a receipt.

### Verification performed (receipts, not prose)

- `scripts/court commit_identity` — the branch's commits carry the operator's
  identity, not an agent's.
- `scripts/court perf_numbers` — all 64 new generated regions resolve from the
  four receipts, with their declared stamps and revisions.
- The Hub's own `safetensors.total` equals the header-derived total for every
  model whose weights were readable (the probe records `matches_hub_total`).
- The one downloaded shard's SHA-256 equals the Hub's LFS oid.

### Failures and fixes

**F1 — the probe's first classifier left a tensor unclassified.** The first pass
could not place `model.embedding_norm.weight` (LFM2.5), so the receipt reported a
non-zero `unknown` bucket instead of guessing. The rule was extended and the probe
re-run; `20261116-probe-p1.json` is kept and `p2` supersedes it. **F2 — the first
ceiling pass mis-bucketed LFM2's FFN**, because its tensors are named
`feed_forward.*` rather than `mlp.*`; the effect was that the whole 1.2B model was
counted as always-active. Fixed and re-run; the receipt records the corrected
split. **F3 — the first generated-region layout failed the generator twice**:
inline sentinels in a Markdown table cell and in a bullet are not recognised as
regions, which the generator reported by name. The document was restructured so
every region owns its line; the failure is recorded here because it is the kind of
"looks generated" document that would otherwise pass review while printing
nothing. **F4 — the manifest's declared revisions were stale for the re-run
receipts**, which the generator also caught (`declared revision ... does not match
receipt revision`); they are now read from each receipt rather than assumed
constant.

### What this entry does not claim

- No capability number of any kind. No harness was run on any of these models.
- No verified vendor benchmark number. The checkpoints' vendor tables are the
  vendors' claims and are not in any receipt here.
- No audited unique-active count. `inspect-model` refuses Qwen3.5, so every count
  here is an exact element sum by component, not the auditor's union.
- No novelty for the loop stage, and no completed novelty argument for per-pass
  memory — only a narrowed residual claim and a falsification design.
- No legal opinion on the LFM Open License.

## 2026-11-16 — S33: the anchor becomes auditable, and the reference the kernels must match

S32 registered the frontier program and recorded, correctly, that its gate (P19.0)
did not exist: `selvedge inspect-model` refused Qwen3.5, so every count in
`docs/FRONTIER-TARGET.md` was an element sum by component rather than an audit.
This session closes the gate's **accounting half** and produces an independent
reference forward. It does not close the kernel half, and says so.

### What ran

- **Rust court**, offline, model cache mounted read-only:
  `cargo test --workspace` (all green, 14 new tests for the v2 auditor),
  `cargo clippy --workspace --all-targets -- -D warnings`, `cargo fmt -- --check`,
  and two `inspect-model` invocations against the real checkpoint.
- **Oracle court, GPU device-passthrough**: a transformers capability probe, a
  smoke forward of the new reference, and the parity driver — 4 prompts ×
  (vendor forward, reference forward, reference forward again, greedy 12 from the
  vendor, greedy 8 from the reference), plus a dtype and a state test.
- **Python court**: the union arithmetic, `perf_numbers --write` and
  `perf_numbers --check`, `commit_identity`.

### What was built

1. **Accounting schema v2** (`crates/selvedge-convert/src/accounting_qwen35.rs`,
   ~1,900 lines with tests). v1 is untouched; the CLI dispatches on `model_type`
   and an unknown type still fails closed. v2 reads the nested `text_config`,
   validates `layer_types` against `full_attention_interval` instead of trusting
   either, derives every tensor shape from the config, classifies each tensor into
   exactly one tower, and refuses anything it cannot classify.
2. **A declared execution path** (`--active-path text-only | text-and-mtp`), with
   the checks that make the declaration mean something: the towers must exist in
   the bytes, the text path's tensors must be namespace-disjoint from every
   excluded tower (computed, not assumed), and cross-tower reads are recorded —
   `mtp_use_dedicated_embeddings` is false here, so the MTP head reads the language
   model's embedding and the receipt says so instead of double counting it.
3. **A tie that is verified rather than trusted.** There is no `lm_head.weight`,
   so the receipt records *tie by absence plus config* with `compared_elements: 0`
   rather than pretending a byte comparison happened. A declared tie with a
   distinct `lm_head.weight` and a declared untied pair with no `lm_head.weight`
   are both refused.
4. **A Hub cross-check that can actually fail.** The auditor sums bytes
   dtype-aware and compares to `model.safetensors.index.json`
   `metadata.total_size`. The checkpoint has 36 f32 tensors (every `A_log` and
   `linear_attn.norm.weight`), and without that fact the byte sum is off by 5,184
   — the element count alone cannot tell 1,746,877,568 from 1,746,882,752.
5. **`--certify-ceiling`** with the component arithmetic and an exit code.
6. **`oracle/reference/qwen35.py`**: an independent text-path forward with every
   equation documented against the config, bf16 and fp32, and
   `scripts/s33_reference_parity.py` as its evidence driver.
7. **`scripts/s33_budget_union.py`**: the per-token union the ceiling rule is
   about, from the component counts, with declared design constants.

### The numbers

- Text tower, tie-aware: 752,393,024. Vision: 100,592,896. MTP: 20,452,864.
  Checkpoint: 873,438,784 elements / 1,746,882,752 bytes, equal to the Hub's
  declared total to the byte. Every one of those is also a checked expectation.
- `--certify-ceiling 1000000000` on the declared text-only path: certified,
  headroom 247,606,976. On `text-and-mtp`: 772,845,888, headroom 227,154,112.
- Planned per-token union: 759,405,376 (adapters shared across passes) or
  777,296,704 (distinct per pass), both with a **dense** FFN; with vision and MTP
  also executed, 880,451,136 / 898,342,464, leaving 101,657,536 in the worst case.
- `transformers==5.23.0` supports Qwen3.5 natively (`Qwen3_5ForConditionalGeneration`),
  so it is the pinned vendor reference and the independent forward is compared
  against it rather than replacing it.

### What it decided

- **The anchor is now auditable and its declared path is certifiable.** P19.0's
  acceptance evidence (i) is satisfied and (ii) is satisfied in a stronger form
  than the plan's wording: vision and MTP are not refused wholesale, they are
  accounted separately and enter the union only when a declared path names them.
- **P19.0 is not done.** Evidence (iii) — Court E parity for the text path — is
  not satisfied and cannot be until four kernels exist. The tier stays
  `CANDIDATE`, and `docs/RUNTIME.md` §13 registers the kernels, the per-kernel
  test vectors, their tolerances and the E1/E2/E3 order.
- **The ceiling is not the binding constraint.** Every reading of the planned
  union fits with room to spare, so no stage below may hide execution behind the
  ceiling rule. What decides the design is whether the mechanism helps.
- **No FLOP number is emitted for this model.** A DeltaNet layer is a scan; the
  v1 formulas would print a number with no meaning, and `--contexts` is refused
  for `qwen3_5` rather than answered.

### Failures and fixes

**F1 — the reference's five bugs, all caught by running it.** (a) RoPE frequency
tensors built on the CPU while the model was on the GPU; (b) the gated RMSNorm
applied over the whole value vector instead of per value head (the gain is 128
wide, not 2,048); (c) `cos`/`sin` broadcast on the wrong axis for `[batch, heads,
T, dim]`; (d) an fp32 rebuild that passed mangled `ParameterDict` keys back into
the loader; (e) an odd rotary width in the tiny test config (head_dim 2 at factor
0.5), which the auditor correctly refused. Each was a real defect that the
"does it say Paris?" check or the shape assertions caught within one run.
**F2 — the first `qwen3_5` test run failed 11 of 14 tests on one config value**,
for the same reason as F1(e): a rotary dimension of 1 is not a positive even
number, and the auditor was right. **F3 — the first clippy pass failed on five
lints** (two derivable `Default` impls, `% 2 != 0`, an `expect` in the shape
builder, a manual `Option::map`); fixed rather than allowed. **F4 — the first
`perf_numbers --write` refused** because the budget receipt was not yet tracked,
which is the guard working: a generated number must come from a committed
receipt.

### What this entry does not claim

- **No executed set.** The auditor is not an executor-watcher. Court E for this
  model does not exist.
- **No bitwise vendor parity.** Four prompts, argmax equality, 8/8 greedy tokens,
  cosine ≥ 0.99978 — strong evidence, not bitwise.
- **No kernel work.** Nothing in `docs/RUNTIME.md` §13 is implemented.
- **No capability, latency, FLOP or novelty claim** for the anchor. No training
  was started and no registered receipt was modified.

## 2026-11-16 — S35: status is derived, the trade is registered, the endpoint is reachable, and the plan stops waiting on kernels

This session was CPU-only and touched no GPU. Its job was not to add a mechanism
but to fix four defects that were blocking the next measurement: status prose that
contradicted the receipts, a trade nobody had registered, an endpoint no feasible
suite could resolve, and a dependency graph that made the whole quality program
wait on four Rust kernels.

### What ran

- **Python court**: `scripts/status_consistency.py --check` and `--write`; the new
  negative-case script; `scripts/s35_t4_registration.py`; the endpoint amendment;
  `scripts/s35_frontier_arithmetic.py`; `perf_numbers --write/--check`;
  `commit_identity`; `pytest` on the capability suites.
- **Python court with network** (`--net`): the sixth prior-art pass, twenty
  queries, receipts in `results/prior-art/prior-art-fetch-20261116-round29-s35-gap.jsonl`.
- No `oracle`, `eval` or `--gpu` court was used, and nothing outside the court ran
  a build or a test.

### What was built

1. **Generated status, and a court that refuses contradictions.**
   `scripts/status_consistency.py` owns `readme-status`, `report-status` and
   `report-t2-arm-state`; it derives them from the pilot state, the verdict and the
   capability comparison, and applies nine contradiction rules to the prose
   outside the blocks. `scripts/perf_numbers.py` grew `external_regions` so a
   region owned by another generator is still refused when undeclared and refused
   when declared but missing. `scripts/court status_consistency` and
   `scripts/docker-court.sh status_consistency` make it a court.
2. **T4, the K18 trade question**, in `docs/EXPERIMENT-CONTRACT.md`, with every
   quoted number generated from `results/t4/preregistration-20261116T192000Z.json`.
   `T3` was already the latent-geometry court, so the section takes the next free
   number and says why.
3. **The endpoint amendment**: `docs/capability-preregistration-v5.json` plus
   `results/capability/endpoint-amendment-20261116T193000Z.json`, generated by
   `scripts/capability_endpoint_amendment.py`. `oracle/capability/court.py` now
   resolves an endpoint *kind* (relative or absolute), refuses a mixed comparison
   and refuses a re-thresholded one; `scripts/capability_compare.py` verifies the
   kind and the minimum the way it verified the threshold. Seven new tests.
4. **WS19 rewritten**: each stage carries Win / Next route if miss / Kill; P19.0
   splits into an accounting half (audit claims) and a kernel half (stages 5–6);
   stage 2 registers the row scale with the development configuration labelled;
   stage 3 and stage 4 keep their mechanism with the prerequisite chain intact;
   stage 4's mechanism became pass- and state-conditioned addressing with three
   controls, per-pass state independence and chunking equivalence.
5. **The arithmetic receipt**
   `results/frontier-models/20261116T193500Z-frontier-arithmetic.json` from
   `scripts/s35_frontier_arithmetic.py`, which recomputes and compares on
   `--check`: the teacher route, the memory scales and the per-pass state.

Cache removed by the online teacher route, at the recommended configuration:

<!-- BEGIN GENERATED: worklog-s35-cache-gib -->
45.6 GiB
<!-- END GENERATED: worklog-s35-cache-gib -->

Required throughput for the 2B bf16 teacher alongside the anchor (an estimate
over a measured student rate, not a measured teacher rate):

<!-- BEGIN GENERATED: worklog-s35-teacher-rate -->
6,585 tokens/s
<!-- END GENERATED: worklog-s35-teacher-rate -->

Resident bytes for the largest registered memory table at bf16:

<!-- BEGIN GENERATED: worklog-s35-mem-8m-gib -->
16.0 GiB
<!-- END GENERATED: worklog-s35-mem-8m-gib -->

Per-pass temporal state across four passes at 32k context:

<!-- BEGIN GENERATED: worklog-s35-state-32k -->
1.54 GiB
<!-- END GENERATED: worklog-s35-state-32k -->

6. **The sixth literature pass** (§18 of the prior-art ledger): eight re-reads and
   twelve gap queries, no 429s, and three corrections that matter — Engram's
   U-shaped scaling law, Memory Grafting's matched-budget numbers from a larger
   grafting model, and SCONE's fixed-cost memory scaling.

### What it decided

- **Status is a derived artefact.** A hand-written status is a defect even when it
  happens to be right, because it cannot be checked.
- **The old endpoint was not strict, it was unreachable.** The measured instruction
  is `required_n_conservative` =

  <!-- BEGIN GENERATED: worklog-s35-required-n -->
686,777
  <!-- END GENERATED: worklog-s35-required-n -->

  items per category for a 10% relative effect against a registered minimum of 18,
  so the absolute endpoint with a minimum of

  <!-- BEGIN GENERATED: worklog-s35-min-effect -->
0.04
  <!-- END GENERATED: worklog-s35-min-effect -->

  replaces it.
- **The kernels gate serving, not quality.** Stages 1–4 on the validated
  reference; kernel work when a serving claim is wanted.
- **Three of the plan's questions are reproductions.** The ledger says so, and
  P19.2 now names the numeric baselines it must beat.
- **The residual novelty is one sentence long** and is stated in
  `docs/PRIOR-ART-AND-NOVELTY.md` §18.3, with three wider sentences forbidden.

### Failures and fixes

**F1 — inline sentinels again.** The first WS19 rewrite put `BEGIN GENERATED`
sentinels inside list items and mid-sentence; the generator refused with
`nested BEGIN GENERATED ... inside open region`, because an inline `BEGIN` is not
recognised while its own-line `END` is. This is the same failure S32 recorded as
F3, which is the honest lesson: the rule "every generated region owns its line"
has to be applied when writing, not when debugging. Every block was moved to its
own paragraph.
**F2 — the amendment's category count.** The script first took the category count
from the calibration receipt's `items_per_category_available`, which names one
category (the only scorable cell), so `per_category * categories != suite_items`
and the script refused. The count now comes from the receipt's own MDE block,
which states the eight categories its formula used.
**F3 — the manifest round-trip.** Rewriting `docs/perf-numbers.json` with
`json.dump(..., indent=2)` produced a 21,000-line diff because the file is
one-space-indented with `external_regions` first; the insert was redone textually,
leaving a 17-line diff.
**F4 — a duplicated sentinel.** §7.3 reused `cap-power-decision-power`, which §7.1
already declares; the generator refused the duplicate rather than silently printing
one of them. A second block (`cap-amend-decision-power`) points at the same
receipt pointer.

### What this entry does not claim

- **No measurement of a model.** No training, no evaluation, no serving.
- **No T4 result and no capability result.** Both are registrations.
- **No novelty.** The literature pass narrowed a claim; it did not clear one.
- **No claim that the teacher estimate or the state bytes are measured.** They are
  arithmetic over receipted inputs, and the plan says which stage measures them.

## 2026-11-17 — S38: the anchor's identity, the state rule, the table that does not fit, and a provenance court

This session did no training and no evaluation. It corrected four defects in the
S37 registration, produced the certificate the scoreboard should have had, and
made one class of process failure checkable.

### What it found

- **The audited model was not the benchmarked model.** The ceiling certificate
  was for `Qwen/Qwen3.5-0.8B` (chat); the scoreboard scored `-Base`. Measured, not
  assumed (`docs/FRONTIER-TARGET.md` §10): the safetensors header, tensor table
  and `config.json` are **identical**, the shard hashes **differ**, the BPE vocab
  and merges are identical, and the chat tokenizer adds four control tokens
  (`<tool_response>`, `</tool_response>`, `<think>`, `</think>`) that Base does not
  define. Element counts and the ceiling transfer; a capability score and a
  tokenizer pairing do not.
- **The per-pass state rule was backwards.** P19.4 said a pass's state is reset
  between tokens. That would make a pass a one-token function and make the
  chunking-equivalence claim false. Corrected and now measured.
- **A missing smoke was printed as a zero.** `docs/STAGE2-PLAN.md` §5 rendered
  `0.0` for every arm delta through generated regions with no receipt behind them.
- **The 8M-row table is untrainable as registered.** Dense AdamW state for it is
  three float32 vectors per element, i.e. 96 GiB on top of a 16 GiB bf16 table.
- **The anchor pre-registration postdates its own first score** by 1 h 32 m.

### What it decided

- **The bar is the distilled control, not the frozen anchor.** Every mechanism is
  trained with the same teacher signal, so "beats the frozen 0.8B" is a statement
  about distillation. The registered order is: throughput measurement, then the
  distillation-only control, then grafted memory with four controls, then the loop
  retrofit, then the pass-conditioned table. The counter-argument (looping has
  matched-compute published wins: RecurTrace, SMELT, the retrofitted-recurrence
  line) is recorded rather than argued away, and the control decides it.
- **Sparse rows and an offloaded optimiser are the registered route** for the 1M
  and 8M scales, with a host-resident LRU-bounded optimiser tier, a 16-step flush
  bucket, a registered staleness bound of 16 steps / 32,768 tokens, per-row step
  counts in the bias correction, and a utilisation-to-admission loop. The
  validation plan is 65K dense (sparse must equal dense), 1M with offload, 8M with
  NVMe.
- **Provenance is now a court.** `prereg_provenance` fails when a registration's
  first commit is not earlier than its first scoring receipt (or when its declared
  time postdates its commit, or predates the inputs it names). Its first run found
  the anchor's defect and **three older capability registrations** with ordering
  defects; all four are recorded as **named historical exceptions with evidence**,
  not silently excused and not repaired by editing history.
- **T4 and the capability-court discrimination endpoint are parked**, not
  withdrawn: GPU focus is WS19. Registrations intact, tier index updated.

### Failures and fixes

**F1 — a zero stand-in for an absent receipt.** The S37 stage-2 plan's generated
regions had no manifest block and no owner, so `perf_numbers` could not see them
and the doc printed `0.0` as if measured. The regions are now bound to the
accounting/graft/stream receipts where a receipt exists, and removed where none
does. The general lesson: a generated region is a *citation*, and a citation with
no source must fail rather than print.
**F2 — inline sentinels again.** The same file had `BEGIN GENERATED` mid-line with
`END` on its own line. The generator refused with `END ... without BEGIN`, and the
whole court stopped at the first one, hiding F1. Eight regions were moved to
own-line form. This is the third occurrence of the same mistake (S32 F3, S35 F1);
it is the writing discipline that fails, not the tool.
**F3 — a test that proved nothing.** The first version of the tiny state model set
`linear_attn.norm.weight` to zero through a too-broad `endswith("norm.weight")`
branch, so the DeltaNet output was multiplied by zero and a perturbed recurrent
state changed the logits by exactly `0.0`. The isolation check caught it — an
isolation test that cannot see the state is worthless — and the branch order was
fixed. The `carry` criterion was then restated on argmax equality and cosine
rather than on an absolute tolerance, because the full-sequence and incremental
paths use different BLAS blocking and the absolute error is a dtype property, not
a state difference.
**F4 — a provenance court that found more than it was asked to.** Its first run
returned three capability-registration violations. They are real (a declaration
six minutes after its commit; two registrations committed after an
instrument-generation receipt under them) and are now recorded as dated
exceptions with the evidence rather than by weakening the rule.
**F5 — the court caught this session's own new registration.** The distillation
control's pre-registration declared `2026-11-17T00:45:00Z` and was committed at
`00:32:22Z`; `prereg_provenance` failed it as `declared_not_after_commit` before
any run existed. The declaration was corrected to the authoring minute, which is
what the rule is for: a future-dated declaration is a false provenance claim even
when nothing has been scored, and the same rule would have caught the anchor's if
it had existed then.

### What this entry does not claim

- **No training, no evaluation, no serving.** The run order, the throughput
  protocol and the sparse-training design are registrations.
- **No measured teacher throughput.** The estimate stands, labelled as one, until
  the first GPU job of the next session runs §4's protocol.
- **No quality claim about the `-Base` checkpoint.** Its graph is audited and its
  reference arithmetic is re-exercised; its capability score is the S36
  scoreboard's, and the reference *vendor-parity* comparison has not been re-run on
  `-Base` (`docs/FRONTIER-TARGET.md` §10.3).

## 2026-11-17 — S40: the train/inference memory defect, the full-distillation bar, the comparison instrument

### The defect

A cached decoding call reached the memory layer with a length-1 hidden state, so
the n-gram address was computed from that single token plus zero padding: the
bigram and trigram heads read a different row during generation than during
training. The causal depthwise convolution had the same shape of bug — it padded
each call independently, so it lost its kernel-1 history on every cached step.
The memory output during decoding was therefore not the function the checkpoint
was trained with. This is a correctness defect, not a performance one.

The fix computes the address from the whole recorded token prefix and slices the
positions the current call executes, and the convolution carries its last
`kernel-1` input vectors in per-site state that is dropped when `TokenContext`
generation changes. The NGM control uses the same prefix/slice rule. The
learning-free addressing property is unchanged: the address is still an exact
integer function of the token ids with no learned index.

Evidence: `oracle/tests/test_stage2_memory.py` (full-sequence memory equals
token-by-token cached output, full-sequence logits equal cached continuation
logits, NGM passes the same history check, state does not leak between
sequences; non-identity convolution and non-zero gates), and
`results/stage2/memory-2m-compat-20261117T023857Z.json`.

### The checkpoint question, answered by measurement

The full-sequence forward of the registered `memory-2m` checkpoint is
bit-identical under the fix (the compat receipt records a maximum absolute
difference of zero), because at the start of a sequence the convolution history
is zeros and the address slice is the whole prefix. Fixed cached decoding
matches that forward. Pre-fix cached decoding did not, at every tested
continuation position. The checkpoint was trained with full-sequence forwards,
so its learned function is unchanged and **retraining is not required**. What
would be invalid is a cached generative score produced by the old code; the
compat receipt lists this arm's evaluation receipts and none exists.

### The bar

The frozen-backbone parameter-matched adapter is not the WS19 bar: it activates
the memory arm's active union, not the text tower's. S40 registers a
full-backbone distillation arm — `docs/ws19-full-distillation-preregistration-v1.json`,
committed before any score — that distils the whole text tower from the online 2B
teacher on the same stream and the same token budget, with the tied token
embedding frozen. The exact active-union arithmetic, per-component bytes, the
projected peak, the limit and the smoke requirement are in
`results/stage2/s40-full-arithmetic-20261117T024151Z.json`. The 8-bit optimizer
is `oracle/frontier/optim8bit.py`; bitsandbytes is absent from the pinned image,
and the in-repo construction is named in every receipt rather than silently
substituted. The peak is a projection from the measured frozen-core reference
plus exact gradient/optimizer bytes and a modelled activation bound, and the
registration requires a measured smoke before the long run.

### The instrument

`scripts/s40_compare.py` scores six arms — frozen base, adapter-2m, memory-2m
after the fix, the ungrafted module, the step-0 graft and full-distill — with one
held-out CE/KL path and merges anchored-harness receipts. It emits per-arm means,
standard errors and paired per-chunk deltas against full-distill.
`scripts/s40_score.py` is its anchored-harness half. The held-out stream is
`results/stage2/stage2-heldout-20261117T024542Z.json`: a superset build whose
first 20,250,000 tokens hash-equal the training stream, sliced after the first
EOS at or beyond that offset, so no held-out token was trained on and the
straddling document is dropped. The CPU self-test builds fake checkpoints for all
six arms and passes (`scripts/s40_compare.py --self-test`).

### The runtime partition, checked on the real model

`s40_full_arm_arithmetic.py` derives the counts from the safetensors header.
`scripts/s40_full_arm_partition.py` loads the pinned checkpoint through
`AutoModelForCausalLM` and checks the object the trainer sees: the text tower,
the frozen tied embedding, the trainable remainder, that the tied output
projection is the same parameter, and that the exported trainable state does not
carry the embedding. Receipt:
`results/stage2/s40-full-partition-20261117T024945Z.json`.

### What this entry does not claim

- **No new arm score.** No held-out CE/KL or harness receipt for the full arm
  exists; the registration and the instrument are the product of this session.
- **No measured full-arm VRAM.** The limit is projected and the smoke must
  replace it.
- **No retraining was performed and none is required** for `memory-2m`, for the
  reason measured above.

## 2026-11-17 — S42: the update was below the grid, the graft was supervised on the wrong quantity

### Plan

Three blocking defects, in dependency order: (1) the bar arm updates bf16 weights
in place with no fp32 master copy, so an update below half an ulp is discarded;
(2) the representation graft is supervised on reconstruction and was measured to
cost CE; (3) the objective that all arms use may be the wrong objective for a
corrective memory. The session implements and tests the fixes on CPU, registers
the new arms before any of them is scored, and only touches the GPU after the
running anchored-harness court exits.

### The optimizer numerics

A bf16 weight stores 7 mantissa bits (8 significand bits including the
implicit one). Within a binade `x ∈ [2^e, 2^(e+1))` the spacing is
`ULP = 2^(e-7) ≈ x·2^-7`, so half an ulp is `≈ x·2^-8 = x/256`. Adam's
normalised step has magnitude of order the learning rate, `1e-4`. A step is
lost under nearest-even when `1e-4 < x/256`, i.e. above `x ≈ 1e-4·256 =
0.0256 ≈ 0.026`; exactly, half an ulp is constant inside a binade, so the
boundary is the binade edge `2^-5 = 0.03125` (half-ulp `2^-13 = 1.22e-4 >
1e-4` above it, `2^-14 = 6.10e-5 < 1e-4` below it). The affected regime is
therefore large-magnitude weights, not the whole model: a weight of order
0.01 has half-ulp `3.05e-5–6.10e-5`, so a `1e-4` update moves it by one or two
grid points. Where the step falls between half and one ulp, nearest-even
rounds it to one ulp; only the fraction below half an ulp is truly discarded.
`oracle/frontier/optim8bit.py` now applies the exact fp32 result to the bf16
grid with one of three registered numerics:

- `stochastic` (default): bracket the exact value between adjacent bf16 grid
  points and take the upper one with probability `(x-lo)/(hi-lo)`. Expected value
  is the exact update, no extra state, and a `q`-ulp update changes the weight
  with probability `q` instead of never. Exact grid values are returned exactly.
- `nearest`: the pre-S42 behaviour, kept as a control.
- `compensated`: nearest plus a bf16 residual carry (error feedback in the
  weight-update domain), one bf16 element per trainable element, with its bytes
  reported by the optimiser rather than estimated.

**Correction (2026-11-17, S43).** This entry originally said "at the registered
1e-4 learning rate the whole update is below half an ulp of a weight of order
0.01–1, so nearest-even rounds every step to zero and the weight never moves".
That is an overstatement and the 0.01 end of the range is wrong. The arithmetic
is above: half an ulp is `≈ x/256`, so the `1e-4` step is below half an ulp only
for `x ≳ 0.026` (`0.03125` at the binade edge), and a weight of order 0.01 is
moved by one or two grid points. The measured S42 tests already showed the
corrected behaviour — a 0.49-ulp update changes about half the elements under
stochastic rounding and exactly none under nearest-even, while a 0.01-ulp
update is not fully lost — but the summary sentence described a wider regime
than the arithmetic supports. The mechanism and the three numerics are
unchanged; only the claimed cutoff is corrected.

The update is applied in bounded chunks of `2**22` elements, so the transient is
tens of MB and does not compete with the registration headroom. The rounding
generator is seeded and its state is written into the checkpoint beside the torch
and numpy states, so `--resume` replays the same sequence exactly.

**Evidence, measured not asserted.** `oracle/tests/test_s42_optim_numerics.py`
(12 tests) shows, on a synthetic weight at 1.5 whose update is 0.49 ulp: about
half the elements change under stochastic rounding and exactly none under
nearest; the measured unchanged fraction agrees with the arithmetic prediction to
within 0.02; at 0.01 ulp the measured unchanged fraction is high, is not 1.0, and
still agrees with the prediction; the seed replays the bit pattern exactly; 1000
steps of 0.1-ulp updates leave the nearest-even weight exactly where it started
while the compensation buffer tracks the exact total inside about one ulp; the
buffer's measured bytes fit the registered headroom. On real steps the optimiser
reports the bitwise unchanged fraction, the round-to-nearest fraction, the
sub-half-ulp fraction and the mean rounding error, element by element, and
`train.py` measures the first step, the first full-LR step and the final
(minimum-LR) step and summarises peak-LR and minimum-LR into the run receipt.
The smoke receipt is the first real-arm evidence; it is recorded in the S42
continuation entry, not assumed here.

**Why stochastic rounding is the default.** It is unbiased and free; the
compensation buffer is the published error-feedback idea but costs a bf16 element
per trainable element, and at the registered full arm that is the whole headroom
margin. The buffer remains registered and implemented, and the smoke's measured
unchanged fraction decides whether the default is enough. Literature and verdicts:
`docs/S42-LITERATURE.md` §1–§2.

### The corrective graft (P19.2b)

The representation graft reconstructs the teacher's mapped hidden state and adds
it to the frozen student. Measured on the 2M-token held-out slice, the step-0
teacher graft costs +0.57475 nats of CE against base while the ungrafted step-0
module is bit-identical to base (quoted from
`results/stage2/s41-heldout-comparison.json`, stamp `20261117T035128Z`, revision
`73f9574fc7ec23e8cc82a5a971a8423509e6d086`). The residual is the harm; the
injection machinery is not.

`oracle/frontier/corrective.py` keeps the reads, projections, convolution, table
and accounting and changes the mechanism: a correction gate `tanh(gate_raw)` with
`gate_raw = 0` exactly, so an untrained graft reproduces the base bit for bit; an
objective supervised by the teacher's advantage (`KL(teacher_T || base_T)` per
position, detached) rather than by mapped teacher states; and a preservation
weight, high where the base already matches the teacher, that penalises a KL
increase and the gate magnitude there. The memory's constant offset is
structurally excluded from the additive residual by splitting
`ConditionalMemory.site_mixed` out of `site_forward` (the registered arms are
numerically unchanged and their tests still pass).

The gradient ordering is stated rather than hidden: a multiplicative gate at
exactly zero blocks the pathway's gradient, so the gate opens on the first step
(`dL/dalpha != 0` at zero) and the rows receive gradient from then on. The tests
assert that ordering, the bit-exact parity with grafted rows and a non-zero
offset, the preservation weight's ordering by base advantage, and that the
preservation term strictly increases the gate derivative when the correction is
harmful where the base is right (`oracle/tests/test_s42_corrective.py`, 11
tests). `train.py` gains `--arm corrective` and `--objective correction|ce_kl`;
`graft.py --corrective` writes the init checkpoint.

Registered before scoring: `docs/ws19-corrective-graft-preregistration-v1.json`
(P19.2b + P19.8). Arms: grafted-corrective, grafted-representation (quoted, not
rerun), adapter and the gate-zero control; 2M tokens, same teacher and stream;
active union 537,601 for the corrective arms against 537,600, stated as +1 gate
scalar per site. No verdict on the mechanism is drawn from 2M numbers against the
20M bar.

### The objective question (P19.8)

Registered in the same file, before scoring: CE+KL against advantage-weighted
correction with the base-preservation term, at equal tokens and active
parameters, with a four-branch decision rule and the next route for each branch.
The *interference cost* — delta KL on held-out positions where the base's own
advantage is in the lowest quartile — is a primary quantity, so a gain elsewhere
cannot hide harm where the base is already right.

### Literature

`docs/S42-LITERATURE.md` records four focused passes with URL, fetch date, the
claim as stated and a verdict: stochastic rounding for low-precision training
(Gupta et al. 2015, Wang et al. 2018 — VERIFIED; Dettmers et al. 2022 PARTIAL for
this question), error feedback (Karimireddy et al. 2019 VERIFIED; De Sa et al.
2018 PARTIAL), gated residual memory (kNN-LM 2020 VERIFIED for the mechanism,
PARTIAL for our construction) and base preservation (L²-SP 2018 VERIFIED in
weight space, PARTIAL for output space; RLAD 2026 VERIFIED for selective
imitation). The exact combination registered here is **UNSUPPORTED** in the
checked sources and is registered as a local construction with a decision rule.

### Courts

`commit_identity`, `perf_numbers`, `status_consistency` and `prereg_provenance`
pass on the S42 commits; receipts are under `results/courts/`. The oracle test
court runs the new files with the existing stage-2 suites: 75 passing.

### What this entry does not claim

- **No new arm score.** The anchored-harness pass was still running when this
  entry was written; the scores it produced are recorded in the continuation
  entry, with their receipts.
- **No smoke and no full-arm VRAM.** The 20-step smoke has not run yet, so the
  modelled activation term in the pre-registration is still modelled.
- **No verdict on the corrective graft or on the objective.** Both are registered
  and tested; neither has been trained for a score.

**Registration timestamp correction (2026-11-17, S42).** The corrective-graft
registration declared `registered_utc` four minutes after its own first commit,
which the `prereg_provenance` court rejects: a declaration cannot be later than
the file's commit. It was corrected to `04:52:00Z`, inside the writing window and
before the commit, at a time when no arm of that registration had been scored.
The correction is recorded in the file itself (`registration_timestamp_note`) and
in the commit message.

### The anchored-harness pass: what it covered, and the runner defect

The S41 pass did **not** finish. It scored `base` and `adapter-2m` and then died
at the first call after `adapter-2m` with `line 41: arm: unbound variable`. The
cause is process, not arithmetic: commit `e8e5908` edited
`scripts/s41_anchored_harness.sh` at 04:36 while the systemd unit was executing
it. Bash reads a script incrementally by byte offset, so the shell resumed in the
middle of a different line. The unit exited `status=1/FAILURE` and
`memory-2m`, `memory-ungrafted`, `memory-grafted` and `adapter-20m` were never
scored.

The runner is now resumable: an arm whose `results/stage2/eval/<arm>/eval.json`
is complete is skipped (`SELVEDGE_FORCE=1` re-scores deliberately), and the file's
header says not to edit it during a pass. The four unscored arms were relaunched
under `systemd-run --user --collect` with `MemoryMax=MemorySwapMax=12G`,
`TasksMax=1024`, `CPUQuota=800%` and the registered `SELVEDGE_*` variables.

The two receipts that exist are quoted here with their stamps and revisions;
neither is a verdict about any mechanism, and neither is compared to the 20M bar.

| arm | task | primary metric | score | stderr | receipt |
| --- | --- | --- | --- | --- | --- |
| base | mmlu_pro | exact_match,custom-extract | 0.26785714285714285 | 0.019587377846722334 | `results/stage2/eval/base/eval.json`, stamp `20261117T043134Z`, revision `3cb6fd23fd1b3eebe13aeb3b952701665832f2e4` |
| base | mmlu | acc | 0.481555333998006 | 0.004098974719703994 | same receipt |
| base | gsm8k | exact_match,strict-match | 0.38968915845337376 | 0.013433123236110721 | same receipt |
| base | minerva_math500 | math_verify | 0.25 | 0.019384310743640384 | same receipt |
| adapter-2m | mmlu_pro | exact_match,custom-extract | 0.23809523809523808 | 0.01898373641139568 | `results/stage2/eval/adapter-2m/eval.json`, stamp `20261117T051019Z`, revision `6105bae3726fdf7fedca5d1dd4718c2bd8f7c986` |
| adapter-2m | mmlu | acc | 0.5101125195841049 | 0.004076313936257515 | same receipt |
| adapter-2m | gsm8k | exact_match,strict-match | 0.36087945413191813 | 0.013228626753925148 | same receipt |
| adapter-2m | minerva_math500 | math_verify | 0.21 | 0.01823362086530592 | same receipt |

The unweighted family means of those quotations are 0.374706 (knowledge) and
0.319845 (math) for base, and 0.374104 and 0.285440 for adapter-2m. The adapter
is higher on MMLU and lower on the other three tasks, so the family means are
close and the per-task directions disagree. That is a description, not a verdict:
both arms are 2M-token arms, the registered comparison has four arms still
unscored, and the brief forbids a mechanism verdict from 2M numbers.

## 2026-11-17 — S43: a math-weighted stream, provenance corrections, and a token-conditioned gate

### Plan

The largest controllable lever on the math family is the training stream, which
was 100% FineWeb-Edu. This session builds and registers a math-weighted mixture
before the full-distillation run, fixes two provenance defects (a total cap that
never capped, and a base score compared across two attention kernels), corrects
one overstatement about bf16 rounding, and replaces the corrective graft's
global scalar gate with a token-conditioned gate plus a signed-advantage
supervision option. GPU work waits for the running anchored-harness court.

### The math-weighted stream

`oracle/frontier/math_stream.py` builds a 20,000,000-token mixture from pinned
sources: NuminaMath-CoT 8.0M (Apache-2.0, revision `9d8d210c...`, shard
`train-00000-of-00005.parquet`, SHA-256 `25c0fa2c...`), MATH-train 2.2M (MIT,
revision `21a56338...`), GSM8K-train 1.2M (MIT, revision `740312ad...`) and
FineWeb-Edu 8.6M (ODC-By 1.0, revision `87f09149...`). That is 57.0% math and
43.0% general. GSM8K rows are written in the scored `Question: … Answer: …`
format and MATH rows in the scored `Problem: … Solution: …` format, so the
online teacher's distribution covers the worked solution after the same marker
the evaluation uses. There is no pre-generated teacher corpus and no logit
cache.

`nvidia/Nemotron-CC-Math-v1` was the intended third math source. It is gated:
an unauthenticated request for `3/part_000000.parquet` returns HTTP 401
`GatedRepo`, and no token is available. NuminaMath-CoT is the declared
substitute, with its own revision, licence and SHA-256 in the manifest; the
substitution is explicit and recorded, not silent.

Determinism is the source revisions plus a fixed 10,000-token cycle
(4,000 Numina, 1,100 MATH, 600 GSM8K, 4,300 FineWeb) repeated 20 times per
200,000-token shard, 100 shards. Every shard carries its own SHA-256 and its
exact per-source counts; rebuilding reproduced stream SHA-256
`9c5fb53b20a8f9165566b8c4e237103eb2e4c658f944bb221b3909d1dc8b6d56` twice.
`math_stream.py verify` re-checks every shard and the token total.

**Contamination.** The guard is 5-word-shingle Jaccard against every cached test
split (GSM8K, MATH-500, MMLU and MMLU-Pro, HumanEval, IFEval, ARC-Challenge,
TriviaQA) and the sealed capability holdback (`fixtures/capability/sealed/**`
plus `SEALED.json`): 51,902 items, 1,922,185 distinct shingles. Documents at or
above 0.2 are excluded and recorded by digest; a near-verbatim copy at or above
0.5 is additionally flagged. The build measured 102 excluded documents (95
MATH-train, 3 GSM8K-train, 4 NuminaMath) and a **maximum retained Jaccard of
0.1935**. The build fails closed if a source cannot fill its budget from clean
documents or if any retained document reaches 0.2. Verdict: **PASS, zero
retained overlap**; the high-Jaccard MATH-train/exam near-duplicates are
excluded rather than trained on. Receipt:
`results/stage2/stage2-math-stream-20261117T071120Z.json`.

The stream is registered as an amendment to the full-distillation bar
(`docs/ws19-full-distillation-amendment-math-stream-preregistration-v1.json`)
**before any training on it**; the previous stream is labelled general-only.
The amendment names the one declared substitution and the unchanged objective,
optimiser, token budget and step count.

### Provenance: the total cap that never capped

The anchored-subset registration says MMLU is capped at 2,000 items; the
receipts show `sample_len = 14042` in every arm. MMLU is a 57-subject group task
and the harness applies `limit` per subtask, so `--limit-map mmlu=2000` selected
`min(2000, n_subject)` everywhere. The full set was scored, and the base wall
time (654 s) is the anchor's full-set time (639 s), not a seventh of it. The
pre-registration is not edited; `docs/ws19-anchored-subset-amendment-preregistration-v2.json`
records the defect and the actual basis (full 14,042, same basis as the
anchor's 37.90 knowledge mean).

The new enforcement is new code, because `scripts/s40_score.py` was executing:
`scripts/s43_score_total_cap.py` expresses a total cap through
`simple_evaluate(samples={leaf: indices})`, allocating the requested total over
the group's leaves by largest remainder. Its self-test measured 57 MMLU leaves
and 14,042 items, and a 2,000-item total cap allocating exactly 2,000, with a
cap below the leaf count failing closed. Existing receipts need no re-run; they
are already full-MMLU.

### Provenance: why the base differs from the anchor

`scripts/s43_discrepancy_report.py` compared the raw fixed samples. Prompts and
targets are identical on every item in MMLU-Pro (504), MATH-500 (500) and GSM8K
(1,319); task configs are identical after normalising process-local function
reprs and loader metadata; model revision, harness commit, harness image,
lm_eval/transformers versions, limits, batch sizes and seeds are identical. The
generations differ (156 / 220 / 436 items), and the metric flips go both ways
(16 vs 12, 9 vs 17, 30 vs 32). The cause is the one unexamined setting: the
anchor loads through `HFLM(model="hf")` with the **transformers default
attention, measured `sdpa`**, while the stage-2 loader passed
`attn_implementation="eager"`. Different kernels, bf16 reductions in different
orders, greedy decoding amplifying near-ties. The fix is a config fix, not a
history fix: `arms.load_model` now takes `attn_implementation`, the eval
backbone uses the anchor default, the held-out instrument keeps its previous
eager path, and `build_named_arm` records the kernel in every future receipt.
Receipt: `results/stage2/s43-base-discrepancy-20261117T071438Z.json`.

### Correction: the bf16 cutoff was overstated

The S42 entry said the `1e-4` update was below half an ulp "of a weight of
order 0.01–1". The arithmetic does not support the 0.01 end. Half an ulp is
`≈ x·2^-8 = x/256`, so the step is lost when `1e-4 < x/256`, i.e. above
`x ≈ 0.026` (the binade edge is `0.03125`). At 0.01 the half-ulp is
`3.05e-5–6.10e-5`, so the update moves the weight by one or two grid points. The
mechanism and the three registered numerics are unchanged; the entry now states
the arithmetic and marks the original sentence as an overstatement.

### The token-conditioned gate and the signed advantage

`oracle/frontier/corrective.py` replaces the global scalar gate with

    alpha_t = tanh(w . [h_t ; r_t] + b),   w = 0, b = 0,

so the untrained model is still bit-identical to base (tested with grafted rows
and a non-zero constant offset). `phi(h, r) = [h ; r]` is parameter-free, so the
active cost is `2*hidden + 1 = 2,049` elements per site and the active union is
539,649 against the memory arm's 537,600. Once `w` is non-zero the gate varies
per token, and identical `(h, r)` gives identical `alpha`.

The objective gains a registered position weight: the existing KL-disagreement
weight, the signed teacher advantage `A_t = log p_T(y_t) - log p_B(y_t)` on the
supervised target, or both as the geometric mean of the normalised weights. The
preservation penalty stays, and the objective now reports harm on the tokens the
base already matches the teacher — the lowest advantage quartile (the P19.8
definition) and the signed `A_t >= 0` set — as count, mean delta KL and the
fraction worse. The comparison is registered before training in
`docs/ws19-corrective-gate-advantage-preregistration-v1.json`, with a four-
branch decision rule and the next route for each branch.

**Tests.** `oracle/tests/test_s42_corrective.py` was updated to the token gate
and `oracle/tests/test_s43_token_gate.py` was added: zero-init bit-identity,
per-token variation, identical-`(h,r)` equality, signed-advantage sign,
weight ordering, row gradient after the gate opens, determinism, the harm
measurement, and the 2,049-element cost. The full oracle suite is **576 passed**
(receipt `results/courts/oracle-20261117T071747Z-9889ee.json`).

### What this entry does not claim

- **No full-arm smoke or launch yet.** The anchored-harness court was still
  running; GPU work is deferred to the continuation entry.
- **No measured full-arm VRAM.** The 12.5 GiB limit is still a projection; the
  smoke must replace it.
- **No verdict on the token gate or the signed advantage.** Both are registered
  and tested; neither has been trained for a score.
- **No claim that NuminaMath-CoT equals Nemotron-CC-Math.** It is a declared
  substitute for an unavailable gated dataset.

### The full-arm smoke and the 20M launch

The anchored-harness pass exited at 08:00Z (the four remaining arms completed
and their receipts are under `results/stage2/eval/`). The GPU was then free.

The 20-step smoke ran on the math-weighted stream. The first attempt used an
8 GiB container bound and measured `memory.events.max = 3,794` with no OOM kill:
the CPU side was being held against the limit, so that peak is not the
workload's peak. The same 20 steps were rerun at a 12 GiB bound, which measured
**6.449 GiB** cgroup peak, `max = 0`, `oom = 0`. The GPU peak was
**11.429 GiB allocated / 11.672 GiB reserved** under the 12.5 GiB limit, so no
micro-batch, accumulation or sequence reduction was needed. The modelled
activation term was replaced by the measured **0.6227 GiB** (0.096 GiB above the
modelled upper bound). Steady throughput was **1,847 tok/s**, ETA **3.0 hours**.

Stochastic rounding's measured bitwise outcome: at peak LR (step 2, `1e-4`) it
left **17.26%** of the 498,113,344 trainable elements unchanged against 16.86%
for nearest-even; at the cosine floor (step 20, `1.07e-5`) it left **82.99%**
unchanged against 88.63% for nearest-even, i.e. it moved 17.01% of the model
where nearest-even would move 11.37%. The sub-half-ulp fraction at the floor was
88.62%. The smoke's CE rose to 17.35 at step 6 because the smoke truncates
warmup to 2 steps; the launch's 200-step warmup is the registered schedule and
is not changed on the strength of a 20-step curve.

Launch: `selvedge-fulldistill-math-20261117T080200Z.service` under
`systemd-run --user --collect`, with `MemoryMax=MemorySwapMax=10 GiB`,
`TasksMax=1024`, `CPUQuota=800%`, and the court's own container limits verified
equal. 10 GiB is 1.55x the measured CPU peak. The run is resumable
(`--resume latest` verifies the config and stream hashes) and checkpoints every
1,000 steps, two newest plus latest. Receipts:
`results/stage2/runs/full-20m-math/{manifest,launch}.json`; the run itself is
in flight and has no final receipt or score at the time of this entry.

### What this continuation does not claim

- **No 20M result.** The run is in flight; its first 1,000-step log is the next
  evidence.
- **No score for any new arm.** The 20M full-distill arm has no checkpoint to
  score yet, and the corrective arms have not been trained.

## 2026-11-17 — S44: the complete arm table, the direct answers, and the corrective arms made ready

### Plan

Three jobs, no GPU work: (1) build the actual ranking from the six anchored
receipts, with paired item-level comparisons rather than family means alone;
(2) make the three stage-2b corrective arms launch-ready and registered; (3)
prepare — not run — the bar scoring and the sequential corrective launches. The
20M math-weighted full-distillation run is training throughout and was not
touched.

### The arm table is generated, and the item level changes the reading

`scripts/s44_arm_table.py` owns two generated regions in `results/REPORT.md`
(`report-s44-arm-table`, `report-s44-answers`) and one receipt,
`results/stage2/s44-arm-table-20261117T081002Z.json` (stamp
`20261117T081002Z`, revision `35b34f72cf513bd7f97e744fe299f5c18fde63e7`). It
re-reads every raw lm-evaluation-harness sample file, pairs two arms by
`(subject, doc_id)` only when their doc-hash sequences match, and reports a
paired percentile bootstrap on the item-level difference plus a McNemar test on
the discordant items.

Two things this found that the family means alone hid:

- **the adapter's knowledge "gain" is a trade inside the family.** `adapter-2m`
  gains on MMLU and loses on MMLU-Pro by amounts of the same size; the family
  means are a tie. The paired intervals separate the two members cleanly, which
  the family mean cannot.
- **the memory arm's held-out CE cost is tiny relative to its anchored math
  loss.** Its held-out CE penalty is a small fraction of a nat while its GSM8K
  and MATH-500 losses are several points. The mechanism is not degrading the
  local next-token distribution much; it is losing the tokens that decide an
  answer.

Item extraction is streaming. A whole-file `json.load` of one MMLU raw receipt
measured 11.7 GiB resident — not a workload-sized limit for reading four fields
per item — so the extractor decodes one sample at a time over a buffered string.
The whole script measured a 0.45 GiB cgroup peak and now runs at a 2 GiB limit
(measured + 1 GiB, rounded). The earlier 4 GiB run was over-provisioned; both
receipts are kept.

A defect this exposed and fixed: `results/stage2/eval/probe-base/eval.json`
declares `arm = base`, and `probe-memory-graft` declares `arm = memory`. Keying
arms by the payload field silently overwrote the real `base` scores with a probe
run. The table keys arms by the receipt directory and excludes `probe-*`.

### The direct answers

The generated block in `results/REPORT.md` carries the numbers and intervals.
In words: no trained arm beats the frozen base on math; the only real gains are
`adapter-2m` on MMLU (and the matching MMLU-Pro loss) and `adapter-20m` losing
less knowledge than `adapter-2m` while also losing less MMLU; the memory arm is
below base on both families with its worst losses on GSM8K and MATH-500; and
`memory-ungrafted` is bit-identical to base on all four tasks, which is the
harness-level confirmation of the gate's zero initialisation. `memory-grafted`
(the S40 reconstruction init) is a collapse, not a regression — the representation
graft's measured hidden-state perturbation predicted exactly that.

### Corrective arms: entry points, registration, tests

The training entry points already existed in `oracle/frontier/train.py`
(`--arm corrective --objective correction --advantage-weighting kl|signed|both`);
what was missing was everything around them:

- `scripts/s44_corrective_math.sh` launches one weighting on the S43 math stream.
  It reads the arm, stream hash, budget, seed and objective weights **from the
  registration**, fails closed if the stream hash drifts, requires the corrective
  init (with the exact `graft.py --corrective` command if it is missing), requires
  an explicit `SELVEDGE_MEM`, and refuses to start while the bar run is active.
  `--dry-run` prints the exact command without launching.
- `scripts/s40_compare.py` now scores the three corrective arms, so the finished
  checkpoints go through the same anchored instrument as every other arm; without
  that, a stage-2b win could not be measured on the basis the bar is measured on.
- `docs/ws19-stage2b-corrective-advantage-preregistration-v1.json` registers the
  three 20M arms on the bar's stream and teacher, one seed each, the S43 optimiser
  verbatim, the win bar (beat the full arm **and** `adapter-2m` on the target
  family without buying it with a held-out or knowledge regression), a four-branch
  decision rule, the next route per branch, and the VRAM/container arithmetic.
- `oracle/tests/test_s44_stage2b.py` (12 tests) covers: the 539,649 active union
  against the 537,600 memory union (a 2,049 per-site gate), zero-init bit
  identity, three-step training for each weighting with a non-zero gate and
  gradients on the gate, rows, key and value projections, the signed advantage's
  sign, bit-identical determinism for a fixed seed, the `--objective correction`
  fail-closed wiring, the launcher's printed command against the registration,
  and the scoring entry point covering every registered arm.

The first step of a corrective run has a gate of exactly zero, so only the gate's
own parameters receive gradient on that step; the rows, key and value
projections move from the second step. The test asserts this explicitly rather
than pretending all parameters move immediately.

Court: `oracle` with the S42/S43/S40 suites — **85 passed** (S44 + S42 corrective
+ S43 token gate + S40 compare/full arm + stage-2 memory/graft).

### Launch preparation (nothing launched)

`scripts/s44_stage2b_queue.sh --plan` prints the exact sequence; `--execute`
will refuse unless the bar unit is inactive. The sizing arithmetic is written in
the registration: the corrective arm has the memory arm's shape (student +
teacher + frozen backbone residual), measured at 9.211 GiB allocated / 9.363 GiB
reserved, against the full backbone arm's 11.429 / 11.672 GiB under a 12.5 GiB
limit; the gate adds 2,049 parameters and one (B, T) float per site, so the
limit is not expected to bind. The ladder if the smoke says otherwise is
sequence 512→384 with accumulation 4→5 under a new amendment — not a silent
change. The container limit comes from the corrective arm's own 20-step smoke
(1.5x the cgroup peak), never from the bar's 10 GiB.

### The bar's progress

Reading `results/stage2/runs/full-20m-math/log.jsonl` at 08:16Z: the run had
logged step 784, 1,605,632 tokens, at 1,846.7 tok/s with a peak of 11.429 GiB
allocated under the 12.5 GiB limit. That is about 8% of the 9,765 steps and an
ETA near 11:00Z at the steady rate. The bar has no anchored receipt, so its row
is absent from the table rather than estimated.

### Literature

`docs/S44-LITERATURE.md` records four focused questions with URL, fetch date,
stated claim and verdict. The strongest new finding: the 2026 on-policy
distillation line (AOPD `2610.31563`, asymmetric OPD `2605.06387`, SKALD
`2608.09826`) has converged on advantage-weighted supervision **plus** a
region-conditioned divergence/preservation term, which is the two-part design
S43 registered from a different starting point. The exact statistic — a signed
log-probability advantage on a supervised token, off-policy, into a frozen-base
residual — remains **UNSUPPORTED** as a published construction, and the
math-mixture setting (57% math, 20M distillation tokens) remains **PARTIAL**.

### Courts

`commit_identity`, `perf_numbers` (1,008 blocks), `status_consistency` (3 blocks,
9 contradiction rules) and `prereg_provenance` (19 registrations) passed; the new
registration is ordered before any score of it.

### What this entry does not claim

- **No bar verdict.** The 20M run is in flight and unscoreable.
- **No mechanism win.** Every arm in the table predates the token gate and the
  signed advantage; the corrective arms are registered and untrained.
- **No seed-variance claim.** The item-level intervals are sampling intervals
  over items, not over training seeds; each arm is one seed.
- **No claim that the corrective arms will fit in VRAM.** The arithmetic says
  they should; the smoke is what decides, and the ladder is registered.

## 2026-11-17 — S45: the generative-math collapse diagnosed, and the RFT/on-policy route registered

### Plan

Three jobs, CPU only: (1) diagnose why `memory-2m` loses 9.55 GSM8K points while
its held-out CE is only +0.0203 nats; (2) build and register the next route
(verified-solution SFT, on-policy KD, both) with an exact verifier, a
contamination check, a stream builder and a longer-budget plan; (3) prepare the
objective ablation (the registered preservation and signed-advantage terms plus
a target-side math-format curriculum). The 20M math-weighted full-distillation
bar was training throughout and was not touched; no GPU job was started.

### The collapse is a persistent bias, not a single bad token

`scripts/s45_collapse_diagnosis.py` has two modes. `--artifacts` re-reads the
stored lm-evaluation-harness samples and pairs by `doc_id`, requiring the prompt
hash to match; `--forward` teacher-forces a bounded set of prompts through base,
`memory-2m`, `adapter-2m` and the 2B teacher on CPU, then `--analyse-forward`
classifies the stored per-position arrays. No number below is typed by hand.

The artifact receipt is `results/stage2/s45-collapse-artifacts-20261117T082147Z.json`
(stamp `20261117T082147Z`, revision `2612c6e49c441dc12740597f6878a36ce4e6fe5c`).
The direct answers:

- `memory-2m` on GSM8K: 208 base-correct items become wrong, 82 base-wrong items
  become correct (net -126, -9.55 points). 203 of the 208 losses change the
  extracted final answer.
- every one of the 208 losses first differs in the **reasoning** region, not at
  the `####` marker; 40 first differences are on a digit and 20 on an operator.
- the loss is length-dependent: -0.0270 in the shortest base-response quartile,
  then -0.1295 / -0.1131 / -0.1131.
- the 100 byte-identical responses have exactly +0.0000 delta; all damage is in
  the items whose trajectory changed.
- MATH-500: 53 base-correct losses, 22 gains, 25.00 -> 18.80 (-6.20).
- `adapter-2m` loses 126 and gains 88; `adapter-20m` loses 139 and gains 97. More
  tokens did not fix the direction.

The forward receipt is
`results/stage2/s45-forward-positional-20261117T082412Z.json`; the analysis is
`results/stage2/s45-forward-analysis-20261117T082511Z.json` (stamp
`20261117T082511Z`, same revision). On the correct (base) solution path the
memory residual adds +0.0373 nats/token over 80.0% of positions, +0.276 nats in
the first ten tokens and +13.220 nats total over five sequences. The per-token
rate does not grow with position (Pearson near zero): it is a persistent sign.
The gate is a near-constant |alpha| = 0.471 (range 0.36-0.50) and the residual
norm is 0.768; neither correlates with the per-token damage. By token class on
the correct path: letter +0.557, space +0.060, digit +0.003, operator -0.010
nats of class-mean CE delta. The residual moves prose, and the prose change
flips the answer on some items.

This is the first measurement in the program that explains the
CE-versus-accuracy dissociation instead of restating it: the local distribution
is only slightly worse, but the sign is positive almost everywhere, so the
cumulative cost over a 60-80-token solution is more than ten nats.

### The next route is built and registered

`docs/ws19-onpolicy-rft-preregistration-v1.json` registers:

- data: GSM8K-train (7,473) and MATH-train (7,500), prompts from the train
  splits only, answers extracted by the registered verifier;
- sampling: teacher Qwen3.5-2B or the frozen 0.8B student, K=8, temperature 0.8,
  top-p 0.95, 512 new tokens, seed 451; one pass is bounded by 61,329,408
  generated tokens;
- verifier: exactly one unambiguous answer; numeric equality at 1e-9 relative,
  otherwise a documented canonical LaTeX string; ambiguity (two `####`, two
  `\boxed{}`, or a boxed `or`) is rejected;
- rejection/dedup: keep only `correct`, deduplicate within a prompt by
  whitespace-normalised solution text, record verified/sampled and kept/verified;
- contamination: 5-word shingle Jaccard against GSM8K-test and MATH-500,
  discard >= 0.2, fail closed >= 0.5;
- arms: four 2M corrective-screen arms (`rft-sft`, `rft-kd`, `rft-kd-signed`,
  `rft-both`) and three 20M full-backbone arms (`rft-sft-20m-full`,
  `rft-kd-20m-full`, `rft-both-20m-full`), plus an optional format-curriculum
  arm; seed 0 per training arm, the bar's Adam8bit/cosine/1e-4 settings;
- eval: the anchored harness on all 14,042 MMLU / 504 MMLU-Pro / 1,319 GSM8K /
  500 MATH-500 items, held-out CE/KL on the 3,906 chunks, paired per-item
  bootstrap and McNemar;
- win bar: beat base **and** the measured full-20m-math bar on the math family,
  with no held-out or knowledge regression, and a seven-branch decision rule.

The implementation is `oracle/frontier/rft.py` (verifier, rejection sampling,
acceptance receipt, contamination check, teacher/student sampler),
`scripts/s45_rft_stream.py` (deterministic accepted-solution stream),
`oracle/frontier/rft_objective.py` (SFT CE, CE+KL, the registered
preservation/signed/gate terms, and the weighted SFT+KD combination), and the
`--objective sft` path in `oracle/frontier/train.py` (verified-solution CE with
`kl_weight 0`, fail-closed validation). The fixture run in
`results/stage2/s45-rft-fixture/receipt.json` samples 10, verifies 4, keeps 3,
with a 0.0200 maximum contamination Jaccard. The stream builder produced a
50,000-token fixture stream from a 121-token corpus in 414 epochs, with the
receipt hash checked `results/stage2/s45-rft-stream-20261117T083022Z.json`.

`oracle/tests/test_s45_rft.py` is 30 CPU tests: correct/wrong/ambiguous/
multiple-answer/LaTeX/numeric verifier cases, the curriculum (including that a
formatted solution still verifies), rejection and dedup arithmetic, the
contamination fail-closed path, the SFT/KD/correction objective arithmetic, and
the `--objective sft` CLI wiring. The pre-existing S40/S42/S43/S44 suites also
pass with the memory instrumentation added (66 tests).

### Longer-budget arithmetic

The receipt is `results/stage2/s45-budget-plan-20261117T082853Z.json` (stamp
`20261117T082853Z`, revision
`2612c6e49c441dc12740597f6878a36ce4e6fe5c`). It reads the in-flight bar's
`log.jsonl` and uses the median of its last 50 lines: **1,836.0 tok/s**. At
2,048 tokens per step, 20M is 9,766 steps / 3.026 h, 40M is 19,532 / 6.052 h,
60M is 29,297 / 9.077 h, and 100M is 48,829 / 15.129 h. One 100M run with
checkpoints at 20/40/60/100M draws the whole curve; the 60M run is the cheaper
first extension. The measured checkpoint is 2,008,524,311 bytes; 60M milestones
are 5.61 GiB, 100M milestones 7.48 GiB (7.48 / 9.35 GiB with the newest/latest
copy). The teacher-sampling pass is bounded at 61.3M generated tokens; the
acceptance rate is unmeasured until the job runs.

### Literature

`docs/S45-LITERATURE.md` records URL, fetch date, stated claim and verdict.
The strongest new sources are CoRD (`2611.07959`: OPD gains on five math
benchmarks with Qwen3/Qwen3.5 0.6B-4B students), SD-OPD (`2611.07722`: style
positions carry 3.5x the teacher-student log-probability difference of math
positions, the same shape as the class split measured here), RFT (`2308.01825`:
LLaMA-7B GSM8K 49.3% versus 35.9% and larger gains for weaker models), and
`2610.07405` (RFT at 1.5B can lose hard-problem coverage and diversity). The
exact 20M-token budget is **UNSUPPORTED**: an arXiv API query for `"20M tokens"
AND distillation` returned zero records on 2026-11-17.

### The bar's progress

At 08:26Z the in-flight log (`results/stage2/runs/full-20m-math/log.jsonl`) had
logged step 1,299 and 2,660,352 tokens, about 13% of the 9,765 steps. The S43
launch receipt (`results/stage2/runs/full-20m-math/launch.json`, revision
`78caa6a1ae96b8c50094f0bdaff6691695777ebe`) recorded 1,847.1 tok/s steady and an
ETA near 11:02Z. There is no anchored receipt and therefore no bar score or bar
row; nothing was launched or scored in this continuation.

### Courts

`commit_identity`, `perf_numbers`, `status_consistency` and
`prereg_provenance` are run after the commit; the registration is ordered
before any score of it. The `oracle` court ran the new S45 tests and the
existing S40/S42/S43/S44 suites; the diagnostic and budget scripts ran in the
`python` court at measured limits (2-4 GiB), and the CPU forward pass ran in the
`oracle` court at 10 GiB, 1.58x its measured 6.35 GiB process peak.

### What this entry does not claim

- **No RFT or on-policy result.** No GPU generation, no training, no acceptance
  rate.
- **No bar score.** The 20M run is in flight; its row is absent, not estimated.
- **No mechanism win.** The diagnosis is a measured bias; the route is
  registered, not run.
- **No seed variance.** Five prompt/sequence pairs on one checkpoint per arm.
- **No symbolic verifier.** The extractor is deliberately conservative; equal
  but differently written MATH answers are rejected.

## 2026-11-17 — S46: the gate headroom is bounded, and a cheap gate does not reach it

S45 measured a persistent, near-uniform damage sign. The question that decides
the next mechanism is different: *could any gate recover it*, and does a gate
that is cheap at generation time recover any of it. This continuation answers
that with one CPU forward receipt and three derived upper bounds, then checks the
answer level with bounded greedy decoding.

### The probe crash, and what was actually wrong

`s46_oracle_gate.py --forward` crashed with `KeyError: 'target'` at the receipt
append. The cause was an input-schema mismatch, not a model bug: `s45._select_items`
returns records with `prompt`, `base_response`, `memory_response`, `base_ok`,
`memory_ok` and `prompt_tokens`, and never a `target`; the forward receipt asked
for `item["target"]`. The committed fix joins the target from the same GSM8K raw
sample by `doc_id` in `_sample` before the forward loop, so the receipt carries
the reference answer the answer-level verifier needs. The earlier court log
(`results/courts/logs/oracle-20261117T084221Z-9a4cbe.log`) keeps the traceback.

Two further defects were found while running the probe, both in the generation
path that had never executed:

- the causal class policy read `all_ids[absolute - 1]`, i.e. the token *before*
  the gated hidden position. The residual at position `t` shapes the prediction
  of token `t+1`, so the causal class is the class of token `t`. Fixed in
  `_gate_tensor` and pinned by
  `test_class_gate_reads_the_token_at_the_gated_position`.
- the forward receipt stores both response sources for each prompt, and the
  generation loop decoded every item, so each prompt was decoded twice for no
  reason. The loop now deduplicates by `doc_id` and accepts `--max-prompts`; a
  `decode.prompts` / `decode.prompt_cap` pair is recorded in the receipt.

The point values did not change for the bounds: `--bounds` reads the stored
per-position arrays.

### The bounds

Sample: 16 GSM8K prompts where the base is correct and `memory-2m` is wrong, plus
4 controls excluded from the bounds; 2,176 completion positions; CPU forwards of
the base and memory branches plus the 2B teacher. Receipt
`results/stage2/s46-oracle-bounds-20261117T090710Z.json` (stamp
`20261117T090710Z`), summarised with prompt-clustered intervals by
`scripts/s46_headroom_summary.py` into
`results/stage2/s46-headroom-summary-20261117T092225Z.json`.

- CE: all-off 0.26616, all-on 0.28499, net damage +0.01883 nats/token.
- Per-token oracle (non-causal): 0.23773, i.e. +0.02843 nats/token better than
  disabling the memory, 2.51x the net damage, prompt-clustered 95% CI
  [+0.02128, +0.03599].
- KL: damage +0.01718, oracle gain +0.01455 [+0.00937, +0.02162].
- Class gate, held out: 0.2666 pooled; on both folds it lands within 0.0008 nats
  of memory-off (0.2468 vs 0.2468; 0.2868 vs 0.2860), and the fitted class set
  is `digit` in one fold and `operator` in the other.
- Cheap one-threshold gates, held out: entropy 0.2662, residual norm 0.2683,
  position 0.2657, i.e. gains of -0.40 [-2.02, +1.17], -0.01, -2.12 [-4.39,
  -0.07] and +0.51 [-1.45, +2.88] millinats per token. None lies above zero.

The reading: headroom exists and is worth about 28 millinats of CE per token on
this sample, but it lives in the per-position choice. A single threshold on any
cheap feature measured here recovers the damage by turning the memory off and
captures none of the oracle's gain. A learned gate must therefore beat
memory-off on held-out prompts, not merely reproduce it.

### The answer level

`results/stage2/s46-oracle-generation-20261117T090913Z.json` (stamp
`20261117T090913Z`) decodes the same 16 prompts greedily
under five policies, 96 new tokens at most, out-of-fold gate parameters, graded
by the RFT GSM8K verifier: base 4/16, memory 2/16, class 6/16, entropy 6/16,
residual norm 4/16. Paired against base: class +2/-0, entropy +2/-0, memory
+1/-3, residual norm +1/-1. With 16 selected prompts this is a direction, not a
result; the exact p-values are 0.5 or larger. The decoding budget (96 tokens,
320-token prompt window) is not the anchored protocol, so these accuracies do not
reproduce the stored `base_ok` labels; they are comparable to each other and to
nothing else.

### Cost of the probe

The forward pass is CPU-only (131.8 s, 2,668 tokens, 8.1 GiB process RSS peak,
3.02 GiB cgroup peak, 12 GiB court limit). The generation pass is also CPU-only
(675.2 s, 7,680 decode tokens, 87.9 ms/token) and ran detached at 6 CPUs while
the 20M bar held the GPU. It cost the bar throughput: the bar's own log fell from
about 1,840 to 1,550-1,700 tok/s for the 11 minutes the two jobs overlapped, i.e.
roughly 2-3 minutes of bar wall time. The generation could not use the GPU while
the bar ran and was needed for the report, so the contention was accepted and is
recorded here rather than hidden.

### Courts

`oracle` ran the S46 suite plus the S43 total-cap and S47 matched-control and
correctness suites; `python` ran the summary self-test and every regeneration.
The four S43/S46/S47 failing test logs from the first attempt are kept in
`results/courts/logs/` with their fixes.
