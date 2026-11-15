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
