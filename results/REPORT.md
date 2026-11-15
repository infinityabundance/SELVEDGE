# SELVEDGE results — honest current state

Status: **M2 complete; M3 part 1 (storage) measured; the T2 trained branch is
running.** The pre-registered M1 frontier and the M2 per-token
selector/oracle experiment have been measured. M1 rejected static whole-group
omission. M2 rejects the inference-only, post-hoc per-token selection branch:
even a causally unavailable per-token oracle at the compliant budget is far
outside the registered usable tier, and no causally available selector beats
activation magnitude at matched realized active parameters. The reports are
`results/M1-FRONTIER.md` and `results/M2-SELECTORS.md`; the registered contract
is `docs/EXPERIMENT-CONTRACT.md` (M1 and M2 sections, amendments A3 and A4).
The correction of record for physical accounting is `docs/M2-CORRECTION.md`.
The first *trained* branch — the T2 five-arm capacity matrix — is launched and
in progress; see "T2, in progress" below. Its result is not claimed here.

This file states what is established, with the receipt that establishes it, and
what is not. It is not a benchmark report: no latency, throughput or quality
number has been measured, and none is claimed.

## Established

1. **A pinned teacher oracle exists and runs offline after one documented fetch
   step.** The oracle court image pins Python, torch (CUDA 13.2 build),
   transformers, tokenizers, safetensors and numpy. A real CUDA matmul and SiLU
   ran inside the container under device passthrough; the probe and receipt record
   the device and the `--gpus all` fallback reason.
2. **Two teachers are acquired, licensed and hash-pinned.** Qwen3-0.6B and
   Qwen3-1.7B at fixed commit SHAs, Apache-2.0, with every file's SHA-256 and
   size in `models/MODELS.lock.json`. The license text is reproduced verbatim in
   `docs/MODELS.md`.
3. **Exact accounting on real safetensors.** `selvedge inspect-model` reads the
   real headers, rejects unclassified tensors, verifies the embedding/output tie
   byte-for-byte, and reports component counts, the always-active backbone,
   removable FFN, ceiling arithmetic and FLOP estimates.
4. **The 1e9 ceiling is not binding for Qwen3-0.6B and is binding for
   Qwen3-1.7B.** Qwen3-4B's backbone alone exceeds the ceiling, so it cannot be
   a compliant teacher; it is deliberately not downloaded.
5. **A real checkpoint converts to group shards and reconstructs exactly on real
   activations.** The default grouping is sixteen contiguous groups per layer,
   stored as f32; the shard count and payload bytes are in each receipt. The
   grouped-from-bundle path and the dense-from-checkpoint path agree bitwise on
   every compared component.
6. **The f32-equivalent teacher computation is reproduced within a documented
   bound**, and the native bf16 output is reproduced when the teacher's rounding
   sequence is emulated explicitly. Both checks pass with zero violations.
7. **The masked-MLP instrument is exact on the measured paths.** With every
   group selected it is bitwise equal to the unmodified dense f32 teacher on
   real held-out windows. An independent Rust grouped sum over the published
   bundle agrees with the Python grouped reference on a real layer and a real
   four-group subset within the registered f32 bound.
8. **The quality-versus-active-parameter frontier is measured, with receipts.**
   Static uniform, magnitude, frequency, whole-layer and greedy allocations
   were evaluated on a pinned, hash-split held-out set with paired bootstrap
   CIs. Every arm at or below the ceiling is far outside the registered
   acceptable and usable tiers. The curve is a cliff near the fully dense end.
9. **The static result is negative and reproducible.** Two independent complete
   runs produced bitwise-identical per-arm aggregates. Failed attempts,
   amendments and the dirty-worktree superseded receipts are retained.
10. **The per-token oracle potential is measured and negative.** A causally
    unavailable oracle that chooses an exact local subset per token and per
    layer, at the same realized budget, was evaluated at 16, 64 and 4 groups.
    The 64-group oracle is the best and is still far outside the registered
    usable tier. A bounded exact-KL refinement does not rescue it. Its bound is
    about logical omission quality, not a compliant runtime.
11. **Per-token selection helps logically but cannot escape the cliff, and the
    best arms are physically non-compliant.** At matched realized active
    parameters, per-token activation magnitude beats the best static M1 arm, and
    every other causally available per-token arm is worse than per-token
    magnitude. The learned unary, learned interaction-aware,
    gradient-attribution and corrected-attribution arms are all worse. Corrected
    accounting shows that the magnitude, frequency and learned selectors read all
    gate/up channels to score groups, so they cannot be executed under the
    ceiling; only static/random masks executed through the sliced sparse path are
    physically compliant, and they are far worse. See `docs/M2-CORRECTION.md`.
12. **The Rust Atlas boundary is implemented and parity-tested.**
    `selvedge-atlas` owns the quality-observation record, the interaction
    graph, the leakage-bounded selector input, seven selectors, the two-view
    head contract and independent matched-budget accounting. `selvedge atlas
    fit|select|compare|cost` run; the Python/Rust parity, leakage, graph and
    calibration tests pass on exported real fixtures.
13. **Exactness and the two verdicts are type-level, not prose.**
    `ExactSelection` requires a `NativeSetWitness` from full set equality
    against a declared `NativeComputationSet`; the old string reclassification
    is gone, and compile-fail cases prove the barrier. `CausalAvailability` and
    `PhysicalCompliance` are separate evidence-bearing records; there is no
    `deployable` boolean. `ParameterLedger::admit_selection_path` checks the
    union of selector and execution parameters before execution. See
    `docs/ARCHITECTURE.md` and `docs/M2-CORRECTION.md`.
14. **The tiered store and its exact-mode reads are measured.** `selvedge-store`
    implements the host-RAM cache policies, scoped reads, advisory prefetch and
    the `prefetch_bytes_wasted` metric. Cold/warm whole-bundle reads on the real
    0.6B and 1.7B bundles, the page-cache control, read-size, queue-depth,
    alignment, checksum and cache-residency sweeps are three-repeat measurements
    with ranges and CVs in `docs/STORAGE.md` and the `results/store` and
    `results/io` summaries.

## Numbers (generated from receipts)

- Qwen3-0.6B unique learned parameters:
  <!-- BEGIN GENERATED: report-06b-unique -->
596049920
  <!-- END GENERATED: report-06b-unique -->
- Qwen3-1.7B always-active backbone:
  <!-- BEGIN GENERATED: report-17b-backbone -->
663610368
  <!-- END GENERATED: report-17b-backbone -->
- Qwen3-1.7B affordable FFN elements per token:
  <!-- BEGIN GENERATED: report-17b-max-ffn -->
336389632
  <!-- END GENERATED: report-17b-max-ffn -->
- Qwen3-4B backbone (config projection, weights not downloaded):
  <!-- BEGIN GENERATED: report-4b-backbone -->
1332870656
  <!-- END GENERATED: report-4b-backbone -->
- Qwen3-0.6B grouped f32 vs teacher f32 max abs:
  <!-- BEGIN GENERATED: report-06b-f32-max-abs -->
0.005859375
  <!-- END GENERATED: report-06b-f32-max-abs -->
- Qwen3-1.7B grouped bf16-emulated vs teacher bf16 max abs:
  <!-- BEGIN GENERATED: report-17b-bf16-max-abs -->
4.0
  <!-- END GENERATED: report-17b-bf16-max-abs -->
- Qwen3-0.6B f32 bundle payload bytes:
  <!-- BEGIN GENERATED: report-06b-payload -->
1056964608
  <!-- END GENERATED: report-06b-payload -->
- Qwen3-1.7B f32 bundle payload bytes:
  <!-- BEGIN GENERATED: report-17b-payload -->
4227858432
  <!-- END GENERATED: report-17b-payload -->
- Best logical-selection arm (greedy-uniform5, dense masked execution)
  active unique parameters:
  <!-- BEGIN GENERATED: report-best-active -->
998630400
  <!-- END GENERATED: report-best-active -->
- Best logical-selection arm mean teacher KL, nats/token:
  <!-- BEGIN GENERATED: report-best-kl -->
7.706657
  <!-- END GENERATED: report-best-kl -->
- Best logical-selection arm PPL ratio:
  <!-- BEGIN GENERATED: report-best-ppl -->
1196.792
  <!-- END GENERATED: report-best-ppl -->
- Uniform magnitude k15 (noncompliant) mean teacher KL:
  <!-- BEGIN GENERATED: report-k15-kl -->
0.279397
  <!-- END GENERATED: report-k15-kl -->
- Registered acceptable tier KL bound (nats/token):
  <!-- BEGIN GENERATED: report-acceptable-kl -->
0.05
  <!-- END GENERATED: report-acceptable-kl -->

## T2, in progress

The first trained branch is registered in `docs/EXPERIMENT-CONTRACT.md` T2 (with
amendments A8 and A9). Five arms — a dense-adapter control `D`, the hashed
memory path `M`, routed specialists `X`, a naive hybrid `H` and an
error-directed hybrid `H*` — train from one frozen Qwen3-0.6B core, on one data
order, at one token budget and one seed, and are then evaluated on the full
registered suite. The decision rule K15-K19 is fixed in the contract and applied
mechanically by `oracle/eval/t2_decision.py`.

Where the evidence lives: `results/t2/pilot-state.json` (phase status,
timestamps, exit codes, per-arm argv and hashes), `results/t2/<arm>/run.json`
(git SHA, config hash, seed, data-order hash, admission record, throughput,
VRAM, wall-clock, checkpoint hash), `results/eval/t2-*.json` (the registered
harness) and `results/T2-PILOT.md` (the published report, generated from the
decision receipt by `scripts/t2_report.py`).

**No T2 quality claim is made in this file while the pilot is unfinished.** A
partial pilot is not a result: the arms that have finished are not interpreted
before the ones that have not, and the verdict appears only in
`results/T2-PILOT.md`.

## Not established

- **No capability claim.** The frontier measures next-token distributions on
  pinned text. KL on Wikipedia is not a code, math, reasoning or factuality
  benchmark. See `docs/LIMITATIONS.md`.
- **No trained or upcycled model is evaluated in this report.** The M2 result
  bounds post-hoc inference-only selection, not sparsity in general. A model
  trained or upcycled to tolerate the sparse FFN is the T2 branch, which is
  running; a distilled smaller teacher, quantization and a learned memory layer
  at scale remain untested.
- **No end-to-end model latency and no hardware-counter FLOPs.** FLOP values
  are derived from realized channel counts, not hardware counters. Arm wall
  clock is a cost observation, not a benchmark. M3 part 1 does measure
  host-RAM/NVMe storage throughput and per-shard latency, repeated three times
  with variance reported (`docs/STORAGE.md`); those are storage baselines, not
  model speed, and this report contains no GPU transfer number.
- **No scheduler or runtime.** `selvedge-atlas` and `selvedge-store` exist;
  `selvedge-sim`, `selvedge-runtime` and GPU staging do not. The Atlas cost head
  is explicitly unavailable until the M3 trace scheduler has real traces, and
  no transfer time is assumed. The storage measurements are host-RAM/NVMe
  only, not GPU transfer numbers.
- **The native-bf16 match is a model of PyTorch's arithmetic on these inputs.**
  A different kernel or backend may round differently.

## Artifacts

The activation files, bundles, dataset cache and derived split live under the
gitignored `results/artifacts/` directory; the tracked receipts carry their
hashes, sizes and measurements. Rebuilding is documented in
`docs/BUDGET-MATH.md`, `docs/DATASETS.md` and `docs/WORK-LOG.md`. The raw
storage JSONL is excluded from git; `scripts/storage-summary.py` regenerates the
tracked variance summaries from the receipts.

## Next

Move to the trained/upcycled branch with a fresh pre-registration: a dense
compliant Qwen3-0.6B base plus external capacity, an UpIT/SIMoE-style
dense-to-sparse upcycle of Qwen3-1.7B trained to tolerate the compliant group
budget, and a distilled smaller teacher, matched for active parameters, FLOPs
and training tokens. The M2 oracle bound says post-hoc selection cannot close
the gap; the next question is whether training can change the model so the same
budget is usable. Do not resume inference-only selector tuning without a new
oracle bound.
