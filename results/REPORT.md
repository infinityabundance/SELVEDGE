# SELVEDGE results — honest current state

Status: **M2 complete; M3 part 1 (storage) measured; M3 part 2 (trace simulator)
implemented and validated on held-out arms; the T2 trained branch is
running.** The pre-registered M1 frontier and the M2 per-token
selector/oracle experiment have been measured. M1 rejected static whole-group
omission. M2 rejects the inference-only, post-hoc per-token selection branch:
even a causally unavailable per-token oracle at the compliant budget is far
outside the registered usable tier, and no causally available selector beats
activation magnitude at matched realized active parameters. The reports are
`results/M1-FRONTIER.md` and `results/M2-SELECTORS.md`; the registered contract
is `docs/EXPERIMENT-CONTRACT.md` (M1 and M2 sections, amendments A3 and A4).
The correction of record for physical accounting is `docs/M2-CORRECTION.md`.
The first *trained* branch — the T2 five-arm capacity matrix — is running; see
"T2, in progress" below. Its result is not claimed here. The P12.1 audit of the
routed-overflow mask found a real defect in the dispatcher that produced every
routed arm; it is corrected behind a default-off flag so the registered
comparison is unchanged, and it is reported separately in
`docs/P12.1-DISPATCH-AUDIT.md`.

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
15. **A trace-driven execution simulator exists, is calibrated from those
    measurements, and is validated on held-out arms.** `selvedge-sim` owns a
    versioned execution trace that can only describe what was selected, an
    empirical cost model (with exponential-versus-lognormal residual evidence),
    a discrete-event engine with coalescing, backpressure, cancellation,
    deadlines, execution envelopes and a two-timescale working set, and a
    placement study whose layouts are ranked by the simulator. Calibrated on
    `plan-order` and `lru-1of4`, it replayed three other 0.6B arms and one 1.7B
    arm with **zero hit/miss mismatches and an exact device-byte prediction**;
    the stall discrepancies are in `docs/SIMULATOR.md`. The live
    simulator-versus-store comparison agreed exactly on bytes and hit counts and
    agreed on the p95/p50 shape, but its absolute latency check failed because
    the host was saturated by the T2 pilot; that is recorded as inconclusive, not
    as a result. No GPU transfer is simulated, and no trained-router trace
exists, so all locality conclusions are conditional on the trained model's
    routing statistics.

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

### Pilot state, recorded read-only

As of the last poll before this commit (`scripts/poll-t2-pilot.sh`,
`results/t2/pilot-state.json`):

| arm | state | evidence present |
| --- | --- | --- |
| `D` (dense control) | **completed** — trained the registered budget and evaluated | `results/t2/D/run.json`, `results/t2/D/eval.json`, `results/eval/t2-D.json` |
| `M` (memory only) | **completed** — trained the registered budget and evaluated | `results/t2/M/run.json`, `results/t2/M/eval.json`, `results/eval/t2-M.json` |
| `X` (specialists) | **completed** — trained the registered budget and evaluated | `results/t2/X/run.json`, `results/t2/X/eval.json`, `results/eval/t2-X.json` |
| `H` (naive hybrid) | **completed** — trained the registered budget and evaluated; the evaluation was re-run under the corrected instrument (below) | `results/t2/H/run.json`, `results/t2/H/eval.json`, `results/eval/t2-H.json`, `results/t2/H/launch-eval.json` |
| `H*` (error-directed hybrid) | **pending** | — |

Four of five arms have both a training receipt and an evaluation summary. The
last hybrid has not finished, and the pilot is therefore **not separable from
"unfinished"**: no ordering between arms is stated here, deliberately — an
early ordering is exactly the reading that T2.8 and the one-seed risk in
`results/T2-PLAN.md` §5 forbid. The per-arm numbers that do exist are in
`results/t2/<arm>/eval.json` and `docs/WORK-LOG.md`; they are not interpreted.

### The evaluation instrument, corrected before the hybrid numbers were read

The first `H` evaluation died with a CUDA out-of-memory. The instrument held the
1.7B teacher, the hybrid student *and* two full f32
`positions x 151,936` logits tensors, and it loaded the 5.9 GB checkpoint
straight onto the device, which kept a second copy of every parameter resident
for the whole run.

The instrument now computes each model's hidden states once and applies the
vocabulary head a fixed number of positions at a time (the count is in the eval
receipt as `instrument.logit_chunk_positions`), reduces in f32 and frees each
chunk; it loads the checkpoint on the host and drops it after the state-dict
copy; it frees the teacher before the generation stages; and it records
per-stage VRAM peaks. The registered metric is unchanged: every position is
still scored against the whole vocabulary with the same KL, NLL, PPL and top-1
definitions. On the real `H` checkpoint the streamed instrument and a
full-width head agree to the last bit on NLL and top-1, and on KL to within the
bf16 rounding of the head itself.

Peak VRAM for the same `H` smoke subset, streamed versus one full-width head
chunk:

<!-- BEGIN GENERATED: report-s16-eval-vram-chunked -->
6.39 GiB
<!-- END GENERATED: report-s16-eval-vram-chunked -->

streamed, and

<!-- BEGIN GENERATED: report-s16-eval-vram-full -->
8.16 GiB
<!-- END GENERATED: report-s16-eval-vram-full -->

with a single full-width chunk (which only fits now that the checkpoint no
longer sits on the device). The full registered `H` evaluation on all
held-out windows took

<!-- BEGIN GENERATED: report-s16-h-eval-seconds -->
667 s
<!-- END GENERATED: report-s16-h-eval-seconds -->

at a peak of

<!-- BEGIN GENERATED: report-s16-h-eval-vram -->
6.39 GiB
<!-- END GENERATED: report-s16-h-eval-vram -->

of the 15.6 GiB usable, against a 24 GiB host limit. The receipt is
`results/eval/t2-H.json`; the arithmetic and the court receipt are in
`results/t2/H/launch-eval.json`.

While the pilot occupies the GPU, three successor courts were built and
CPU-tested without touching it: `P8.2` (bounded-gate composition, including the
missing `--run-composition` entry point), `P11.1` (causal bottleneck repair),
`T3` (latent geometry) and `P9.3` (combination utility). They are registered as
new sections of `docs/EXPERIMENT-CONTRACT.md` and their launch commands are in
`results/T2-PLAN.md` §6. None has a measurement.

Read-only monitoring is `scripts/poll-t2-pilot.sh` (one report) or `--follow`. It
reads state, per-arm logs, `docker ps` and free disk; it writes nothing to the
run. Nothing in this session started, restarted or touched any container, and no
GPU work was performed.

### What still has to happen before a verdict

1. `H*` training (its label phase runs at its warm-up step, then the router is
   frozen) and evaluation.
2. `oracle/eval/t2_decision.py` applied to the five summaries, then
   `scripts/finish-t2-pilot.sh` to render `results/T2-PILOT.md` and run the
   courts.
3. The P2.4 instrument check over the five final checkpoints.

The remaining order is fixed by `scripts/run-t2-pilot.sh`, not by this document.

### The routed-overflow audit, kept separate from the verdict

The P12.1 audit found a real defect in the shared dispatch: the capacity
**validity mask is built in expert-sorted pair order and applied after the pairs
are restored to token order**, so a pair is gated by another pair's validity
whenever any expert overflowed. The finished routed arm's own training log shows
that overflow is routine, not a corner case, and the toy reproduction shows an
overflowing pair reading another token's activation.

What this report therefore says about provenance, and nothing more:

* every routed arm (`X`, and `H`/`H*` as they finish) was trained and is being
  evaluated by a dispatcher whose overflow mask is misapplied;
* the correction is `mask_order="pair"`, **off by default**, recorded in every
  run receipt, so no registered arm, threshold or comparison has changed;
* the correction, its reproduction, the measured overflow rate and the list of
affected receipts are in `docs/P12.1-DISPATCH-AUDIT.md` and
  `results/t2/dispatch-audit-20261115T234953Z.json`, and the follow-up runs that
  measure whether the verdict moved are registered as P12.1a.

No T2 quality number appears in this section, because the pilot is unfinished and
the audit is not a verdict either: it is a statement about which code produced
which receipt.

## Not established

- **No capability claim.** The frontier measures next-token distributions on
  pinned text. KL on Wikipedia is not a code, math, reasoning or factuality
  benchmark. See `docs/LIMITATIONS.md`.
- **Not: a complete T2 verdict, or any frontier advantage.** Four of the five
  trained arms (`D`, `M`, `X`, `H`) have finished training *and* been evaluated on
  the registered T2 instrument; `H*` is training. What does **not** exist is a
  T2 verdict (it requires all five summaries and `oracle/eval/t2_decision.py`),
  and no arm has been evaluated on the fresh Rust capability suite, so there is no
  capability number and no frontier advantage in either the likelihood or the
  correctness sense. The M2 result still bounds post-hoc inference-only selection,
  not sparsity in general: a model trained or upcycled to tolerate the sparse FFN
  is the T2 branch, and a distilled smaller teacher, quantisation and a learned
  memory layer at scale remain untested.
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

## Capability court and the fresh Rust suite (S14–S26, CPU-only)

The instrument every frontier claim will use now exists and is larger.
`docs/CAPABILITY-COURT.md` is the reference. The current evidence, at revisions `2beba67` and
`5e5543f`, is:

- `results/capability/selftest-20261116T090000Z.json` — every checker outcome
  over the scored suite and the trio, including the naive-fix and sandbox-probe
  outcomes (counts in `docs/CAPABILITY-COURT.md` §2);
- `results/capability/sealed-score-20261116T090100Z.json` — the sealed set's
  instrument self-check behind its unlock record;
- `results/capability/contamination-20261116T090400Z.json` — disjointness over
  every authored item;
- `results/capability/similarity-20261116T090600Z.json` — the near-duplicate
  screen, with its counts in the receipt;
- `results/capability/power-20261116T090700Z.json` — the power arithmetic at the
  registered n.

Established, on CPU:

- a fresh executable Rust suite with hidden checkers, reference solutions and
  declared wrong mutations, materialised from an authoring source that fails
  closed on drift, licensed MIT OR Apache-2.0;
- an anti-superficial-specialisation trio (lifetime chain, numeric dependency,
  quantifier chain) annotated with one shared operation and judged per surface;
- a bounded checker sandbox that records pass, assertion failure, compile error,
  timeout and allocation failure as distinct outcomes, with the limits, digests,
  cgroup OOM counters and workdir size in the receipt;
- a staged generate/check harness with paired category-stratified confidence
  intervals, per-category reporting, failure accounting, the ten-field control
  matrix, capacity scaling, Pareto and no-hidden-teacher checks;
- anti-gaming checks that are tested rather than promised: prompt-order
  invariance, few-shot shingle disjointness, tool and feedback parity, and a
  structural separation between the model process and the checker process;
- the pre-registration, bound to the suite hash before any arm run (v4 for the
  144-task suite);
- a near-duplicate screen over the diagnostic, code and assertion views of every
  authored item, with thresholds registered before it ran;
- a frozen 24-task sealed holdback, gated behind an unlock flag plus the
  registered hash, append-only, with refusal and acceptance proved by CPU tests;
- one declared **naive** shallow fix per category, proved by the self-test to
  compile and then fail a hidden behavioural test.

### Instrument corrections before any arm run (S16C, CPU-only)

Three defects in the same instrument were found and fixed before it scored a
single arm, each with a regression test that fails on the old behaviour:

1. **The verdict was not conjunctive.** `compare()` computed the ≤5 pp
   category-regression limit and then ignored it: an arm that cleared the 10%
   primary threshold while wrecking one category was reported as a win. The
   verdict now requires both conditions, names the regressing categories, and
   reports `improvement_met` separately from `threshold_met`.
   Test: `test_compare_is_conjunctive_a_category_regression_kills_an_aggregate_win`.
2. **A missing pre-registration digest skipped the binding.** `check()` only
   compared the recorded digest when one was present, so a run could be checked
   against a registration it was never made against. A registered run now requires
   a nonempty matching digest, the contract is verified against the suite hash and
   its threshold first, and `compare()` takes the threshold from the receipt —
   refusing an externally supplied value, a tampered recorded value, and a
   registered/unregistered mix. Tests:
   `test_a_registered_check_requires_a_recorded_digest`,
   `test_a_registered_check_refuses_a_contract_that_changed_after_generation`,
   `test_compare_refuses_a_threshold_supplied_after_the_results`,
   `test_compare_refuses_a_tampered_registered_threshold`,
   `test_compare_refuses_to_mix_a_registered_and_an_unregistered_result`.
3. **T2 checkpoints loaded with an unchecked `strict=False`.** The allowlist is
   now exact and derived: a checkpoint may contain exactly the arm's trainable
   tensors and may omit exactly the frozen base. A supplied training receipt is
   checked for arm, config digest, admission identity, checkpoint
   name/digest/step, and (for routed arms) the recorded dispatch mask placement.
   Tests: `test_a_matching_checkpoint_verifies_against_its_receipt`,
   `test_a_checkpoint_from_another_arm_is_refused`,
   `test_a_checkpoint_with_an_unexpected_tensor_is_refused`,
   `test_a_checkpoint_missing_a_trainable_tensor_is_refused`,
   `test_a_routed_arm_without_a_recorded_mask_placement_is_refused`,
   `test_a_checkpoint_digest_that_does_not_match_the_receipt_is_refused`.

One consequence is recorded rather than worked around: the four finished T2
`run.json` receipts do not record `dispatch_mask_order`, so a routed T2 arm cannot
yet be handed to the capability court with a verified dispatch identity, and the
adapter refuses it instead of guessing. That is the honest state, and it is why
no capability number exists for those arms.

Not established: any capability result. No model has been evaluated on the
correctness suite — the four evaluated T2 arms were scored on the *likelihood*
instrument (KL/NLL/top-1 against the teacher), which is a distillation measurement
and not a correctness measurement; P14.6 records that separation as a required
receipt field. The trio is below P11.21's item count, feedback loops are not
implemented, the corpus contamination comparison is a bounded sample, and no
3B-8B quantised baseline exists locally. Each is recorded as a gap rather than
papered over.

The P14.4 power receipt adds one more honest limit: the registered composite's
simulated 80%-power minimum detectable effect is above the registered 10%
threshold. The threshold is unchanged; a null near it is under-powered and must
be reported as such. The sealed holdback is frozen and gated but has not been
used, and it was authored by the same author and process as the scored suite, so
it controls for tuning against these exact tasks rather than for a shared
authoring style.

Recorded as a quotation, because these are statistical estimates rather than the
kind of performance number directive 12 governs: the receipt
`results/capability/power-20261116T090700Z.json` (stamp `20261116T082751Z`,
revision `5e5543fc08cae4f9e2065e9d38be53143b05c1f1`) puts the analytic
(normal-approximation) MDE at 0.1034 relative to the 0.5 baseline — almost
exactly the registered 10% threshold — while the *simulated* probability of
satisfying the registered conjunctive decision at that effect is 0.492, and the
simulated 80%-power MDE is 0.14 relative, 0.18 once 20% of gains are reversed.

**So the scored suite is under-powered for a 10% claim at n = 144.** The binding
constraint is the conjunctive rule, not the confidence interval: the interval
nearly covers at the analytic MDE while the joint decision does not. A null near
the threshold on this instrument is therefore **uninformative** and will be
reported that way — it cannot separate "no 10% effect" from "a 10% effect this
suite is too small to see" — and no under-threshold null in this file is or will
be read as evidence of no effect. The 10% threshold, the 5 pp category-regression
limit, the bootstrap seed and the resample count stay exactly as registered; the
instrument, not the threshold, is what falls short. Closing the gap is
**P14.8** (appended to WS14, no renumbering), whose acceptance evidence is a
power curve **simulated from the real per-item matrix**, not a formula.

The 24-task sealed holdback (P14.7) is **frozen and unscored**. Its hash, item
hashes and one-shot unlock rule are registered in
`fixtures/capability/SEALED.json` and bound in
`docs/capability-preregistration-v4.json`; its instrument self-check runs behind
an unlock record, but **no arm has been scored against it**, so this file reports
no sealed result for any arm. It also cannot carry a 10% claim on its own: 24
items, three per category, are far below the n the power arithmetic above
requires, and it was authored by the same author and process as the scored suite,
so it controls for tuning against these exact tasks rather than for a shared
authoring style.

## Next

The trained branch now exists: T2 (above) is the upcycled/sparse-budget branch the
previous version of this section asked for, and `H*` is its last arm. What comes
next is fixed by `docs/PROGRAM-PLAN.md` and is, in order: a T2 verdict; the
`H*`-target diagnostics (P12.2) and the successor utility target (P12.17) as a
single registered run against the dense control; T3 (P12.3). The M2 oracle bound
still says post-hoc selection cannot close the gap, so no inference-only selector
tuning is resumed without a new bound.

### Where the next phases are written down

The plan of record is `docs/PROGRAM-PLAN.md`:

- **§0.1** the three tiers (ACTIVE / CANDIDATE / HYPOTHESIS) and the tier index
  for all 140 items;
- **§0.2** the ordering inversion that put the control matrix and thresholds
  after the experiments they judge, and its correction;
- **§10–§17** WS7–WS12 (doctrine, complementarity, deficit-directed capability,
  eighteen mechanisms, twenty-one proposals, the staged frontier claim), with the
  dependency-ordered queue in §16 and the single NEXT in §17;
- **§18** WS13 — 21 items: MoRE-style expert reuse and three other published
  mechanisms as competing alternatives, plus sixteen inventions, each with a
  decisive experiment, a falsifying control and a kill criterion;
- **§19** WS14 — 7 items: the three runtime corrections (decoded-weight
  residency, synchronous prefetch, token-at-a-time prefill) and the four
  evaluation corrections (suite scale, executed Rust verifiers, correctness
  separated from likelihood, a sealed set).

That plan's first step is deliberately not a new training run: it is the
routed-overflow audit and the `H*`-target diagnostics (P12.1/P12.2), because
either can invalidate a scale-up and both are diagnosable without the GPU.

The citations those items rely on changed in the same pass. Twelve works named
by the strategy note could not be located and are marked `UNSUPPORTED`; one
("Routing in Gradient Space") is right about the mechanism and wrong about the
paper; and one (mHC) is published with an analysis showing its *effective*
stream count is low, which forbids reading it as a capacity win. See
`docs/PRIOR-ART-AND-NOVELTY.md` §10–§11 before quoting any of them.


## M4 part 1 — the first real inference runtime (S13a)

`selvedge run` executes the pinned Qwen3-0.6B control on one device with every
SwiGLU group read from a SELVEDGE bundle. Exact mode executes the complete native
shard set and proves, per token, that the executed set equals the set the
validated manifest declares; approximate mode executes a selector-chosen subset
and records both verdicts. Every executed token carries a JSONL accounting row.

The backend decision the mandate requires was taken with measurements: Candle was
built in the rust court, its dependency and build weight recorded, and rejected
for this phase in favour of an in-repo f32 kernel backend whose reduction order we
can state and test — which is what makes the Court E parity claim bitwise rather
than approximate. `tokenizers` 0.23.2 is pinned because it is the teacher's own
tokenizer engine.

Evidence: `docs/RUNTIME.md` (method, modes, witness, accounting, tolerances,
generated CPU numbers, and the explicit boundary of what is proven),
`results/runtime/court-e-0.6b-*.json` (parity across streaming variants),
`results/runtime/court-e-parity-0.6b-*.json` (real-model agreement with an
independent NumPy forward), `results/runtime/court-f-cpu-*.json` (CPU latency and
bytes per token), and `crates/selvedge-runtime/tests/` (tiny parity and the
twelve-variant Court E matrix).

What is not claimed: no GPU kernel, no GPU number, no quality result for the
approximate path, and no claim that the manifest's declaration is honest — the
witness proves equality with the declaration, and the declaration is a documented
trust input.
