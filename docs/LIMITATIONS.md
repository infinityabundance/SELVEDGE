# Limitations, losses and external validity

Status: current as of S3b (2026-11-14). This file records what the implemented
system and the measured results do **not** establish, including hazards that
could mislead a reader. It is deliberately blunt. Results are in
`results/M1-FRONTIER.md`; the registered experiment is
`docs/EXPERIMENT-CONTRACT.md`.

## What the M1 frontier result is

The frontier measures the dense f32 teacher's next-token distribution against a
model whose FFN computes a fixed subset of whole channel groups. The metric is
teacher KL, PPL ratio and top-1 agreement on pinned held-out text.

- **It is not a capability benchmark.** KL on Wikipedia text does not measure
  code generation, arithmetic, reasoning, factuality, instruction following or
  long-context behaviour. A model with low KL can still fail a task; a model
  with higher KL can still pass one. The observed divergences here are so large
  that the direction is not in doubt for the measured arms, but the mapping
  from KL to task quality is not established.
- **It bounds one family of interventions only.** The masks are static over
  tokens, operate on sixteen equal contiguous channel groups per layer, use no
  retraining, no learned router, no dynamic per-token selection, no gradient
  attribution, no interaction-aware selection, and no upcycling. The negative
  result does not generalise to those methods. It is evidence about this family.
- **It uses one grouping granularity.** Sixteen groups is the converter
  default. A coarser or finer partition changes both the mask resolution and
  the arithmetic; nothing here says the result is invariant to it.
- **It is one model family.** Qwen3-1.7B and Qwen3-0.6B. The all-or-nothing
  sensitivity may be specific to how these checkpoints use their FFNs. Other
  architectures, or models trained with FFN dropout or sparsity, may behave
  differently.
- **Absolute perplexity is not comparable to published numbers.** Wikitext-2,
  HumanEval and GSM8K are plausible pretraining data for Qwen3. We cannot test
  membership locally. The KL and PPL-ratio comparisons are relative to the same
  teacher on the same tokens, which is the only claim made; absolute PPL is
  optimistic.

## What the numbers do not support

- Do not read the KL values as "the model has lost N% of its ability". KL is a
  distributional distance on one text distribution.
- Do not read the non-uniform allocation result as "optimal allocation solves
  the ceiling". It is a greedy heuristic, it is causally unavailable at runtime
  (it uses teacher logits during calibration and its mask is an offline oracle
  result), and its candidate set is pruned to one add and one remove per layer.
  A better search could do better; this experiment did not run one.
- **Do not read the M2 active-count column as physical reads.** The original
  instrument charged logical selections, not parameter elements. The corrected
  accounting and the two verdicts (`causal`, `physical`) are in
  `docs/M2-CORRECTION.md`. Arms whose selector reads full gate/up projections or
  whose execution is dense-masked are physically non-compliant even when their
  logical active count is under the ceiling.
- Do not read the code and math slices as capability evaluations. They are
  prompt-only next-token-distribution slices. The canonical solutions and gold
  answers are never fed to the model.
- Do not compare the 1.7B and 0.6B curves as if they were one model at two
  sizes. Each is measured against its own dense teacher. The 0.6B dense model is
  already compliant, so its sparsity curve is a cost experiment, not a
  compliance experiment.

## Instrument limitations

- **f32 reference arithmetic.** The dense teacher is the bf16 checkpoint loaded
  as f32 (an exact widening) and run with TF32 disabled. This is a
  f32-equivalent teacher, not the model's native bf16 inference. The native
  bf16 path was characterised in S3a and differs within the documented bf16
  bound. Native bf16 KL could differ slightly from the f32 KL measured here.
- **Reduction order.** The fast masked path keeps the dense down-projection
  reduction order with excluded channels exactly zero. The Rust cross-check
  sums group-by-group. They agree within a registered f32 bound on a real
  subset, not bitwise. `masked-all` is bitwise equal to the dense teacher on the
  measured windows.
- **Single machine, single driver.** GPU model, driver, CUDA runtime, torch and
  transformers versions are in every receipt. Different kernels could round
  differently and change KL at the margin. The qualitative collapse is far
  outside any plausible rounding difference; the last digits are not portable.
- **Timings are cost observations, not benchmarks.** Each arm records the sum
  of CUDA-event forward times over its windows. There is no warm/cold control,
  no interleaved A/B schedule, no p95 latency, no memory-bandwidth
  instrumentation and no concurrency. FLOP values are derived from realized
  channel counts, not hardware counters.
- **Mask overhead is not modelled.** The Python harness applies a per-channel
  mask; a production runtime would repack or select contiguous channels. The
  measured wall clock includes Python and mask-application overhead and must
  not be read as the achievable sparse-inference speed.
- **Bootstrap scope.** 40 held-out articles (up to two windows each) and 40
  items per domain slice. Windows within an article are averaged before the
  bootstrap, so the CI clusters by document, but the article population is
  small and drawn only from wikitext-2 validation.

## Data and leakage limitations

- The near-duplicate check is exact 5-word-shingle Jaccard at a pre-registered
  0.2 threshold. It runs before the split is finalised. It cannot detect
  semantic duplication, paraphrase or shared facts.
- Splits are disjoint by article hash, but articles from the same Wikipedia
  topic can still share entities and phrasing below the shingle threshold.
- Pretraining contamination is unverifiable locally (see above). No benchmark
  score was computed, so no contamination-corrected benchmark claim exists.
- The domain slices are prompt-only; they carry the style of HumanEval and
  GSM8K but not their task structure.

## Engineering limitations

- The Python oracle is instrumentation only. It is not a serving runtime, has no
  batching, no KV cache management, no CUDA kernels and no memory plan. The
  Rust side reads bundles and computes grouped sums; it does not yet serve
  inference.
- The bundle stores f32 and is large (the f32 FFN payload of 1.7B is several
  gigabytes). No compression, quantization or IO path is evaluated here.
- `selvedge-atlas`, `selvedge-store`, `selvedge-sim` and `selvedge-runtime` do
  not exist yet. There is no selector training, no scheduler, no trace replay
  and no physical transfer measurement.
- The oracle image was rebuilt with a pinned `pyarrow` for the parquet datasets.
  A build under the default 4 GiB build limit was OOM-killed; the successful
  build used a higher explicit bound. The image lock records the result.
- `trust_remote_code` is never enabled and models are loaded from local paths
  with `local_files_only=True`. No model repository code runs.

## Residual risks that would invalidate a stronger claim

- Treating the KL frontier as a capability frontier.
- Treating the static-mask bound as a bound on dynamic or trained selectors.
- Quoting absolute PPL as if the evaluation text were unseen by the teacher.
- Quoting the derived FLOPs or the harness wall clock as measured runtime
  performance.
- Treating the two-group additive-greedy observation as the registered
  matched-budget result; it is not (see `results/M1-FRONTIER.md`).
- Treating a `PhysicalCompliance::Compliant` record as a runtime benchmark. It
  says the measured path reads only counted elements; it does not measure
  wall-clock throughput or a trained model.

## Storage / trust limitations

- **Publication proofs are a trust input.** `VerificationMode::Always` hashes
  every shard on every read. `VerificationMode::PublicationProof` trusts a
  registry whose bundle id and manifest digest match; a forged registry could
  therefore hide corrupted bytes. Every reported measurement in this repository
  uses `Always`; the proof path exists for a warm audit path that is not yet
  measured.
- **Sparse read counts are per matmul call, not per token.** A batched forward
  reads each sliced weight once per matmul; a one-token decode re-reads the same
  slice. The reported `execution_elements_read` is for the measured batched
  forward, and the *unique* set (what the ceiling constrains) is per output
  token. Both are recorded; they are not interchangeable.
- **The storage measurements are single-node.** One consumer NVMe device, one
  kernel, one page-cache policy. They are repeated, but they do not generalise
  to a different drive, firmware, filesystem or concurrency level.
- **The simulator is calibrated on one host and one device, and does not model
  a GPU tier.** Its transfer and verification distributions come from the
  measured per-request JSONL; simulated queue depth above one is a what-if,
  because the measured engine reads synchronously. The one non-measured curve
  is the log-log size law for request sizes that were not measured. GPU staging,
  PCIe and pinned-buffer effects are absent, so no GPU-latency claim is possible
  from a simulation.
- **The live simulator-vs-store comparison was run on a contended host.** The
  device bytes and hit counts agreed exactly and the p95/p50 shape was within a
  few percent, but the absolute latency comparison failed its tolerance because
  the detached T2 pilot was saturating the CPU and the drive; a same-window
  `io-scan` recorded the drift. Absolute latency validation on an idle host is
  outstanding.
- **No trained-router execution trace exists.** The T2 per-step records carry no
  routing field, and the M2 selector decisions come from a degraded, untrained
  setting. Every simulator locality conclusion is conditional on a trained
  model's routing statistics.

## M2 selectors: what the per-token result does not establish

The M2 report is `results/M2-SELECTORS.md`; the registered contract is the M2
section of `docs/EXPERIMENT-CONTRACT.md`. The measured result is negative for
the inference-only, post-hoc selection family.

- **The M2 selection traces come from a degraded, untrained-router setting.**
  The per-token subsets are post-hoc statistics on an unmodified dense
  checkpoint, not the output of a trained router. Their instability (subsets
  change almost every token; the one-pass and two-pass arms differ) should not
  be extrapolated to a trained router, a model trained with sparsity, or a
  different granularity.
- **The one-pass comparison is small.** Eight held-out windows, one seed, one
  teacher. Its paired difference excludes zero on that sample; that is a
  measured comparison, not a population claim.
- **No exact neural execution is claimed.** The `NativeSetWitness` and
  `ExactSelection` types exist and are tested, but no end-to-end exact-mode
  runtime consumes them yet. M1-M3 measurements use the approximate masked
  path.
- **Two trust boundaries the types cannot close.** A `NativeSetWitness` proves
  equality with the *declared* native set, not that the declaration is the
  model's true native computation; the declaration must come from the validated
  bundle manifest, and no runtime does that yet. A `PhysicalCompliance::Compliant`
  verdict proves that the reads are labelled measured and names the audit, not
  that the audit was performed; the committed receipt is the evidence a reader
  must check.
- **The one-pass and offline timings use different conventions.** For the online
  arm the selector hooks run inside the timed forward; for the offline and
  random arms the selector runs outside it. `selector_seconds`,
  `execution_seconds` and `total_seconds` are recorded separately. The online
  `execution_seconds` is a subtraction and is not a clean execution measure,
  because the per-layer synchronisation inside the hook also waits for forward
  work; the online total is the defensible comparison across arms.
- **The oracle is local, not globally optimal.** It exactly minimises a
  one-layer squared-output error per token and then applies the map; it does
  not solve the joint per-token combinatorial problem over all layers and
  positions. A stronger search could do better. The gap to the usable tier is
  so large that a better local search is unlikely to close it at this budget,
  but that is an inference, not a theorem.
- **The 64-group oracle uses bounded greedy, not exact enumeration.** Exact
  `C(64,20)` enumeration is infeasible; each greedy step is exact under the
  incremental local objective, but the beam is not widened. A different
  ordering could produce a better subset.
- **The final refinement is a 2-window subset.** It uses 32 token positions
  and one pass, not all eight windows and two passes. It is an exploratory
  upper-bound probe, not the registered eight-window comparison.
- **The oracle sample is eight articles.** Its CIs are article-level and
  coarse. The qualitative gap to the usable tier is far larger than the CI
  width, but the exact mean is sample-specific.
- **The learned selectors use one label definition.** The registered label is
  the marginal local value given a four-group magnitude base. A standalone or
  greedy-path label, a larger model, or a different feature set could learn
  more. The M2 result shows that this label and these features do not transfer
  to final-KL quality; it does not show that no learned selector can.
- **The learned interaction-aware arm is a linear bilinear surrogate.** The
  interaction measurement shows the pairwise term is context-dependent and not
  predicted by the tested graph features. A richer interaction model is
  untested.
- **Gradient x activation and corrected gradient x activation are causally
  unavailable here (`backward_target`).** Their backward target uses
  teacher-forced tokens and a summed loss. They are attribution baselines, not
  online selectors. The corrected term is a reproduction of the stated
  corrective term in the 2026 paper, not of its full experimental pipeline.
- **The 868-element controller mismatch is real.** Amendment A3 explains why
  exact element equality is impossible at whole-group granularity. The learned
  arms are 868 elements above the baseline, never below it.
- **KL is not capability.** The measured numbers are next-token distributional
  fidelity on pinned Wikipedia, code-prompt and math-question slices. No task
  accuracy, generation quality, latency or throughput claim is made.
- **The Rust selector latency is CPU, release-mode, fixture-cycled.** It is a
  controller-cost measurement, not an end-to-end serving benchmark. It excludes
  the FFN forward, memory pressure and concurrency.

## The capability court is an instrument, not a result

`docs/CAPABILITY-COURT.md` is the reference. As of S26 the scored suite (size in
that document's §2), the trio, the sealed holdback, the checkers, the sandbox,
the contamination gate, the near-duplicate screen and the v4 pre-registration
exist and are CPU-tested; no model has been evaluated on any of them.

**Dated addition, 2026-11-16 (S28).** The five T2 arms were then evaluated and
every one scored 0.0000. `docs/CAPABILITY-COURT.md` §12 classifies all 765
arm/item outcomes from the stored receipts and shows the result is an **instrument
floor**: the v4 format asks for complete multi-file crate replacements, no arm
produced a compiling answer on any multi-file task, and tolerant extraction of
the discarded answers rescued nothing that passed. The 0.0000 ranks nothing and
must not be read as a capability ranking between `D`, `M`, `X`, `H` and `H*`.
The registered power analysis assumed a 0.5 baseline that the measurement
contradicted; it is invalid, not merely optimistic. The redesigned formats and
the calibration protocol (§13, §14) are registered and CPU-verified, but **no arm
has been run on them**, so no capability claim exists at any tier.

**Dated addition, 2026-11-16 (S30).** The S29 pass refused a suite version because
every arm's answer was constant. S30 measured the cause and replaced the
selection formats with a likelihood ranking (`docs/CAPABILITY-COURT.md` §18).
Three new limits follow, and they are narrower than the instrument they replace:

- **Likelihood selection measures discrimination, not generation.** It asks
  whether the arm ranks a checker-verified candidate above wrong ones. It
  bypasses writing code, so a correct ranking must never be reported as "can
  solve these tasks". An arm can rank correctly and still emit nothing that
  compiles; that is exactly what the free-form formats show for these arms.
- **Varying answers are necessary, not sufficient.** The answer-entropy screen
  excludes a format whose answers are near-constant, but a format can have high
  answer entropy and a zero pass rate. On the S29 receipts, `file-replacement`
  passes the screen for `D` while passing no item; it is uninformative for a
  reason the screen does not measure. The screen is therefore used to *exclude*,
  never to *accept*, a format.
- **The pilot is two arms of one family and is unpowered.** `D` and `X` produce
  identical totals on the selection items, the across-arm spread is zero, and the
  measured required n at the registered effect exceeds the available items per
  category by more than an order of magnitude. No arm comparison at the
  registered threshold may be reported from it, and nothing here separates the
  five arms.

Diagnostic-identification is also *informative but not calibrated*: its verified
rate for these arms is at or below the chance rate of the option alphabet, so it
is reported on its own metric and never pooled with executable success.

The following remain true and are added to rather than replaced by the S28 work.

- **The suite is authored and still small.** It is not a benchmark and not a
  random sample of programming work. The P14.4 expansion is recorded in
  `docs/CAPABILITY-COURT.md` §2 (generated suite and category counts); one item
  moves a category by the fraction implied by that count, and the confidence
  intervals are over items, not over the population of software tasks. The
  easy/medium/hard labels are author-assigned, not calibrated.
- **The instrument is under-powered for the registered threshold.** The
  simulated 80%-power minimum detectable effect reported in
  `docs/CAPABILITY-COURT.md` §7.1 is above the registered 10% threshold. A null
  result near the threshold must be reported as under-powered, not as evidence
  of no effect. The threshold itself is unchanged; the numbers and the model are
  in §7.1 and the power receipt.
- **Correctness and likelihood are not merged.** The suite's judge is executable
  success (`cargo test` under the sandbox). A reference-continuation
  likelihood/NLL or KL is a **selector signal only**: it may choose a checkpoint
  or compare distillation, and it must never be reported as capability. The
  result schema keeps verified success and any likelihood field separate.
- **Disjointness is a property of this repository's data, not of a
  checkpoint.** No local test can show that a model never saw these patterns
  during pretraining; the contamination receipt compares our text against
  HumanEval, GSM8K, TruthfulQA, the T2 evaluation windows and a *bounded
  deterministic sample* of the training corpus, and it fails closed on overlap.
- **The verifier is executable but not complete.** Hidden tests plus one
  authored wrong mutation per task, and one declared naive fix for one task per
  category, show that the checker discriminates; they do not prove it rejects
  every wrong program. The suite also proves that a forged successful stdout does
  not pass.
- **Near-duplicates are screened, not eliminated.** The similarity screen
  compares diagnostic, code and assertion views with thresholds registered before
  the run and rejects no pair involving a new item. Two pairs from the
  v3-registered historical suite are above a threshold and are reported rather
  than rewritten, because replacing them would change a registered fixture.
- **The anti-superficial-specialisation trio is still below P11.21's item
  count.** It now has the operation families and surfaces recorded in
  `docs/CAPABILITY-COURT.md` §3, but the operation annotations are author-only,
  there is no inter-annotator agreement, and with so few tasks per surface an
  operation-family effect cannot be separated from a surface effect. Per-surface
  results are reported separately and never pooled.
- **The sealed set controls for tuning, not for the author's taste.** The
  sealed items are gated behind an unlock flag and the registered hash, and no
  arm has run against them. They were authored by the same author and process as
  the scored suite, so a sealed result can separate an in-sample gain from tuning
  against these exact tasks, but not from a shared authoring style. The set has
  also been instrument-checked more than once during authoring; every unlock
  record is retained, and the freeze predates any model evaluation.
- **Compiler/test feedback loops are not implemented.** Feedback is recorded and
  parity-enforced, so no arm can currently receive a hidden tool subsidy, but
  the repair loop itself is future work.
- **The Rust-runtime adapter has no end-to-end generation yet**, and the
  PyTorch adapter cannot report device read counters or host residency, so those
  fields are null with reasons.
- **A Level 1 gain is not a Level 3 product.** Level 3 additionally needs
  responsiveness from Court F and a product-branch (3B-8B quantised) baseline,
  which is a planned control because no such checkpoint is present locally.
- **The calibrated formats are a redesign, not a proven improvement.** Item
  difficulty, discrimination and the easy/medium/hard ladder are unmeasured until
  the calibration pass runs; the derivation proves only that each item's failing
  state fails, its reference passes, exactly one patch candidate passes and
  exactly one diagnostic consequence resolves the failure. A format that turns
  out to be saturated is retired by the registered window, not assumed useful.
- **`verified_diagnostic` is not capability.** Naming the right compiler error
  and having the named repair resolve it is a weaker claim than producing the
  repair. It is reported on its own metric, per category, and never pooled with
  executable success or with partial credit.
- **Partial credit is descriptive.** The fraction of hidden tests passed rewards
  progress and is never a decision input; a model that fails every item but
  passes some assertions cannot outrank one that solves fewer items completely.
- **The visible test is one hidden test.** Body and patch items show a single
  test drawn from the item's own hidden set. That is intentional and does not
  leak the hidden set as a whole, but it does mean the visible test is not an
  independent sample of the behaviour the hidden tests check.
- **Calibration is five arms of one family.** The item statistics come from
  `D`, `M`, `X`, `H` and `H*` — one seed, one budget, one base model. An item
  that discriminates among these five is not thereby validated against a different
  model family, and the item–total correlation it reports is low-precision by
  construction with five arms.
- **The address-space limit is a toolchain constraint, not a measured workload
  peak.** The per-checker bound is the smallest value at which the pinned
  toolchain links a one-file crate; the enforced memory bound for the workload
  is the container's. A cgroup OOM kill and an allocator abort under
  `RLIMIT_AS` are recorded distinctly, and the probe receipt shows the latter.
