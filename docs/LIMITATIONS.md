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
  the ceiling". It is a greedy heuristic, it is explicitly non-deployable
  (it uses teacher logits and calibrates with an oracle), and its candidate set
  is pruned to one add and one remove per layer. A better search could do
  better; this experiment did not run one.
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
