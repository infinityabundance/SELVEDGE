# Experiment contract — M1, part 2: the quality/active-parameter frontier of Qwen3-1.7B under exact group omission

Status: **pre-registered**. This file was written and committed before any
frontier measurement was run. Its thresholds, arms, sample sizes and kill
conditions are fixed at the registration timestamp below. Any later change is
an amendment appended under "Amendments", dated, and must say what was learned
before the change. Results live in `results/M1-FRONTIER.md` and
`results/frontier/`, never here.

- Registered (UTC): 2026-11-14T21:06:27Z
- Registration commit: the commit that introduces this file; its SHA is
  recorded as `preregistration_commit` in every frontier receipt and in
  `docs/WORK-LOG.md`. The contract cannot contain the SHA of the commit that
  creates it, so the receipts carry it.
- Predecessor facts (not re-tested here): `docs/BUDGET-MATH.md`,
  `docs/WORK-LOG.md` S3a, `results/real-parity/20261114T204209Z/`.

## 1. Question

The strict rule is `unique active learned parameters per output token <= 1e9`
(`MANDATE` constraint 1). H0 proved that a dense SwiGLU FFN decomposes exactly
into channel groups and that the published f32 bundles reconstruct the dense
FFN bitwise. H0 does **not** prove that omitting groups preserves behaviour.

This experiment measures, for Qwen3-1.7B (the compliance-binding teacher) and
Qwen3-0.6B (a control whose dense form is already under the ceiling), the
achievable quality-versus-active-parameter frontier when a fixed set of whole
groups is omitted per layer at inference time, with no retraining and no
learned router.

The measured quantity is distributional fidelity to the dense f32 teacher on
held-out text. **It is not a capability benchmark.** A good KL means the
sparse model computes almost the same next-token distribution as the dense
model on this text; it does not mean the model can still write code, reason, or
answer questions. Any interpretation beyond that is out of scope and must be
labelled conjectured.

## 2. Registered model facts

These are receipt-sourced, not re-measured here.

Qwen3-1.7B (`70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`, Apache-2.0):

- ceiling:
  <!-- BEGIN GENERATED: contract-ceiling -->
1000000000
  <!-- END GENERATED: contract-ceiling -->
- always-active backbone (embedding + attention + norms + tied output):
  <!-- BEGIN GENERATED: contract-17b-backbone -->
663610368
  <!-- END GENERATED: contract-17b-backbone -->
- FFN elements affordable per token at the ceiling:
  <!-- BEGIN GENERATED: contract-17b-max-ffn -->
336389632
  <!-- END GENERATED: contract-17b-max-ffn -->
- whole groups per layer affordable with every layer active:
  <!-- BEGIN GENERATED: contract-17b-groups-per-layer -->
5
  <!-- END GENERATED: contract-17b-groups-per-layer -->
- full FFN elements per layer:
  <!-- BEGIN GENERATED: contract-17b-ffn-per-layer -->
37748736
  <!-- END GENERATED: contract-17b-ffn-per-layer -->
- elements in one of the sixteen contiguous groups:
  <!-- BEGIN GENERATED: contract-17b-group-elements -->
2359296
  <!-- END GENERATED: contract-17b-group-elements -->
- intermediate channels per layer:
  <!-- BEGIN GENERATED: contract-17b-intermediate -->
6144
  <!-- END GENERATED: contract-17b-intermediate -->

Qwen3-0.6B (`c1899de289a04d12100db370d81485cdf75e47ca`, Apache-2.0):

- unique learned elements (dense, compliant):
  <!-- BEGIN GENERATED: contract-06b-unique -->
596049920
  <!-- END GENERATED: contract-06b-unique -->
- always-active backbone:
  <!-- BEGIN GENERATED: contract-06b-backbone -->
331808768
  <!-- END GENERATED: contract-06b-backbone -->
- removable FFN:
  <!-- BEGIN GENERATED: contract-06b-ffn -->
264241152
  <!-- END GENERATED: contract-06b-ffn -->
- elements in one of the sixteen contiguous groups:
  <!-- BEGIN GENERATED: contract-06b-group-elements -->
589824
  <!-- END GENERATED: contract-06b-group-elements -->

The exact active count of every arm is computed by the harness from the realized
mask and written into that arm's receipt. No arm is described as compliant from
an estimate.

## 3. Hypotheses

- **H0 (already established, re-checked as an instrument gate).** The grouped
  MLP with all groups selected reproduces the dense f32 teacher within the
  documented f32 tolerance. This is a harness gate, not a new scientific claim.
- **H1 (primary).** There exists a selection of whole FFN groups with realized
  active unique parameters at or below the ceiling that preserves the dense
  teacher's next-token distribution within the pre-registered acceptable
  threshold of section 9.
- **H1a (secondary, within-uniform).** At a fixed budget, selecting groups by a
  calibration statistic (mean activation magnitude or activation frequency)
  preserves more of the teacher than a random selection of the same size.
- **H1b (secondary, non-uniform).** At a fixed total group budget, a
  non-uniform allocation across layers chosen by measured marginal KL on the
  calibration split preserves more of the teacher on held-out text than the
  uniform allocation of the same budget.
- **H1c (secondary, transfer).** The gain of the non-uniform allocation is not
  specific to its calibration domain: it must survive, at least in part, on
  code and math text that the allocation never saw.

## 4. Primary metrics (fixed before measurement)

All are computed at every predicting position of every held-out sequence,
teacher-forced on the held-out token ids. "Teacher" means the dense f32
Qwen3-1.7B forward (`model.float()`, TF32 disabled, `torch.inference_mode`).

1. **Teacher KL (primary).** For a position with teacher distribution `p` and
   masked distribution `q`, `KL(p||q) = sum_v p(v) (log p(v) - log q(v))`,
   in nats per token. Reported as the sequence-mean over positions, then
   averaged over sequences, with a paired bootstrap CI.
2. **Top-1 agreement.** Fraction of positions where `argmax q = argmax p`.
3. **Held-out perplexity ratio.** `PPL_ratio = exp(mean_nll_masked -
   mean_nll_teacher)` over the same positions, excluding position 0 of each
   sequence (no context). This is the ratio of the masked model's own NLL to
   the teacher's own NLL on identical tokens.

The primary metric order is KL, then PPL ratio, then top-1 agreement. A
configuration is "acceptable" only if all three thresholds of section 9 hold.

## 5. Secondary metrics

- KL median and p95 over positions, with a sequence-level bootstrap CI.
- Mean teacher entropy (nats/token) on the same positions; reported so a KL
  number can be read against the entropy it is a fraction of.
- Per-domain KL, PPL ratio and top-1 (general/factual wikitext, code, math).
- Realized active unique parameters per arm, from the realized mask and the
  receipt-sourced component counts (not estimates).
- Derived active FFN matmul FLOPs per token from the realized channel count
  (`6 * active_intermediate * hidden` per layer, `docs/ARCHITECTURE.md`).
  Labelled derived, not measured by hardware counters.
- Wall-clock per arm and per forward, measured with `torch.cuda.Event`, plus
  device, driver, torch/CUDA versions, dtype and arithmetic flags in the
  receipt.
- Per-layer, per-group calibration statistics: mean group activation L2 norm
  and the frequency statistic of section 6.2.
- Cost of the intervention: number of groups in the mask, mask bytes,
  selection time, and the total GPU wall-clock of the arm.

## 6. Arms

All arms share the same tokenized held-out inputs (paired design). A mask is a
per-layer vector of sixteen booleans; the masked MLP output is
`W_down[:, S] (SiLU(W_gate[S] x) * (W_up[S] x))` where `S` is the union of
channels of the selected groups. Every group is one of the sixteen contiguous,
equal-width groups of the existing bundle policy.

### 6.1 Reference arms

- `teacher-dense`: unmodified dense f32 model. Noncompliant by construction,
  the reference for every metric.
- `masked-all`: every group selected through the masked path. Must equal
  `teacher-dense` within the harness tolerance (gate G1).
- `masked-none-ffn`: all FFN groups omitted. Included as the floor; it is not
  a candidate operating point.

### 6.2 Selectors (static per layer, fixed over tokens)

For each selector, `k = 1..16` groups per layer:

- `random`: for each layer, the selected set is the first `k` elements of a
  deterministic permutation drawn with `torch.Generator(seed=20261114)` from
  the sixteen groups, one permutation per layer, generated in layer order.
  Reproducible from the seed and layer index alone.
- `magnitude`: rank groups within each layer by the mean over calibration
  tokens of `||a_g(t)||_2`, where `a_g(t)` is the group's intermediate
  activation vector `SiLU(W_gate[G_g] x_t) * (W_up[G_g] x_t)`. Select the top
  `k`. Ties break by group index.
- `frequency`: for a calibration token, let
  `mass_g(t) = ||a_g(t)||_2 / sum_j ||a_j(t)||_2`. Rank groups within each
  layer by the fraction of calibration tokens with `mass_g(t) > 1/(2*16)`
  (twice the uniform share). Select the top `k`. Ties break by group index.

Selectors see calibration text only. They never see held-out text, teacher
logits, token labels, or task answers.

### 6.3 Global-budget allocation (offline oracle)

- `greedy-uniform5`: start from `uniform k = 5` of the `magnitude` selector's
  ranking (or, if that is not the uniform baseline, from the uniform
  `magnitude` set of size five per layer; the receipt records the exact starting
  mask). Add groups one at a time, up to the largest number of whole groups
  whose realized active count stays at or below the ceiling. At each step,
  evaluate every unselected group in every layer by exact teacher KL on a fixed
  calibration probe of six calibration sequences truncated to 128 tokens
  (768 positions), and add the group with the largest KL reduction. Ties break
  by (layer, group index). This is explicitly **non-deployable**: it uses
  teacher logits and held-out-free calibration text, and is an offline oracle.
- `greedy-swap4`: start from `uniform k = 4` of the `magnitude` ranking. For up
  to three iterations, evaluate all `(layer_remove, layer_add)` swaps where the
  removed group is the lowest-ranked selected group in its layer and the added
  group is the highest-ranked unselected group in its layer (28 x 28 = 784
  candidates), using the same calibration probe. Accept the single swap with
  the largest KL reduction if it is positive; stop when no swap improves. Same
  non-deployable label.

Both allocations are produced by a script that logs every candidate evaluation
(count, KL, mask hash). Those logs are receipts, not prose.

### 6.4 Whole-layer omission

- `layer-omit-i` for each single layer `i`: omit all sixteen groups of layer
  `i`, keep all groups elsewhere. 28 arms.
- `layer-omit-first-n` and `layer-omit-last-n` for `n` chosen so the realized
  active FFN element count is closest to, but not above, the affordable FFN
  share: the receipt picks the largest feasible `n` and records it. These are
  the coarse-granularity comparison to group omission.

### 6.5 Control model

Qwen3-0.6B, same three selectors at `k = 1..16`, plus `teacher-dense` and
`masked-all`. "Comparable relative sparsity" means the same retained fraction
of groups per layer, not the same absolute parameter count, because the 0.6B
dense FFN is smaller than the 1.7B compliant FFN share.

## 7. Data, splits and sample sizes

### 7.1 Held-out text

- `wikitext-2-raw-v1` validation split, pinned dataset revision and file
  SHA-256 in `docs/DATASETS.md` and the committed manifest. License: the
  dataset card's CC BY-SA 4.0 terms are recorded verbatim with the pin.
- Domain slices: HumanEval prompts (code, MIT) and GSM8K test questions
  (math, MIT), pinned revisions and hashes, short excerpts only.

### 7.2 Design sizes (not measured results)

- Wikitext held-out: target 48 sequences of up to 512 tokens from distinct
  documents; minimum 12 sequences. Calibration: target 16 sequences from
  distinct documents; minimum 6. The calibration and held-out sequence sets
  are disjoint by document.
- Code: target 40 HumanEval prompts; minimum 20.
- Math: target 40 GSM8K questions; minimum 20.
- Greedy probe: 6 calibration sequences, first 128 positions each.
- If a minimum cannot be met after the near-duplicate filter, the run is
  invalid and reported as invalid; sample sizes are not silently reduced.

### 7.3 Split rule

A wikitext "document" is the text of one line of the raw validation split,
excluding empty lines, section-header-only lines and lines shorter than 200
characters. Documents are ranked by the hex SHA-256 of their normalized text
(lowercase, whitespace collapsed). The first `N_calibration` ranked documents
form calibration; the next `N_heldout` form held-out. The split is therefore by
document hash, deterministic, and content-independent.

Chunking: a document longer than 512 tokens contributes the first 512 tokens as
one sequence; a shorter document contributes all its tokens (the sequence is
left-padded nowhere and right-truncated nowhere). No sequence crosses a
document boundary. Tokenization is the pinned Qwen3 tokenizer with
`add_special_tokens=True`.

## 8. Leakage prevention and contamination

1. **Split by document hash**, as in 7.3. A document contributes to exactly one
   side.
2. **Near-duplicate check.** Normalize (lowercase, collapse whitespace,
   strip punctuation), form word 5-shingles, and compute the Jaccard index for
   every calibration x held-out pair. Any held-out document with Jaccard >=
   0.2 against any calibration document is discarded before selection; the
   discards, the maximum Jaccard before and after, and the number of pairs in
   `[0.2, 0.5)` and `>= 0.5` are recorded in the split receipt. If a discard
   would push a set below its minimum, the run is invalid.
3. **No labels or prompts in selectors.** Selectors are computed from
   calibration text and calibration activations only. The teacher's logits,
   the held-out token ids and the domain slice text are never inputs to a
   selector. The greedy oracle may use teacher KL on calibration, which is
   allowed by `MANDATE` constraint 5 (offline calibration), and is labelled
   non-deployable.
4. **Domain slices are held out for selection as well as evaluation.** Code and
   math text are never used by any selector, ranking or greedy step. They are
   the distribution-shift test.
5. **Pretraining contamination cannot be checked locally.** Wikitext-2 is
   public Wikipedia text and is plausibly present in Qwen3's pretraining
   corpus; we have no local membership test. This is recorded as a residual
   risk. Absolute perplexity is therefore optimistic; the KL comparison is
   relative to the same teacher on the same tokens and is not made more
   favourable by memorization in any way we can detect. No capability claim is
   made.

## 9. Pre-registered thresholds

These numbers are fixed now, before any measurement.

At realized active unique parameters `<= 1e9` (and at every reported budget,
for comparability):

- **Acceptable (H1 supported).** All of:
  - held-out mean teacher KL `<= 0.05` nats/token,
  - held-out PPL ratio `<= 1.02`,
  - held-out top-1 agreement `>= 0.95`,
  and the 95% paired-bootstrap CI upper bound for mean KL is `<= 0.05`, the CI
  upper bound for the PPL ratio is `<= 1.02`, and the CI lower bound for top-1
  agreement is `>= 0.95`.
- **Usable but degraded (reported, H1 not supported).** All of:
  - mean KL `<= 0.25`, PPL ratio `<= 1.10`, top-1 agreement `>= 0.85`.
- **Unacceptable.** Anything worse than the usable tier.

Non-uniform value (H1b/H1c):

- The non-uniform allocation must reduce held-out mean KL by at least 15%
  relative to the matched-budget uniform `magnitude` arm, with a paired
  bootstrap 95% CI for the relative reduction whose lower bound is above zero,
  to claim a practical benefit over uniform.
- It must retain at least 25% of that relative reduction on code and on math
  (each domain assessed separately), with a CI lower bound above zero, to claim
  that the benefit transfers. Otherwise the result is a calibration-domain
  overfitting finding.

## 10. Instrument gates and kill conditions

- **G1 (harness exactness).** `masked-all` versus `teacher-dense` on every
  evaluated sequence: max absolute logit difference `<= 0.01` and mean KL
  `<= 1e-4` nats/token. Cross-check: for one real layer and one real subset,
  the Python masked partial sum and the Rust grouped partial sum agree
  elementwise within `|a-b| <= 1e-3 * max(1, |b|)` (both f32; reduction order
  differs). If G1 fails, no frontier claim is made; the failure is retained and
  the harness is fixed first.
- **K1 (instrument).** Any non-finite logit, any comparison that cannot be
  computed, or a G1 failure. Stop; publish the failure.
- **K2 (H1).** If no `<= 1e9` arm reaches the acceptable tier, H1 is rejected
  for static whole-group omission on this teacher. The negative result is
  published with the frontier. This is an expected possible outcome and is not
  a failure of the experiment.
- **K3 (non-uniform).** If the best non-uniform allocation does not meet the
  15% criterion in section 9, non-uniform static allocation has no demonstrated
  practical value at this budget. Reported as such.
- **K4 (transfer).** If the non-uniform allocation loses the 25% retention on
  code or math, the gain is a calibration-domain overfitting result.
- **K5 (operational).** OOM, timeout, NaN/Inf, court failure or a crashed run
  is recorded verbatim with command, environment and exit status, and is never
  silently replaced. A resumed run records both the failed attempt and the
  successful one.

## 11. Statistical method

- Unit of analysis: sequence (document). Metrics are first averaged over
  positions within a sequence, then over sequences. All arms see the same
  sequences (paired).
- CIs: paired percentile bootstrap over sequences, 10,000 resamples, RNG seed
  `20261114`, implemented with `numpy.random.Generator(PCG64(seed))` and the
  same resampled index matrix for every arm of a comparison. 2.5th and 97.5th
  percentiles.
- Effect size: for a paired comparison, the mean per-sequence difference, its
  CI, and the relative reduction `(KL_baseline - KL_arm) / KL_baseline`.
- Confirmatory comparisons: K2 (best compliant arm vs acceptable tier), K3
  (non-uniform vs matched-budget uniform), K4 (transfer). The rest of the
  frontier is descriptive; its CIs are reported but no multiplicity-adjusted
  significance claim is made.
- Locale and platform: bootstrap uses only integer sequence indices and float
  accumulation; it does not depend on BLAS or GPU arithmetic.

## 12. What a negative result means

If H1 is rejected, the reading is: under this exactness regime, with this
teacher, this grouping granularity (16 equal contiguous groups), static masks,
no retraining and no learned routing, the 1e9 ceiling cannot buy a
distribution-preserving 1.7B execution. That would not prove that no sparse
method can; it would bound this family. It would also mean that the 1.7B
teacher is not a viable base for compliant group omission, and that the smaller
compliant model (0.6B) plus external capacity, or a retraining branch, would be
the remaining path. The report must state the bound exactly as measured,
including the granularity and the static-mask restriction, and must not
generalise it to dynamic or trained selectors.

If H1 is supported, the reading is only that next-token distributions are
preserved on the measured text; it is still not a capability result.

## 13. Deviations

Planned deviations are not allowed. Unplanned deviations are recorded in
`docs/WORK-LOG.md` and in the affected receipt under `deviations`, with the
reason and the timestamp. A change to any threshold, arm definition, selector
definition or split rule after the first measurement is an amendment and must
be appended here with its own timestamp; the pre-amendment results are reported
against the original contract.

## Amendments

None at registration.
