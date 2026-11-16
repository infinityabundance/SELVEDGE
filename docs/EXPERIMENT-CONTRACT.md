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
   in nats per token. Reported as the position mean within a window, then the
   window mean within an article, then the article mean. The CI resamples
   articles, not windows.
2. **Top-1 agreement.** Fraction of positions where `argmax q = argmax p`.
3. **Held-out perplexity ratio.** `PPL_ratio = exp(mean_nll_masked -
   mean_nll_teacher)` over the same positions, excluding position 0 of each
   sequence (no context). This is the ratio of the masked model's own NLL to
   the teacher's own NLL on identical tokens.

The primary metric order is KL, then PPL ratio, then top-1 agreement. A
configuration is "acceptable" only if all three thresholds of section 9 hold.

## 5. Secondary metrics

- KL median and p95 over positions, with an article-level bootstrap CI.
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
  layer by the fraction of calibration tokens with `mass_g(t) > 2/16`
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
  calibration probe (six calibration articles, first 128 positions of their
  first window; 768 positions), and add the group with the largest KL
  reduction. Ties break by (layer, group index). This is explicitly **non-deployable**: it uses
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
  dataset card's CC BY-SA 3.0 and GFDL tags are recorded with the pin.
- Domain slices: HumanEval prompts (code, MIT) and GSM8K test questions
  (math, MIT), pinned revisions and hashes, short excerpts only.

### 7.2 Design sizes (not measured results)

- Wikitext held-out: 40 article documents, each contributing up to two
  non-overlapping 512-token windows; minimum 30 articles. Calibration: 16
  article documents, each contributing up to four non-overlapping 512-token
  windows; minimum 10 articles. The calibration and held-out article sets are
  disjoint.
- Code: 40 HumanEval prompts; minimum 20.
- Math: 40 GSM8K questions; minimum 20.
- Greedy probe: 6 calibration articles, first 128 positions of their first
  window.
- If a minimum cannot be met after the near-duplicate filter, the run is
  invalid and reported as invalid; sample sizes are not silently reduced.

### 7.3 Split rule

A wikitext "document" is one article: a level-1 header line (` = Title = `)
plus every following line until the next such header. Documents are ranked by
the hex SHA-256 of their normalized text (lowercase, whitespace collapsed).
The 16 lowest-hash articles form calibration; of the next 44 by hash, the first
40 that pass the near-duplicate filter of section 8 form held-out. Remaining
articles are unused. The split is therefore by document hash, deterministic,
and content-independent.

Chunking: a document's body is tokenized once (pinned Qwen3 tokenizer,
`add_special_tokens=True`, no padding). Non-overlapping windows of 512 tokens
are taken from the start: calibration documents contribute up to 4 windows,
held-out documents up to 2. A trailing window shorter than 512 tokens is kept
only if it has at least 64 tokens. No window crosses a document boundary.
Code prompts and math questions are each one sequence per item, truncated to
512 tokens if longer. Code prompts are chosen as the 40 lowest numeric
`task_id`s; math questions are chosen as the 40 lowest SHA-256 hashes of their
normalized text.

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
  evaluated window: max absolute logit difference `<= 0.01` and mean KL
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

- Unit of analysis: article for wikitext (windows within an article are
  averaged first), item for a code prompt or math question. All arms see the
  same inputs (paired). This clusters the bootstrap by document, so the
  dependence between windows of one article cannot shrink the CI.
- CIs: paired percentile bootstrap over articles/items, 10,000 resamples, RNG
  seed `20261114`, implemented with `numpy.random.Generator(PCG64(seed))` and
  the same resampled index matrix for every arm of a comparison. 2.5th and
  97.5th percentiles.
- Effect size: for a paired comparison, the mean per-article difference, its
  CI, and the relative reduction `(KL_baseline - KL_arm) / KL_baseline`.
- Confirmatory comparisons: K2 (best compliant arm vs acceptable tier), K3
  (non-uniform vs matched-budget uniform), K4 (transfer). The rest of the
  frontier is descriptive; its CIs are reported but no multiplicity-adjusted
  significance claim is made.
- Locale and platform: bootstrap uses only integer article/item indices and float
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

### A1 — 2026-11-14T21:22Z, before any measurement

The dataset was fetched and its structure inspected. Two corrections were made
before any frontier run, so no result has been seen or can have influenced
them:

1. The wikitext-2 card tags `license:cc-by-sa-3.0` and `license:gfdl`; the
   original text said CC BY-SA 4.0. Corrected.
2. The validation split contains 60 articles, not enough to give 64 distinct
   one-sequence documents. The unit of analysis and the chunking rule were
   therefore changed to article-level documents with multiple non-overlapping
   windows, and the bootstrap clusters by article. Sample-size minima were
   restated accordingly.

No threshold, arm definition, selector definition or kill condition was
changed. The images, arms, selectors, metrics, thresholds and gates of sections
4-10 stand as registered.

### A2 — 2026-11-14T21:31:50Z, before any sweep

The registered frequency selector had a formula/gloss conflict: it wrote
`mass_g(t) > 1/(2*16)` but glossed that as "twice the uniform share". Those are
different thresholds (half and twice the uniform share of `1/16`). The first
calibration-statistics run (`results/frontier/stats-Qwen-Qwen3-1.7B-20261114T213138Z.json`)
showed the half-share formula gives a frequency of approximately 1.0 for every
group in every layer, which makes the selector numerically degenerate. The
gloss is adopted: the threshold is `mass_g(t) > 2/16`.

No sweep, held-out measurement or quality number existed when this was decided.
The calibration statistics are a selector sanity check, not a result. The
receipt for the corrected run supersedes the first one; both are retained.

# Experiment contract — M2: per-token causal selection, oracle potential, and matched-budget selectors

Status: **pre-registered**. Appended after the M1 frontier was measured and
reported (`results/M1-FRONTIER.md`). It does not change any M1 threshold, arm or
kill condition; the M1 result stands as registered and reported. M2 tests the
hypothesis that M1 explicitly did **not** test: token-dependent, causal
selection at the same realized budget, together with the oracle-potential
ceiling that decides whether the inference-only branch is viable at all.

- Registered (UTC): 2026-11-14T22:38:44Z
- Predecessor facts (not re-tested here): `results/M1-FRONTIER.md`,
  `docs/BUDGET-MATH.md`, `data/dataset-spec.json`,
  `results/splits/split-20261114T223127Z.json`,
  `models/MODELS.lock.json`, `results/accounting/20261114T201402Z-qwen3-1.7b.json`.
- The registration commit is the commit that introduces this section. It cannot
  contain its own SHA, so every M2 receipt carries
  `preregistration_commit` set to that SHA.

## M2.1 Question

> **Amendment A4 (2026-11-15, after the formal review):** the `deployable` flag
> used in the M2 text below is superseded by two independent typed verdicts,
> `CausalAvailability` and `PhysicalCompliance`, in every receipt. "Deployable
> arm" in the K6 and M2.13 text is read as "causally available arm"; it never
> meant physically compliant. The correction of record, including the measured
> selector and execution reads, is `docs/M2-CORRECTION.md`.

M1 measured **static, token-independent** whole-group omission: one mask per
layer for every token. It rejected H1 at the ceiling for Qwen3-1.7B. The
measured masks used calibration statistics only; no per-token, causal,
gradient, interaction-aware or learned selector was tested.

M2 asks three questions, in order of information value:

1. **Oracle potential.** If a non-deployable per-token oracle could choose a
   different group subset for every token position at the same realized budget,
   how close could the masked model get to the dense f32 teacher? If even that
   oracle is outside the registered usable tier, no selector of this family can
   rescue inference-only sparsification of this teacher, and the program must
   move to the trained/upcycled branch.
2. **Granularity.** Is the failure caused by the budget, or by the granularity
   of sixteen equal contiguous groups per layer? Repeat the oracle at 64 and 4
   groups per layer with the total active element count held fixed.
3. **Deployable selection.** At exactly matched realized active unique
   parameters, does any deployable per-token selector beat the strongest
   implementable baseline (activation magnitude) with a paired-bootstrap CI
   that excludes zero, and does any approach the oracle?

These are distributional-fidelity questions, as in M1. KL to the teacher is not
a capability benchmark.

## M2.2 Registered budgets (fixed before measurement)

The accounting receipt is
`results/accounting/20261114T201402Z-qwen3-1.7b.json` (revision
`9bccaffe8b4f3e11113e0c389d8fb3418ab779a4`). Its registered fields:

- ceiling: 1,000,000,000 active unique learned elements per output token;
- always-active backbone: 663,610,368 elements;
- removable FFN: 1,056,964,608 elements;
- maximum compliant FFN elements per token: 336,389,632;
- one of the sixteen contiguous groups: 2,359,296 elements
  (`3 * 384 * 2048`, where 384 is the group width and 2048 the hidden width);
- full FFN per layer: 37,748,736 elements; intermediate channels: 6144;
  layers: 28; hidden: 2048.

Three budgets are registered. Every arm's realized selection is written to a
fixture and `selvedge atlas compare` recomputes the realized active unique
element count in Rust from that selection and the receipt-sourced component
counts. A comparison is valid only if every arm in it has the **same** realized
count and that count is `<= 1e9`.

- **B140 — the 5/16-per-layer-equivalent budget (primary for the granularity
  study and the 5/16 baseline).** Exactly five of the sixteen groups in every
  layer: 140 groups, `140 * 2,359,296 = 330,301,440` FFN elements,
  `663,610,368 + 330,301,440 = 993,911,808` active unique elements. This is the
  budget M1 called `uniform-magnitude-k5`, and it is 6,088,192 FFN elements
  below the ceiling's FFN affordability because the ceiling cannot add a whole
  group to every layer.
- **B142 — the ceiling-maximum whole-group budget (primary for the decisive
  matched-budget selector table).** The largest whole-group allocation at or
  below the ceiling: 142 groups, `142 * 2,359,296 = 335,020,032` FFN elements,
  `663,610,368 + 335,020,032 = 998,630,400` active unique elements. The
  unspent headroom is 1,369,600 elements, less than one group. This is the same
  realized count as M1's `greedy-uniform5`, so the M1 best static arm is
  exactly matched here. Arms allocate the 142 groups globally (five per layer
  plus two extras chosen by the arm's own score), not uniformly.
- **Granularity budget G = 330,301,440 FFN elements** (equal to B140's FFN
  share), used to compare 16, 64 and 4 groups per layer at identical active
  elements:
  - 16 groups: width 384; 5 per layer = 140 groups, 53,760 channels.
  - 64 groups: width 96; 20 per layer = 560 groups, 53,760 channels.
  - 4 groups: width 1536; 35 groups total = 53,760 channels, allocated as 21
    layers with one group and 7 layers with two groups (the oracle chooses
    which seven).

**Controller parameters count.** Every learned selector weight (unary and
interaction matrices, feature scaling constants fitted from data) is added to
the arm's active unique element count. Static calibration ranking tables and
realized masks are controller state, not learned weights, and are not counted;
this is the same convention as M1 and is stated in every receipt. The learned
controller must fit the B142 headroom of 1,369,600 elements. The registered
model sizes are far below it (unary: 16 groups x 6 features plus a layer term;
interaction: a 6 x 6 matrix), and the receipt prints the exact count.

## M2.3 Hypotheses

- **H2.1 (oracle potential).** A per-token oracle at B142 can reach the
  registered usable tier (`KL <= 0.25`, PPL ratio `<= 1.10`, top-1 `>= 0.85`)
  on the oracle sample. If false, the inference-only branch is dead at this
  granularity and budget.
- **H2.2 (granularity).** Finer fragments at the same active element count
  (64 groups per layer, 20 per layer) bring the oracle closer to the teacher
  than 16 groups, and 4 groups is worse. This distinguishes a budget
  constraint from a granularity constraint.
- **H2.3 (deployable selection).** At least one deployable per-token selector
  beats `token-magnitude` by at least 5% relative mean KL on held-out general
  text, with a paired-bootstrap 95% CI lower bound above zero, at exactly
  matched realized active unique parameters.
- **H2.4 (attribution and interaction).** A learned predictor trained on
  intervention-labeled local utilities beats activation magnitude, and the
  interaction-aware predictor beats the unary predictor by at least 5%
  relative KL with a CI excluding zero. This is the M2 version of H1.
- **H2.5 (interaction structure).** A low-rank or ridge surrogate fitted on
  calibration pairs predicts held-out pairwise interaction with a Spearman
  correlation at least 0.2 and a sign accuracy above 0.5 + its own 95% CI
  lower bound.

## M2.4 Arms (all at B142 unless stated; every arm has a `deployable` flag)

Reference and M1 baselines:

- `teacher-dense`: unmodified dense f32 model. Noncompliant by construction;
  the reference for every metric.
- `masked-all`: every group selected through the masked path; must equal the
  teacher within the M1 G1 tolerance. Instrument gate.
- `static-magnitude-k5` (B140): M1's per-layer top-5 magnitude mask, exactly
  matched at B140.
- `static-frequency-k5` (B140): M1's per-layer top-5 frequency mask.
- `static-random-k5` (B140): M1's seeded per-layer random top-5.
- `static-greedy-swap4` (112 groups, `927,851,520`): M1's registered K3 arm.
  Not budget-matched; included as a baseline and flagged.
- `static-greedy-uniform5` (142 groups, `998,630,400`): M1's best compliant
  static arm, exactly matched at B142.

Deployable per-token arms at B142:

- `token-random`: for every token and layer, five groups from a deterministic
  hash of `(seed, layer, position, token id prefix hash)`. No calibration
  information. This is the chance baseline.
- `token-magnitude`: per token, per layer, the five groups with the largest
  `||a_g(t)||_2`, where `a_g(t)` is the group's intermediate activation from
  the token's pre-decision hidden state. This is the **strongest implementable
  baseline** for K6.
- `token-frequency`: per token, per layer, the five groups with the largest
  `mass_g(t) = ||a_g(t)|| / sum_j ||a_j(t)||`. The static frequency statistic
  is replaced by its per-token value. This is a deployable per-token control,
  not the M1 static frequency arm.
- `token-task-static-nn`: a per-document static mask. Each calibration
  document gets a top-5-per-layer magnitude mask from its own activation
  statistics; at inference the arm computes the word 5-shingle Jaccard between
  the input document's token text and each calibration document, picks the
  nearest calibration document, and uses its mask for the whole document. The
  assignment uses only the input text, never a gold task label. This is the
  SEAP/DoMoE-adjacent "task-static" hypothesis at our data scale.
- `token-learned-unary`: a linear model on pre-decision group features
  (section M2.7), trained on calibration intervention labels, selecting five
  groups per layer by predicted score.
- `token-learned-interaction`: the same unary model plus a bilinear pairwise
  term, with the interaction weight chosen on the calibration validation split,
  selecting five groups per layer by greedy maximization of the surrogate.

Attribution arms at B142 (labelled `deployable: false` where stated):

- `token-gxo`: per token, per layer, the five groups with the largest signed
  sum `sum_{i in g} x_i * grad_i`, where `x` is the intermediate activation and
  `grad` is the gradient of the teacher-forced next-token log-probability. The
  gradient target uses the gold next token and a summed loss, so this arm is
  **non-deployable**; it is an attribution baseline, not an online selector.
  Its runtime cost includes a backward pass.
- `token-cor-gxo`: the published corrected GxO metric of Song et al.,
  *Attribution-based Sparse Activation in Large Language Models* (Proc. MLSys
  2026, PMCID PMC13570667), section 5.2 and Theorem 5.3, implemented as the
  paper states: the per-neuron score is
  `x_i * grad_i + 0.5 * x_i * (sum_k grad_k)^2`, where the sum is over all
  intermediate channels of the layer. The group score is the sum over its
  channels. The same non-deployable label applies. This is a reproduction of
  the stated corrective term, not of the paper's full experimental pipeline.
- `token-gxo-causal`: on the 8-window oracle sample only, a causal variant that
  backpropagates each position's own loss separately so no future token target
  enters a decision at an earlier position. Labelled experimental and reported
  separately; it is included to check whether the non-causal summed-loss GxO
  ranking is materially different.

Non-deployable oracle arms (section M2.5):

- `oracle-local-16` at B140 and B142;
- `oracle-local-64` at G;
- `oracle-local-4` at G;
- `oracle-final-refine` on a 32-position sample (bounded exact-KL swaps).

The oracle arms use the teacher only through the measured KL of a candidate
map; they never feed teacher logits into a deployable selector. They are
explicitly non-deployable and their search cost is recorded.

## M2.5 Oracle protocol (registered before measurement)

For a window of `T` predicting positions:

1. Run a forward with the current per-token mask map and capture the pre-MLP
   hidden state `h_{l,t}` for every layer `l` and position `t`.
2. For every `(l,t)`, compute the layer's intermediate activation
   `a = SiLU(W_gate h) * (W_up h)` and each group's output contribution
   `c_g = a[G_g] W_down[:, G_g]^T` (2048 floats). Let
   `r0 = sum_g c_g` (the dense layer output). Define the exact local squared
   error of a subset `S` as `L(S) = ||r0 - sum_{g in S} c_g||^2`. It is computed
   from the Gram matrix `G_{gj} = c_g . c_j` and `b_g = r0 . c_g`:
   `L(S) = ||r0||^2 - 2 sum_{g in S} b_g + sum_{g,j in S} G_{gj}`.
3. B140: enumerate all `C(16,5) = 4368` five-subsets per `(l,t)` and choose an
   exact minimizer of `L(S)`. B142: choose the exact five-subset per `(l,t)`,
   then add two extra groups per token, greedily over all `(layer, group)`
   pairs by the largest reduction in `L(S)` (the layer-local objectives are
   additive across layers, so this is exact given the five-subsets).
4. Apply the resulting per-token mask map in a model forward and measure the
   exact per-position KL, top-1 and NLL against the dense teacher.
5. Repeat steps 1-4 for up to three passes, so that the local decisions see
   hidden states produced by the arm's own masks. Keep the map with the lowest
   measured mean KL on the sample; record every pass.
6. The granularity arms replace step 3:
   - 64 groups, 20 per layer: beam search of width 8 over the 20 additions,
     scored by the exact local `L(S)` (incremental Gram updates);
   - 4 groups, 35 total: exact enumeration of all one-group and two-group
     subsets per layer, then add the second group to the seven layers with the
     largest local reduction (exact for the additive local objective).
7. `oracle-final-refine` starts from the best `oracle-local-16` map and, for a
   sample of 32 token positions, tries the four single-swap candidates per
   layer with the largest local `L(S)` improvement (4 candidates x 28 layers x
   2 passes), measuring the token's own exact final KL with a prefix forward
   (the prefix contains only positions `<= t`, so no future token enters the
   decision). The map is accepted only if the token's exact KL improves.
8. The oracle is non-deployable: it needs the per-token subset search, several
   full forwards per pass and, for the final refinement, prefix forwards. The
   receipt records forwards, subset evaluations, selector seconds and GPU
   seconds. Those are not part of any compliant runtime claim.

Oracle sample: the first 512-token window of each of the first eight held-out
general articles, in the order of `results/splits/split-20261114T223127Z.json`
(8 windows, 4,096 predicting positions). The eight articles are held out from
every calibration and from every selector. No held-out text is used to choose
a deployable selector; it is used only to measure the oracle and the arms.

Reported oracle quantities:

- mean, median, p95 and max per-position KL of the best map;
- top-1 agreement and PPL ratio;
- fraction of positions with KL `<= 0.05` (the acceptable per-position bound),
  `<= 0.25` (usable), and with correct top-1;
- the same quantities for the per-position minimum over the three passes,
  labelled "oracle over passes, upper bound" (it is not a single execution);
- the number of distinct subsets used per layer, the fraction of adjacent token
  pairs with an identical subset, and the mean Jaccard between the subset
  distributions of two disjoint oracle windows.

## M2.6 Cross-token consistency

For each layer, on the oracle sample: (a) the fraction of adjacent positions
with the same selected subset; (b) the number of distinct subsets used; (c) the
fraction of positions whose subset equals the window's modal subset; (d) the
mean pairwise Jaccard between the per-position subset sets of two disjoint
windows from different articles. These seed the M3 execution-envelope design:
if the subset changes on nearly every token, no envelope can cover it.

## M2.7 Attribution, interaction and learned-selector protocol

**Feature vector (pre-decision only).** For a token at layer `l`, the allowed
inputs are the pre-MLP hidden state `h`, the layer index, the position index,
and quantities computed from `h` and the layer's own gate/up/down weights with
no backward pass and no teacher: for each group `g`,
`[log1p(||a_g||), ||a_g||/sum_j||a_j||, mean(a_g), max(a_g), min(a_g)]`, plus a
global `l/L` term and a bias. Raw `h` is allowed but is not used by the
registered learned arms; the atlas input record carries it for auditability.

**Training labels (intervention-measured, calibration only).** For a
calibration context `(window, layer, token)`, fix the base set
`S0` = the four `token-magnitude` groups of that layer at that token. Compute
`L(S)` exactly by the Gram method. The unary label is
`delta_g = L(S0) - L(S0 union {g})` for all 16 groups. The pairwise label is
`I_local(g,j) = L(S0 union {g}) + L(S0 union {j}) - L(S0) - L(S0 union {g,j})`
for sampled pairs `g,j not in S0`. Both are exact one-layer interventions on
the layer output, not final-KL labels; they are labelled as such.

**Sampling for training.** 4,000 `(window, layer, token)` contexts sampled
uniformly without replacement from the 63 calibration windows and 28 layers;
8,000 pairs from the same contexts. Training/validation split by calibration
article: the 12 lowest-hash calibration articles train, the 4 highest-hash
articles validate. The held-out split and the code/math slices are never used.
Standardise features on the training split; store the constants in the
weights JSON.

**Models.**
- Unary: `score_g = w . phi_g + v * (l/L) + b`. Fit by ridge regression
  (closed form, f64) on the unary labels; the ridge coefficient is chosen from
  the pre-registered grid `{1e-4, 1e-3, 1e-2, 1e-1}` on the validation local
  objective.
- Interaction: `score(S) = sum_{g in S} score_g + sum_{g<j in S} phi_g^T M phi_j`,
  with `M` a 6x6 matrix fitted by ridge on the pairwise labels. The mixing
  weight `lambda` on the pairwise term is chosen from `{0.25, 0.5, 1.0, 2.0}`
  on the validation local objective. Selection is greedy: start from the empty
  set, add five groups per layer, each time maximizing the surrogate.
- The weights are exported to `results/selectors/atlas-weights.json` with a
  documented schema, feature names, per-layer/global parameters, the exact
  training receipt hashes and the exact parameter count.

**Attachment/interaction measurement.** On the calibration interaction sample
(8 calibration windows x layers {0,7,14,21,27} x 2 stratified pairs), with the
static `static-magnitude-k5` map in every layer except the probed layer `L`,
measure the exact sequence-mean KL for the four masks `S0`, `S0+g`, `S0+j`,
`S0+g+j`. The pair is positive (synergy) if
`I = KL(S0+g) + KL(S0+j) - KL(S0) - KL(S0+g+j)` is positive, negative
(antagonism) otherwise. Because `I` is algebraically symmetric in `g,j`, the
"sign asymmetry" reported is the asymmetry that is actually measurable:
(a) the fraction of pairs whose sign changes between contexts (base-set
dependence), (b) the distribution of the sequential marginal gains
`Delta(g|S0)` and `Delta(j|S0)`, which need not have equal magnitude, and
(c) the numerical difference between the two orderings, which should be zero
and is reported as a correctness check. Report the mean, median, IQR, sign
fractions, the five largest positive and five largest negative contexts, and
a ridge/low-rank surrogate predicting `I` on held-out calibration articles
(Spearman, sign accuracy, CI by article bootstrap).

## M2.8 Metrics (fixed before measurement)

Primary metrics are the M1 definitions: teacher KL in nats/token, top-1
agreement, PPL ratio, per domain (general/code/math), paired by item. Added for
M2:

- per-position KL distribution: median, p95, fraction `<= 0.05`, fraction
  `<= 0.25`, fraction with correct top-1;
- realized active unique elements and compliance, recomputed in Rust from the
  realized selection plus the learned controller parameter count;
- selector cost: selector seconds per window, selector FLOPs per token
  (formula, explicitly derived), controller parameter count, and, for the
  attribution arms, backward-pass cost. The controller cost is part of the
  comparison and is reported next to every arm.
- oracle cost: forwards, subset evaluations, prefix forwards, GPU seconds.

## M2.9 Data, splits and sample sizes

The M1 split is reused unchanged and hash-checked:
`results/splits/split-20261114T223127Z.json` (derived
`results/artifacts/split/m1-split.json`, SHA-256 in the split receipt). The
calibration and held-out sets are disjoint. The code and math slices remain
held out from every selector and every training label.

- Held-out evaluation: all 40 general articles (80 windows), 40 code items, 40
  math items. If the split receipt's counts differ, the run is invalid.
- Oracle sample: 8 general articles x first 512-token window (4,096 positions).
- Granularity: the same 8 windows.
- Interaction: 8 calibration windows x 5 layers x 2 pairs = 80 contexts, 320
  model forwards.
- Learned training: 12 calibration articles; validation 4 calibration articles;
  4,000 contexts and 8,000 pairs. Every sampled context is identified by
  `(doc_id, window_index, position, layer)` in the receipt.
- Seeds: 20261114 for token permutations, sampling, ridge and bootstrap. The
  deterministic permutation is `SplitMix64(seed ^ layer ^ position ^ prefix)`.

## M2.10 Leakage rule (non-negotiable)

A deployable selector may read only: the pre-MLP hidden state `h_{l,t}` at the
layer where it decides, the current layer's gate/up activations computed from
`h_{l,t}` and that layer's weights, the layer index, the current position
index, the token prefix strictly before the decision, and static constants
fitted on calibration data. It may **never** read: teacher logits, the dense
teacher's intermediate activations, future token ids, the gold next token, any
held-out text, any code/math evaluation item, or the realized mask/KL outcome
it is being scored on. `token-gxo`, `token-cor-gxo`, `token-gxo-causal` and all
oracle arms are labelled non-deployable precisely because their target uses
teacher-forced tokens or held-out measurements. The Rust `SelectorInput` record
has no field for future state; a Court B test rejects serialized inputs that
carry one.

## M2.11 Statistical method

The M1 method is reused: unit of analysis is the article for general text and
the item for code/math; windows within an article are averaged first; paired
percentile bootstrap over articles/items with 10,000 resamples, seed 20261114,
the same resampled index matrix for every arm in a comparison, and 2.5/97.5
percentiles. The bootstrap uses only integer indices and float accumulation.
CIs are reported for mean KL, PPL ratio, top-1, per-domain metrics, and for
every paired relative reduction. Interaction CIs resample calibration articles.

## M2.12 Kill conditions (fixed now)

- **K6 (deployable selector).** If no deployable arm achieves at least a 5%
  relative reduction in held-out general mean KL versus `token-magnitude` at
  the same realized active unique parameters, with a paired-bootstrap 95% CI
  lower bound above zero, then no deployable per-token selector is demonstrated
  to beat activation magnitude. Record K6 as failed in `docs/WORK-LOG.md` and
  recommend the pivot to the trained/upcycled branch, with the evidence.
- **K7 (oracle ceiling).** If the best per-token oracle at B140 or B142 has
  mean KL above 0.25, or PPL ratio above 1.10, or top-1 below 0.85, on the
  oracle sample, then per-token selection at this granularity and budget cannot
  reach even the registered usable tier. The inference-only branch is dead for
  this teacher; record K7 as failed and pivot. This is the decisive condition.
- **K8 (granularity).** If the 64-group oracle is also above the usable tier
  and the 4-group oracle is worse than the 16-group oracle, granularity is not
  the binding constraint. If the 64-group oracle reaches the usable tier while
  16 does not, granularity is binding and the next experiment is a
  fine-grained deployable selector. Report the exact comparison; do not
  generalise beyond the three measured granularities.
- **K9 (integrity and leakage).** Any selector that reads a forbidden input,
  any NaN/Inf, any realized active count above the ceiling, any mismatch
  between the Python arm decision and the Rust selector on the exported
  fixtures, or any failure of the instrument gate G1. Stop; publish the
  failure; do not report a quality number from the affected run.

## M2.13 What would justify continuing the inference-only branch

Continuation requires **both**:

1. the oracle (best of 16/64/4 groups at the matched element budget) reaches
   at least the usable tier on the oracle sample (`KL <= 0.25`, PPL ratio
   `<= 1.10`, top-1 `>= 0.85`); and
2. a deployable arm at exactly matched active unique parameters beats
   `token-magnitude` by at least 5% relative mean KL with a paired 95% CI lower
   bound above zero, and captures at least half of the oracle's relative
   improvement over the best static M1 arm.

If (1) holds but (2) does not, the branch is scientifically alive but not
deployable; the next step is a stronger learned approximation of the oracle,
not a runtime. If (1) fails, the branch is dead at this granularity and budget,
regardless of (2), and the trained/upcycled branch becomes the primary path.

## M2.14 What a negative result means

If K7 fails, the reading is: with this teacher, this grouping granularity (or
even 64 fragments per layer at the same elements), this budget, exact group
omission and no retraining, even an oracle that chooses a different subset per
token cannot preserve the teacher's next-token distribution within the usable
tier. That bounds the inference-only, post-hoc selection family for this
teacher. It does not bound trained sparse models, upcycling, MoE conversion,
quantization or a different teacher. It also does not show that the selectors
are useless for a model trained to tolerate sparsity; that is the branch the
result would recommend.

If K7 passes but K6 fails, the reading is: the potential exists but our
deployable predictors do not capture it. The next experiment is a better
predictor, and the Rust selector infrastructure is still the right place to
measure it.

## M2.15 Deviations and amendments

As in M1: no planned deviations; unplanned ones are recorded in
`docs/WORK-LOG.md` and in the affected receipt with reason and timestamp. A
change to a budget, arm, selector definition, sample size, threshold or kill
condition after the first measurement is an amendment appended here with its
own timestamp, and the pre-amendment results are reported against the original
text.

### A3 — 2026-11-14T23:04Z, after the M2 oracle and full-selector measurements

The registered M2 text required every arm in a comparison to have exactly the
same realized active unique element count. The measurements show that exact
equality is impossible at whole-group granularity once a learned controller is
counted:

- the smallest removable FFN unit in this bundle is one channel, which is
  `3 * 2048 = 6,144` learned elements (a group is 384 channels,
  `2,359,296` elements);
- the registered learned controller has 868 learned elements, which is not a
  multiple of 6,144, so no whole-channel allocation can make its count
  element-identical to a controller-free arm at the same group count.

The realized counts are reported exactly in every receipt and in
`results/selectors/table.md`. The operative comparison rule is therefore
restated, without changing any threshold, arm or kill condition:

1. arms are compared at the same whole-group allocation (142 groups for the
   B142 confirmatory table);
2. the learned controller's weights are counted in that arm's active unique
   parameters, as the hard rule requires;
3. a comparison is budget-matched when the realized counts differ only by the
   controller and that difference is at most one channel (`<= 6,144`
   elements). The difference and its direction are printed for every
   comparison. A learned arm is never *below* the baseline in this rule, so it
   cannot obtain a hidden budget advantage; here the learned arms are 868
   elements (0.000087% of the budget) above `token-magnitude`;
4. the arms that are not matched at all (`static-greedy-swap4`, 112 groups;
   and the B140 arms in the B142 table) remain flagged and are excluded from
   the K6 comparison.

Why this does not change the conclusion: the confirmatory K6 test compares
each learned arm with `token-magnitude`; the learned arms have more active
parameters and a worse KL, so the 868-element difference cannot rescue them.
The pre-amendment measurements and their `matched_budget=false` labels are
retained in `results/selectors/eval-full-20261114T230320Z.json` and
`results/selectors/selectors.json`; the post-amendment analysis is the one
cited by `results/M2-SELECTORS.md`.

### A4 — 2026-11-15, after the M2 formal review

The registered M2 text assigned every arm a single `deployable` flag. A formal
review found that this conflated two independent questions and that the M2
instrument did not measure the parameter reads it charged. No budget, arm,
selector definition, sample size, threshold or kill condition changes. The
following definitions replace the flag in every future receipt; the historical
text above is retained unchanged as the pre-registration.

1. **`CausalAvailability`** is a typed, evidence-bearing verdict: either
   `Available { decision_record, allowed_inputs, audit, one_pass }` or
   `Unavailable { reason, detail }` with one of `future_tokens`,
   `future_layers`, `teacher_logits`, `gold_labels`, `unmasked_future_state`,
   `held_out_measurement`, `backward_target`, `undeclared_input`. The K6 rule
   is read as applying to causally available arms. The old phrase "deployable
   arm" in K6 and M2.13 means "causally available arm"; it never meant
   physically compliant.
2. **`PhysicalCompliance`** is a typed, evidence-bearing verdict: either
   `Compliant { evidence }` with measured selector reads, execution reads,
   their union, and FLOPs, or `NonCompliant { reason, evidence }` with
   `dense_computation_masked`, `selection_reads_outside_active_set`,
   `missing_sparse_execution_path`, `unmeasured_parameter_reads` or
   `over_ceiling`.
3. The M2 `token-magnitude`, `token-frequency` and learned arms read every
   gate/up channel to score groups and executed dense projections with a mask.
   They are `selection_reads_outside_active_set` and
   `dense_computation_masked`. Their reported active counts were logical
   selection counts. This is recorded in `docs/M2-CORRECTION.md`.
4. The corrected accounting and the sparse execution path are an instrument
   change, not an arm change. The pre-correction receipts remain valid for the
   logical-omission and final-KL comparisons they contain, and are not
   comparable on the physical axis. The corrected table is generated from
   `results/selectors/eval-full-20261115T005000Z.json`.

# Experiment contract — T1: trained / upcycled compliant architectures and the dense control

Status: **pre-registered before any training-data pass or baseline evaluation
that informs a threshold.** This section is appended to the M1/M2 contract and
does not modify it. M1 and M2 rejected the inference-only, post-hoc selection
branch for Qwen3-1.7B: even a causally unavailable per-token oracle at the
compliant budget was outside the usable tier, and the best physically compliant
arms were static masks with unusable quality. T1 asks the next question the
mandate requires: **if the model is trained to tolerate the compliant budget,
does conditional computation buy quality that a dense model of the same size and
cost cannot?**

The decisive comparison is arm B against arm C at matched active parameters,
matched tokens and matched training compute. If B does not beat C, the
conditional-computation claim is rejected at this scale (K13), whatever B does
against the untrained baseline D.

## T1.1 Registered model facts

All counts below are the receipts' unique, tie-aware counts, not total tensor
elements. The ceiling is `1,000,000,000` unique active learned parameters per
output token, enforced by the Rust ledger path (`selvedge admit`, T1.9), not by
a Python assertion.

| model | unique learned | always-on backbone | removable FFN | verdict |
| --- | ---: | ---: | ---: | --- |
| Qwen3-0.6B | 596,049,920 | 331,808,768 | 264,241,152 | not binding |
| Qwen3-1.7B | 1,720,574,976 | 663,610,368 | 1,056,964,608 | binding, backbone fits |

Qwen3-0.6B: 28 layers, hidden 1024, intermediate 3072, 16 heads / 8 KV heads,
head_dim 128, vocab 151936, tied embeddings. Qwen3-1.7B: 28 layers, hidden
2048, intermediate 6144. Both Apache-2.0; revisions and weight hashes are in
`docs/MODELS.md`. One group of the default 16-group partition is 2,359,296
elements for the 1.7B FFN and 589,824 for the 0.6B FFN.

## T1.2 Architectures (all arms)

Every arm is a next-token model. "Active" means the union of learned parameter
elements read by the router **and** the executed computation for one output
token, counted once. Frozen weights are still learned parameters and are counted
when read. Router and adapter parameters are counted. Parameter counts below
are design counts; the authoritative count for every run is the Rust admission
record in T1.9, and a mismatch between the design count and the admission record
fails closed (K10).

**D — dense baseline, no training.** Unmodified `Qwen/Qwen3-0.6B` at revision
`c1899de289a04d12100db370d81485cdf75e47ca`. Active unique `596,049,920`;
compliant. This is the model a more complex architecture must beat at all.

**E — quality reference, non-compliant, never in compliant results.**
Unmodified `Qwen/Qwen3-1.7B` at revision
`70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`. Active unique `1,720,574,976`;
`research_noncompliant_teacher`. It is the distillation teacher and the quality
ceiling. No compliant comparison may include it.

**A — sparse-upcycled Qwen3-1.7B.** Frozen 1.7B backbone (`663,610,368`) plus,
per layer, a trainable router that selects `k = 5` of 16 contiguous FFN channel
groups, plus trainable rank-32 residual adapters on the selected groups. Each
selected group's gate/up/down slices are read; unselected groups are never
computed. The adapter for group `g` acts on the group's 384-wide pre-down
activation `a_g = SiLU(W_gate_g x) * (W_up_g x)`:
`a_g -> a_g + B_g (A_g a_g)` with `A_g: R^384 -> R^32` and `B_g: R^32 -> R^384`.
`B_g` is zero-initialised. Router: one `d_model x 16` linear per layer, no bias,
softmax over 16 logits, hard top-5 forward with a straight-through gradient.

Active unique is
`663,610,368 + 28*5*2,359,296 + 28*5*(384*32 + 32*384) + 28*2048*16`
= `663,610,368 + 330,301,440 + 3,440,640 + 917,504` = **`998,269,952`**.
Compliant. (The adapter is a low-rank factorisation over the 384-wide group
activation: `r*(384+384)` elements per selected group. Correction of record:
the first committed text wrote the adapter as acting on the hidden state and
counted `r*(384+2048)`; the code, the admission config and this number were
corrected before any measurement, with no change to any threshold, arm or kill
condition. See T1.15 A5.)

**A0 — untrained A reference.** Arm A's architecture with the router at its
seed-1001 initialisation and zero adapters, no training. This is the
M2-comparable point (per-token selection at k=5 with an untrained router) and it
lets the report say whether training rescues omission. Compliant, same count as
A.

**B — 0.6B core plus routed specialist capacity.** Frozen unmodified 0.6B core
(`596,049,920`, including its dense FFN) plus, per layer, `m = 8` specialist
SwiGLU FFN blocks of intermediate width 512 and a router selecting `k = 4` of
them. Each specialist is `W_gate: 1024x512`, `W_up: 1024x512`, `W_down:
512x1024`; its output is added residually to the layer output; `W_down` is
zero-initialised so step 0 is exactly the frozen core. Router: `d_model x 8`
linear per layer, no bias, hard top-4, straight-through.

Active unique is `596,049,920 + 28*4*1,572,864 + 28*1024*8`
= `596,049,920 + 176,160,768 + 229,376` = **`772,440,064`**. Compliant.

**C — dense control at matched active parameters.** Frozen 0.6B core plus a
*dense* (unrouted) SwiGLU adapter of intermediate width 2048 in every layer,
with `W_down` zero-initialised and added residually: `3*1024*2048 = 6,291,456`
elements per layer, `176,160,768` in total. Active unique is
`596,049,920 + 176,160,768` = **`772,210,688`**. Compliant. The only difference
between B and C is conditional (routed) versus dense computation at the same
active parameter count; the `229,376`-element router makes B *more* expensive,
never cheaper, which is the A3 direction rule carried forward.

**C2 — ceiling-max dense control.** Frozen 0.6B core plus a dense SwiGLU
adapter of intermediate width 4608 in every layer: `3*1024*4608 = 14,155,776`
per layer, `396,361,728` total. Active unique is
`596,049,920 + 396,361,728` = **`992,411,648`**. Compliant. C2 answers "is the
ceiling better spent densely" but is not parameter-matched to B, so it is a
secondary arm and is flagged as such in every table.

Arms A, A0, B, C and C2 are in *approximate* mode whenever selection changes
which groups execute; the M1/M2 exact-mode distinction is unchanged and no
exact-mode claim is made here. Every trained arm's report carries
`CausalAvailability::Available` (the router reads only the current layer's
pre-FFN hidden state, the layer index and its own parameters; no future token,
teacher logit or gold label) and `PhysicalCompliance::Compliant` with measured
reads from the Rust admission record.

**Step-0 identity.** For B, C and C2 the zero-initialised `W_down` makes the
adapter's contribution exactly zero, so the modified model must reproduce the
frozen 0.6B core's logits within the bf16 instrument tolerance
`max |logit_delta| <= 1e-3` on the registered verification batch (expected
exactly `0`, since adding a zero tensor is exact). For A, the same check is run
with `k = 16` (all groups) so the omitted groups contribute too; at the
registered `k = 5` the model does **not** match the dense teacher, by design.
The omission delta at `k = 5` is measured and reported, never hidden. A failure
of the zero-point check is K12.

## T1.3 Data sources, revisions and licences

Spec: `data/train-corpus-spec.json`; lock: `data/train-corpus.lock.json`; fetch
receipt: `results/train/fetch-*.json`. Files are downloaded only by an explicit
`--net` court step, hashed against the Hugging Face LFS object id, and never
committed. The corpus text lives in the gitignored
`results/artifacts/train-corpus/`.

| source | revision | file | licence | role |
| --- | --- | --- | --- | --- |
| `HuggingFaceFW/fineweb-edu`, config `sample-10BT` | `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9` | `sample/10BT/013_00000.parquet` | ODC-By 1.0 (`license:odc-by`) | text, 70% of tokens |
| `codeparrot/codeparrot-clean` | `35a59fb025bc0a102f7d96eac09d145b896d487b` | `file-000000000001.json.gz` | per-row `license` field; allowlist below | code, 20% of tokens |
| `EleutherAI/hendrycks_math`, all seven configs, **train** splits only | `21a5633873b6a120296cce3e2df9d5550074f4a3` | `*/train-00000-of-00001.parquet` | MIT | math, 10% of tokens |

`codeparrot/codeparrot-clean` declares no dataset-level licence. Its rows carry
a per-file `license` field, so the corpus is the subset whose row licence is in
the exact allowlist
`{mit, apache-2.0, bsd-3-clause, bsd-2-clause, isc, mpl-2.0, unlicense,
cc0-1.0}`; rows with any other value, including `unknown`, are discarded and
counted. This is a row-level licence filter, not a dataset-level claim, and is
recorded as a residual risk in `docs/LIMITATIONS.md`. The MATH **test** splits
are never read. GSM8K, HumanEval and TruthfulQA are not in any training source,
and are similarity-filtered against the corpus in T1.5 anyway.

**Deterministic materialisation.** Documents from each source are ordered by the
SHA-256 of their normalised text (never by file position), concatenated per
source with one end-of-text separator between documents, tokenised with the
pinned Qwen3 tokenizer, and interleaved into one stream by a fixed 70:20:10
block schedule at 1024-token granularity. The stream is written as
little-endian `uint32` to `train.bin` with a sidecar `train.index.json`
recording the SHA-256, the token count and the per-source spans. Every arm reads
the same file from offset 0 in the same order. A source that runs out of tokens
repeats from its own start (the MATH source is small and is repeated); the
repeat factor is recorded in the index. The corpus is materialised to the
registered budget plus 10% headroom; no arm reads beyond the registered budget.

## T1.4 Teacher targets

Teacher: arm E, dense Qwen3-1.7B, bf16, `torch.inference_mode`, TF32 disabled.
Targets are computed **once** over the materialised stream, in chunks of 4096
tokens (4 x 1024), and cached so the teacher is never re-run per arm.

Per position the cache stores the top-16 teacher logits in bf16, the top-16
vocabulary indices in int32, and the teacher log-sum-exp in fp32: exactly
`16*2 + 16*4 + 4 = 100` bytes per position. At the registered budget of
`134,217,728` tokens the cache is `13,421,772,800` bytes (12.5 GiB) plus a
manifest. The manifest records, per chunk, the token span, the SHA-256 of the
chunk file, the teacher revision and the row range; a mismatch fails closed.
Disk use is bounded by the registered budget plus headroom, and no chunk is
written without its checksum.

**Verification.** After the cache is built, a registered sample of 16 chunks is
recomputed by a live teacher forward and compared: the top-16 index sets must be
identical, the logits must agree within the bf16 tolerance
`|a-b| <= 1e-2`, and the log-sum-exp within `1e-3`. The sample, the deltas and
pass/fail are in the build receipt. A failure invalidates the cache and stops
training.

The objective is the top-k distillation cross-entropy with one tail bucket:
`L = -sum_j p_t(j) log p_s(j) - p_tail log p_s(tail)`, where `p_t` is the
teacher distribution restricted to its top-16 support plus a tail of mass
`1 - sum_j p_t(j)`, and `p_s` is the student's probability on the same support
plus the sum of the remaining student mass as the tail. The support term is
exact; the tail is a **single bucket**, so this is not the full-vocabulary
cross-entropy (the teacher's tail distribution is not stored, only its mass).
It is the objective the compact cache determines, it is identical for every arm,
and it is what every receipt's `loss` reports. No hard-label cross-entropy term
is added.

Teacher targets are derived from a non-compliant model. They are a training
signal only and are never part of a compliant inference path.

## T1.5 Contamination and leakage

Before any training token is produced, every candidate training document is
checked against every evaluation item (HumanEval prompts, GSM8K questions,
TruthfulQA questions, and the 40 held-out wikitext articles) by (a) exact 5-word
-shingle Jaccard at the pre-registered threshold `0.2`, and (b) an exact match
of any 64-token substring. A document failing either test is discarded and
counted per evaluation set. The counts appear in the corpus receipt; if any
evaluation item survives at Jaccard `>= 0.5` the corpus build fails closed
(K10).

No evaluation label, gold answer, canonical solution or test is ever tokenised
into the corpus. The router sees no teacher logits, no future token and no
evaluation item. The teacher cache is built on training text only.

Residual risk: fineweb-edu, codeparrot-clean and MATH are public and are
plausibly in Qwen3's pretraining corpus, so absolute teacher KL is optimistic.
All comparisons are relative to the same teacher on the same tokens.

## T1.6 Token budget, sequence length, optimiser, routing, seeds

- **Token budget.** `134,217,728` tokens (2^27) per full-budget arm. Registered
  before measurement from the analytical estimate in T1.14 and the 6-10 hour
  window. The smoke receipt either confirms it fits or triggers the single
  pre-run amendment in K14. No arm is silently shrunk.
- **Sequence length** 1024 tokens. No window crosses a 1024-token block
  boundary. The corpus separator token is the model's end-of-text token.
- **Batch.** Micro-batch 4 sequences x 1024 tokens, gradient accumulation 16,
  effective batch `65,536` tokens per optimiser step. `134,217,728 / 65,536 =
  2048` steps per full-budget arm.
- **Precision.** bf16 compute for the frozen backbone; fp32 master weights for
  every trainable adapter/router with AdamW. Rationale: the frozen backbone must
  fit in 16 GiB in bf16, while fp32 masters keep the adapters' updates stable at
  the registered learning rate.
- **Optimiser.** AdamW, `betas = (0.9, 0.95)`, `eps = 1e-8`, weight decay
  `0.0` on adapters and routers, gradient clip `1.0`. Peak LR `1e-3` for
  adapter/router parameters, linear warm-up over 100 steps, cosine decay to
  `1e-4` at the end of the registered budget. No parameter group uses a
  different schedule.
- **LoRA / adapter ranks.** A: rank 32 on the selected FFN groups. B: specialist
  width 512 (not low-rank; the specialist *is* the capacity). C: dense width
  2048. C2: dense width 4608.
- **Routing.** Hard top-k (A: k=5 of 16; B: k=4 of 8). Training uses a
  straight-through estimator: the forward gate is the hard 0/1 top-k mask; the
  backward pass treats the mask as the softmax probability. Eval uses the hard
  mask. A soft-mixture evaluation (all experts weighted by the softmax) is run
  as a registered ablation on the same items and the hard-minus-soft KL gap is
  reported per arm; the hard value is the primary.
- **Load balancing.** A Switch-style auxiliary loss
  `0.01 * m * sum_i f_i * p_i` is added to the distillation loss, where `f_i` is
  the fraction of tokens routed to expert `i` in the batch and `p_i` is its mean
  router probability. Expert utilisation is logged every step. The auxiliary
  loss is identical for A and B.
- **Seeds.** Every full-budget arm uses seed `1001` (torch, numpy and the data
  order derive from it). The decisive pair B/C is replicated at `33,554,432`
  tokens (2^25, 512 steps) with seeds `2001`, `2002`, `2003`; the replication is
  used only to estimate run-to-run spread and is reported as a seed-level CI.
- **Consequence of one full-budget seed.** A single full-budget seed cannot
  separate a small B-C difference from seed noise. The registered decision rule
  therefore requires the 3% relative margin to exceed **twice the seed-level
  standard deviation** measured in the 32M-token replication; if it does not,
  the result is reported as "not separable from seed noise" and K13 is not
  treated as passed.

## T1.7 Evaluation suite (fixed before measurement)

Instrument: bf16 model forward (TF32 disabled), logits cast to f32 for the
metrics, greedy decoding for pass@1 and exact match, fixed prompt templates in
`oracle/eval/tasks.py`, fixed item order, no sampling for primary metrics. This
is a **different instrument** from M1/M2, which used an f32 teacher; T1 KL values
are not comparable to M1/M2 KL values and no such comparison is made.

1. **HumanEval pass@1** (MIT, revision `7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544`),
   all 164 problems. Prompt is the dataset `prompt` field; greedy decode up to
   256 new tokens, stopping at the first of `\nclass `, `\ndef `, `\n#`,
   `\nif __name__`, or EOS. The completion is executed with the canonical test
   under a 3-second per-problem wall-clock limit in a subprocess with no
   network, no inherited file handles and a bounded address space; a timeout,
   crash or syntax error is a failure. pass@1 is the fraction of problems whose
   test passes. The code-execution risk is recorded in `docs/LIMITATIONS.md`.
2. **GSM8K exact match** (MIT, revision
   `740312add88f781978c0658806c59bc2815b9866`), the 300 problems with the
   lexicographically lowest SHA-256 of the normalised question (fixed before
   measurement). Prompt `Question: {q}\nAnswer:`; greedy decode up to 512
   tokens; the predicted value is the last number on the line after `####`, or
   the last integer in the output; exact match after stripping commas and
   leading zeros.
3. **TruthfulQA MC1** (Apache-2.0, revision
   `741b8276f2d1982aa3d5b832d3ee81ed3b896490`, `multiple_choice` validation),
   all 817 questions. For each question the model scores each supplied choice
   under the registered template `Q: {question}\nA: {choice}`; accuracy is the
   fraction where the argmax-scored choice has label 1. This is a closed-book
   factual-consistency set; it is permissively licensed and pinned, and its
   contamination risk is declared (T1.5).
4. **Teacher KL and PPL** on the existing pinned held-out split: 40 wikitext
   articles (up to two 512-token windows each), 40 HumanEval prompts and 40
   GSM8K questions from `results/splits/`. KL is `KL(teacher || student)` per
   position in nats/token, and PPL is the student's perplexity on the true token
   on the same tokens. Reported per domain and pooled.
5. **Mixed-task prompts.** A fixed interleaving of the three domain slices
   (wikitext window, HumanEval prompt, GSM8K question, repeating) measured with
   the same KL/PPL definitions, so a domain-specific gain cannot hide a pooled
   regression.
6. **Routing ablation.** For A and B only: hard top-k versus soft mixture KL on
   the same items, per domain. This is the reported train/eval routing gap.

**Strict no-training-on-eval.** No evaluation item, label or test is in the
corpus. The evaluation harness reads the pinned dataset cache and the model
checkpoint only; it never writes to the corpus or target cache.

## T1.8 Metrics, CIs and statistical method

- Primary metrics are the two co-primary endpoints:
  - `P1` = pooled held-out teacher KL in nats/token (lower is better), with the
    per-domain values reported alongside.
  - `P2` = task composite = arithmetic mean of HumanEval pass@1, GSM8K exact
    match and TruthfulQA MC1 accuracy (higher is better), each in `[0,1]`.
- Secondary: per-domain KL/PPL, top-1 agreement with the teacher, teacher KL on
  the mixed interleaving, expert utilisation, the routing gap, and wall-clock.
- CIs are paired percentile bootstraps with 10,000 resamples, seed `20261115`;
  KL resamples windows, HumanEval/GSM8K/TruthfulQA resample items. The
  B-minus-C and B-minus-D differences are computed on paired items, never on
  independent means.
- One full-budget seed per arm; the seed-level CI comes from the 32M-token
  replication (T1.6). Both are reported. A difference that is inside the
  item-level CI but outside the seed-level spread, or vice versa, is reported as
  ambiguous, not as a win.
- Multiple comparisons: there are two co-primary endpoints. A claim needs the
  threshold on at least one endpoint and no regression beyond the margin on the
  other. No other endpoint can be substituted after measurement.

## T1.9 Accounting enforcement (Rust is the source of truth)

`selvedge admit --config <arm-config.json> --out <admission.json>` builds a
`SelectionPath` from the declared selector and execution tensor slices, admits
it through `ParameterLedger::admit_selection_path` under
`ComplianceMode::Compliance`, and writes the admitted path, the verdict, the
selector/execution read counts, their union and the FLOP estimates. The command
exits non-zero and writes no admitted path for an over-budget arm.

The training and evaluation entry points refuse to start a config whose
admission record is missing, is not `Within { .. }`, or whose config SHA-256 does
not match the record. A Python-side mirror of the union computation is run as a
cross-check on every run and its disagreement with the Rust record is a hard
error (K10). The Python mirror is never the source of truth.

Routers and adapters are declared in the same config and counted:
`selector_uses` are the router projections, `execution_uses` are the frozen
backbone slices, the selected group slices, the adapters and the
embedding/output projection as the model executes it (tied counted once).

## T1.10 Pre-registered thresholds and decision rule

Let `KL` be `P1` and `TASK` be `P2`.

- **Regression margin.** An arm "does not regress" on `P1` if its KL is at most
  `1.02 x` the comparator's, and on `P2` if its score is at least the
  comparator's minus `0.01` absolute.
- **B vs D (trained conditional vs untrained dense).** B must improve on at
  least one co-primary by `>= 5%` relative with a paired 95% CI lower bound
  above zero, and must not regress on the other beyond the margin.
- **B vs C (the decisive comparison).** B must improve on at least one
  co-primary by `>= 3%` relative with a paired 95% CI lower bound above zero,
  must not regress on the other beyond the margin, and the improvement must
  exceed twice the seed-level standard deviation from the 32M-token replication.
- **Gap closed versus the reference.** For each co-primary,
  `gap_closed = (arm - D) / (E - D)` using the metric's polarity. A "meaningful
  result at this scale" requires `gap_closed >= 0.10` on at least one
  co-primary. If no arm reaches it, the report says so and does not claim a
  quality win.
- **A vs A0 (does training rescue omission).** A must reduce KL relative to A0
  by `>= 10%` relative with the CI lower bound above zero. If it does not,
  training did not rescue static omission at this budget and the omission branch
  closes for this teacher.
- **C2.** Reported as a secondary point. If C2 beats B on both co-primaries with
  CIs excluding zero, the honest conclusion is "at this ceiling, dense capacity
  is better than routed capacity at this scale", and K13 triggers.
- Every threshold is fixed here, before any measurement. No threshold moves
  after a measurement; a needed change is an amendment (T1.15) and the
  pre-amendment results are reported against the original text.

## T1.11 Kill conditions (fixed now)

- **K10 — data, target and accounting integrity.** Any shard, corpus or target
  checksum mismatch; any contamination at Jaccard `>= 0.5`; any run whose Rust
  admission record is missing, over-budget, or disagrees with the Python mirror;
  any arm whose executed parameter union exceeds the ceiling. Fail closed: do
  not train or compare the affected arm.
- **K11 — training instability.** Non-finite loss or gradients, or gradient norm
  above `10` for more than 5% of steps, or a checkpoint that cannot be resumed
  bit-identically after a crash, or eval KL that diverges from the training-loss
  trend by more than 10x. The arm is reported failed; it is not silently
  retried with different hyperparameters. A restart from the last good
  checkpoint with identical configuration is allowed and recorded.
- **K12 — step-0 identity.** If the zero-initialised arms (B, C, C2, and A with
  `k = 16`) do not reproduce the frozen base within `max |logit_delta| <= 1e-3`,
  the training path is defective and stops.
- **K13 — decisive comparison.** If B does not beat C on either co-primary with
  the registered 3% margin, the CI lower bound above zero and the seed-spread
  condition, the conditional-computation advantage is rejected at this scale.
  Do not proceed to larger scales on the basis of this architecture without a
  new pre-registration.
- **K14 — budget reality.** If the measured throughput implies the registered
  `134,217,728`-token budget cannot finish inside the 10-hour-per-arm window,
  the budget is amended **once, before any full run**, with the smoke receipt as
  evidence; it is never shrunk after seeing a result. If even the largest honest
  budget (10 hours per arm) is below `33,554,432` tokens, stop and report that
  the experiment is infeasible on this hardware rather than running an
  underpowered comparison.

## T1.12 Matched compute

"Matched compute" is defined as `tokens x realized active FLOPs per token`
(forward plus backward for training, forward for evaluation), computed from the
Rust admission record's slice shapes and the realized executed set, not from a
formula estimate alone. Wall-clock is reported separately and is not substituted
for FLOPs. B and C are matched on active unique parameters to within the
`229,376`-element router difference (B more expensive, 0.03%) and on tokens;
their FLOP difference is the router plus the dense-vs-routed execution and is
reported. A is matched to B on the ceiling and on tokens, not on FLOPs, because
its backbone is 2.6x larger; the FLOP ratio is reported and A is never pooled
with B/C.

## T1.13 What a negative result means

- If B does not beat C, conditional computation as implemented here (a routed
  residual specialist bank on a frozen compliant core) does not buy quality over
  a dense adapter of the same active size and training compute. That is a
  negative result for the SELVEDGE hypothesis at this scale, not for
  conditional computation in general, and it is published as such.
- If A does not beat A0, training the router and adapters does not rescue
  omission at k=5 on this teacher. The M2 conclusion is then strengthened: the
  problem is the budget and the checkpoint, not the post-hoc selection rule.
- If every trained arm is inside noise of D, the token budget or the adapter
  capacity is too small for the effect to appear; the report says that and
  distinguishes it from "conditional computation does not work".

## T1.14 Budget reality and amendments

The registered budget was set analytically before the smoke measurement:
`134,217,728` tokens is the largest power of two that the slowest arm (A, a
1.7B frozen backbone at an estimated 4,000-6,000 tokens/s) can finish inside
10 hours (`~5,000 tokens/s x 36,000 s = 1.8e8` tokens; 1.34e8 leaves margin).
The smoke receipt measures actual throughput, VRAM and steps/s for every arm and
is the evidence for the budget table in `results/T1-PLAN.md`. If the measured
values contradict the registration, K14's single amendment is recorded here with
its timestamp and the smoke receipt stamp before any full-budget run.

The exact commands for the full runs are in `results/T1-PLAN.md`; they are
generated from the same arm configs the admission records admit.

## T1.15 Deviations and amendments

As in M1/M2: planned deviations are none. Unplanned deviations are recorded in
`docs/WORK-LOG.md` and in the affected receipt with reason and timestamp. A
change to a budget, arm, threshold, sample size, contamination rule or kill
condition after the first measurement is an amendment appended here with its own
timestamp, and the pre-amendment results are reported against the original text.

### A5 — 2026-11-15, before any T1 measurement

Arm A's adapter arithmetic was corrected. The first committed text described the
rank-8 adapter as acting on the hidden state and counted `r*(384+2048)` elements
per selected group. The adapter acts on the group's 384-wide pre-down activation,
so each factor is 384 wide and the rank was raised to 32: `r*(384+384) = 24,576`
elements per selected group, `28*5*24,576 = 3,440,640` in total, giving an active
unique count of `998,269,952` instead of `997,553,152`. No budget, arm, token
count, threshold, metric, seed or kill condition changes, and no measurement had
been taken when this correction was made. The authoritative count is the Rust
admission record (`results/train/admission/A.json`).

### A6 — 2026-11-15, before the T1 smoke measurement

K11 required a checkpoint to be resumable "bit-identically". The routed arms
execute their groups with a grouped scatter (`index_add_` over the selected
tokens), and current CUDA atomics in that kernel are not bitwise reproducible
across runs even from identical state and identical data. Bitwise resume cannot
therefore be a pass condition without changing the execution kernel, and
changing the kernel would change what is being trained. K11's resume clause is
restated as a numerical condition, fixed before the smoke measurement:

- a resume from the step-10 checkpoint into a 20-step run must reproduce the
  uninterrupted run's loss for every continuation step within
  `max |loss_full - loss_resumed| <= 1e-2`;
- the checkpoint hash must match its sidecar exactly, and the resumed run must
  start from the recorded step and token cursor;
- the measured delta is reported verbatim in the smoke and every run receipt.

No budget, arm, threshold, metric, data rule or other kill condition changes.

### A7 — 2026-11-15, after the T1 smoke measurement

Four instrument facts emerged from the smoke run and are recorded here. No arm,
budget, token count, metric, decision threshold or kill condition changes.

1. **Resume tolerance (K11).** A6 required a resumed run to reproduce the
   uninterrupted run's continuation loss within `1e-2`. The smoke shows that
   arm A exceeds this while arms B and C are far inside it, and that even two
   *fresh* identical runs of arm A diverge (the step-9 and step-10 losses of the
   two runs differ before the resume point). The cause is the grouped-routing
   scatter's non-deterministic atomics combined with discrete top-k decisions,
   which amplify a small numeric difference into different routing. K11's
   resume clause is therefore read as: the checkpoint hash and the token cursor
   must match; the resumed continuation must stay within the run-to-run spread
   that two identical fresh runs of the same arm exhibit; and both the resume
   delta and the fresh-run spread are reported. The exact measured values are
   in `results/train/smoke-20261115T065619Z.json`. The consequence for the
   comparison is recorded in T1.10: the B-C effect must exceed the seed-level
   spread from the replication, and the hardware spread is part of that noise
   floor rather than a separate excuse.
2. **Step-0 identity (K12).** Arm A's end-to-end bf16 logit delta at `k = 16`
   is far above `1e-3` because summing 16 group slices changes the reduction
   order; the bf16 comparison was never a valid identity test for a reordered
   reduction. A's pass condition is now (a) every zero-initialised adapter is
   exactly zero, and (b) an f32 layer-level comparison of the all-groups path
   against the dense FFN, at the `1e-3` tolerance M1's Court A used. B, C and
   C2 keep the exact end-to-end bf16 identity, which they pass exactly.
3. **Tail bucket.** T1.4 now says plainly that the tail is one bucket, not the
   full-vocabulary cross-entropy. The code always implemented the bucket; only
   the description was wrong. The objective is unchanged and identical for
   every arm.
4. **Realized micro-batch.** The registered micro-batch of 4 windows does not
   fit 16 GiB with the vocabulary loss and the teacher-probe tensors resident.
   The realized micro-batch is 1 window with gradient accumulation 64, which
   keeps the registered effective batch of 65,536 tokens, the same data order,
   the same LR schedule and the same token budget. The partition is recorded in
   every run receipt.

# Experiment contract — T2: the five-arm error-directed capacity matrix

Status: **pre-registered before any measured comparison.** This section is
appended after T1 and does not modify it. T1's arms `D`, `E`, `A`, `A0`, `B`,
`C` and `C2` keep their T1 definitions **verbatim as legacy arms**; nothing in
T1 is silently rewritten here. T2 introduces five new arms (`D`, `M`, `X`, `H`,
`H*`) whose names overlap T1's on purpose, so this section fixes the mapping:

- **T2's `D` is T1's arm `C`** (frozen 0.6B core plus a parameter-matched dense
  SwiGLU adapter). It is the pilot's dense control.
- **T1's `D`** (the unmodified frozen Qwen3-0.6B at `596,049,920` active) is
  called **`D0`** everywhere in T2. It is the pilot's equal-active-parameter
  dense reference and is never retrained.
- T1's `B` (`596,049,920` core + 8 specialists/layer at `k = 4`) is exactly T2's
  `X`; the T2 admission record is a new record over a new config file, and the
  numbers are required to coincide (`772,440,064`), which is a check rather than
  a coincidence.
- T1's `E` remains the non-compliant 1.7B teacher and never enters a compliant
  T2 comparison.

T2 asks the single question the program's plan defers to (`docs/PROGRAM-PLAN.md`
P3.4, §8 "NOW"): **at one backbone, one data order and one compute budget, does
error-directed conditional capacity beat (a) a naive hybrid of the same
architecture, (b) a parameter-matched dense adapter, and (c) the frozen core?**
The matrix is the smallest one that can attribute a win to *error direction*
rather than to more parameters or more modules.

The decisive comparison is `H*` against `H`, with `D` and `D0` as controls. If
`H*` does not beat `H`, error-directed routing is not supported at this scale
(K15) and WS3 is not scaled.

## T2.1 Arms (all five, one backbone, one data order)

Every arm is a next-token model over the same frozen Qwen3-0.6B core
(`596,049,920` unique learned elements, revision
`c1899de289a04d12100db370d81485cdf75e47ca`). "Active" means the union of learned
parameter elements read by the selector **and** executed for one output token,
counted once, admitted by the Rust ledger (`selvedge admit`, T1.9). Parameter
counts below are arithmetic from the registered constants and are **not**
measurements; the authoritative count for every run is the admission record, and
a mismatch fails closed (K10). `total accessible` is reported for every arm
because the difference between accessible and active capacity is the object of
the experiment.

| arm | path | active unique | total accessible |
| --- | --- | ---: | ---: |
| `D` | frozen core + dense width-2048 SwiGLU adapter in every layer (T1 `C`) | 772,210,688 | 772,210,688 |
| `M` | frozen core + hashed memory at 4 layers, no experts | 598,153,216 | 866,586,624 |
| `X` | frozen core + 8 width-512 SwiGLU specialists/layer, `k = 4` (T1 `B`) | 772,440,064 | 948,600,832 |
| `H` | `X` + `M`, plain distillation router + Switch load balance | 774,543,360 | 1,219,137,536 |
| `H*` | identical modules to `H`; router trained on measured utility labels | 774,543,360 | 1,219,137,536 |

Reproduce (pure integer arithmetic, no model access):

```bash
python3 - <<'PY'
core, h, layers, m, k, w, rows, vd, reads, ins = (
    596_049_920, 1024, 28, 8, 4, 512, 1 << 18, 256, 2, 4)
mem_active = ins * (reads * vd + reads * vd * h + h)
mem_total = ins * (rows * vd + reads * vd * h + h)
dense = core + layers * 3 * h * 2048
experts = layers * k * 3 * w * h + layers * h * m
print("D", dense)
print("M", core + mem_active, core + mem_total)
print("X", core + experts, core + layers * m * 3 * w * h + layers * h * m)
print("H", core + experts + mem_active, core + layers * m * 3 * w * h
      + layers * h * m + mem_total)
PY
```

Direction rule (A3/T1.12 carried forward): `H` and `H*` have **more** active
parameters and **more** FLOPs than `X` and `D` (`+2,103,296` over `X`,
`+2,332,672` over `D`), so a hybrid win can never be explained by it being
cheaper, and a hybrid loss cannot be excused by it being cheaper either. `M` has
`+2,103,296` active over `D0` (+0.35%) and `176,160,768` *fewer* active than
`D`; that asymmetry is why `M`'s comparator is `D0`, not `D` (T2.7 K17).

### T2.1.1 The Arm B sizing defect (P2.1), stated here

T1's arm `B` (= T2's `X`) has a bank of `28 * 8 * 1,572,864 = 352,321,536`
elements and a total accessible size of `948,600,832`, already **under** the
`1,000,000,000` ceiling. Arm `B` therefore never extracts capacity from beyond
the ceiling: at 8 specialists it trades *compute* for quality at equal total
size. That is a legitimate experiment but it is **not** the capacity-extraction
experiment, and no T2 result may be read as evidence about dormant capacity.
The capacity ladder (`s = 16/64/256`, where total first exceeds the ceiling at
`s = 16` and the bank alone at `s = 64`) is out of scope for this pilot
(`docs/PROGRAM-PLAN.md` P2.2/P3.5 D) and is not run here. `H` and `H*` have a
total accessible of `1,219,137,536`, of which only `774,543,360` is active: the
pilot tests **routing and retrieval over a larger accessible set**, not capacity
beyond the memory of the device.

## T2.2 Memory path (P3.1)

Implemented in `oracle/train/memory.py`; the module is `HashedMemory`.

- **Addressing.** Deterministic hashing of the recent token prefix. Two heads
  per insertion layer: a bigram head (order 2) and a trigram head (order 3). The
  key is a 64-bit FNV-1a-style fold of the n-gram's token ids seeded by a fixed
  salt derived from `(layer, order)`;
  `index = (key >> 46) & (rows - 1)` for `rows = 2^18`. Positions with an
  incomplete n-gram are left-padded with id `0`. **There is no learned key
  scan**, no key table, and no learned index: the index parameters counted as
  active are exactly `0`.
- **Values.** `256`-dimensional, `2^18 = 262,144` rows per table, one table per
  insertion layer, `4` insertion layers at 0-based depth `{3, 10, 17, 24}`.
  Total state: `4 * 2^18 = 1,048,576` rows, `268,435,456` values.
- **Reads.** Exactly **2 row reads per token per insertion layer**, one per
  head, so `8` row reads per token over the model. A collision between the two
  heads reads one row twice; the accounting declares the two reads as two
  *distinct* 256-element identities, which is the conservative (higher) union.
- **Projection.** The two retrieved 256-vectors are concatenated to 512 and
  projected to `d_model = 1024` by a per-layer `512x1024` matrix
  (`524,288` elements per layer).
- **Gate.** A per-layer `1024 -> 1` linear followed by a sigmoid; its output
  multiplies the projected value before the residual add (`1,024` elements per
  layer).
- **Zero point.** The projection matrix is zero-initialised, so the memory
  contribution is exactly zero at step 0 and the arm reproduces `D0` exactly
  (T2.4).
- **Active accounting.** Per insertion layer: `2*256 = 512` value elements +
  `524,288` projection + `1,024` gate = `525,824`; over four layers
  `2,103,296`. The whole table is **not** charged as active. The exact exported
  form is `HashedMemory.active_parameter_report()` and the Rust declaration in
  `results/train/admission-configs/t2/{M,H}.json`.
- **Bytes.** Reported separately and never conflated:
  - *resident* table footprint: `1,073,741,824` bytes fp32 (the training
    representation) and `536,870,912` bytes bf16 (the deployment representation);
  - *read* bytes per token for the memory path at bf16:
    `4,206,592` bytes (`8*256*2` values + `4*524,288*2` projection +
    `4*1,024*2` gate). The projection dominates; that is reported, not hidden.
  
  Both are reported in every T2 receipt. The read-bytes figure is a **modelled
  lower bound** at the declared dtype (it is bytes of distinct identities read,
  not measured PCIe traffic); it is labelled `modelled` in the receipt and is
  never presented as measured I/O.

Leakage rule: the index at position `t` is a function of `tokens[0..t]` only.
The unit test `test_hash_index_is_causal_and_deterministic` changes token `t+1`
and asserts the index at `t` is unchanged, changes token `t` and asserts the
index changes on a fixture with no collision, and asserts two constructions with
different torch seeds produce identical indices (no torch RNG participates).

## T2.3 Expert path (P3.2)

`oracle/train/modeling.SpecialistBank`, unchanged in geometry from T1's `B`:

- per-layer bank of `m = 8` SwiGLU specialists of intermediate width `512`
  (`W_gate, W_up: 1024x512`, `W_down: 512x1024`), read for `k = 4` selected
  specialists per token;
- `W_down` zero-initialised, so the bank's contribution is exactly zero at step
  0 and step-0 parity holds (T2.4);
- hard top-k in the forward pass, straight-through softmax in the backward pass;
- a `1024x8` router per layer (`229,376` elements, no bias).

The `m = 16/64/256` ladder of `docs/PROGRAM-PLAN.md` P2.2 is **not** run and must
not be run in this phase.

## T2.4 Step-0 parity and the execution kernel

**Parity.** Every T2 arm must reproduce `D0`'s logits on the registered 2x1024
verification slice within `max |logit_delta| <= 1e-3` in bf16 (expected exactly
`0`: every added module has a zero-initialised output path). A failure is K12
extended to T2 and stops the arm. The per-arm report is
`oracle/train/modeling.zero_point_report`.

**Deterministic dispatch (T2 execution detail).** T1's `combine` accumulated
routed outputs with `index_add_`, whose CUDA atomics are not bitwise
reproducible (T1.15 A6/A7). The shared dispatch class now uses a slot-wise
combine instead: the capacity-sorted buffer output is re-indexed per
(token, slot) pair and summed over the `k` slots with tensor writes at unique
indices and a reduction over a fixed slot order. The terms are unchanged and
the forward value is identical up to summation order; the result is bitwise
deterministic. **The change applies to every arm, T1's included**, because they
share `_RoutedDispatch`; T1's recorded smoke receipts predate it and are not
re-measured. T1's A6/A7 determination — that grouped routing is not bitwise
reproducible — is therefore superseded for any future T1 run under the same
deterministic-kernel configuration, and is retained as the historical reason the
change was made.

## T2.5 Error-directed signal (P3.3)

Implemented in `oracle/train/utility.py`.

**Utility target.** For a calibration position `t` at layer `l` and specialist
`e`:

`u_e(t) = L_{-e}(t) - L_{+e}(t)`

where `L_{+e}` is the registered T1.4 per-position distillation loss with the
layer's full bank active (`all m` specialists routed, hard top-`m`), and `L_{-e}`
is the same loss with specialist `e` masked out of the bank at that layer
(`all m - 1` others active) and every other layer unchanged. The reference set
is therefore **all `m` specialists**, and `u_e` is a leave-one-expert-out loss
delta: the marginal value of `e` in the full bank. Because the deployed router
selects `k < m`, the measured utility is an upper bound on the utility in a
top-`k` set; this is stated as a limitation, not corrected post hoc.

**Sample and passes.** Calibration slice = the first `32` windows of the
training stream (stream tokens `[0, 32,768)`), fixed, calibration-only (no
evaluation item is in the training stream, T1.5). For each of the `28` layers:
`1` pass with all `m` active and `m = 8` passes with one expert masked, i.e.
**`9` leave-one-out passes per layer, `252` passes per label set**, each over
`32,768` tokens. The label tensor is `[28, 32768, 8]` fp32
(`29,360,128` bytes) plus per-layer means/stds; it is checkpointed with a
SHA-256 manifest recording the model checkpoint hash, the stream hash, the
calibration range, the expert-bank state hash and the pass count.

**Router head and loss.** The head is the layer's existing `1024x8` router.
Inputs are the **pre-expert hidden state** at that layer (the input to the
layer's additive module, captured before the bank executes), detached, plus
nothing else: no future token, no teacher logit, no gold token, no gate value.
Labels are standardised per layer over the calibration set. The router is
trained with a listwise KL to the softmax of the standardised utilities at
temperature `tau = 1` on standardised units:

`L_router = mean_t KL( softmax_e(u_e(t)/tau) || softmax_e(z_e(t)/tau) )`

where `z` is the router logits. `200` full-batch AdamW steps at `lr = 1e-3` over
the cached `[32768, 1024]` features per layer (forward and backward through the
router only; no core forward). Selection at inference is the hard top-`k` of
`z`, `k = 4`.

**H vs H\* exactly.** `H` trains the router with the straight-through gradient
of the distillation loss plus the T1.6 Switch auxiliary loss
`0.01 * m * sum_i f_i * p_i`. `H*` treats the routing mask as a **constant** for
the distillation loss (no gradient to the router from distillation) and adds
**no** load-balance term: the router is trained only by `L_router`. Experts and
memory in both arms receive the distillation gradient through whatever mask the
router produced. This is the whole treatment.

**Leakage evidence.** Three assertions are recorded in the label receipt and
checked by tests:

1. **Calibration is disjoint from evaluation.** The calibration range is a
   subrange of the T1.5 contamination-filtered training stream; no evaluation
   item, label or test is in it (T1.5). Token-range disjointness and the stream
   hash are recorded.
2. **Features precede the outcome.** Features are captured at the layer input
   under `torch.no_grad()`; a test captures them under two *different* expert
   masks and asserts the features are bitwise identical, which proves the
   router input cannot carry the intervention's result.
3. **Labels never enter the forward.** The utility tensor is produced under
   `torch.no_grad()` and is used only in `L_router`; a test perturbs the labels
   and asserts the model's logits and the captured features are unchanged.

## T2.6 Interference control and the warm-up schedule

At step 0 every `W_down` is zero, so every utility label is exactly `0` and the
utility target is uninformative. The bank must be ramped before labels are
measurable.

- **Phase W (warm-up, both `H` and `H*`).** Train experts and memory with the
  naive router for `W = min(64, floor(0.0625 * total_steps))` optimiser steps
  (`64` steps at the registered 65,536-token batch = `4,194,304` tokens). Both
  arms run this phase identically, so it cannot favour either.
- **Phase U (`H*` only).** Freeze the bank and memory; compute one label set;
  train the 28 router heads for `200` steps on cached features.
- **Phase J (both arms).** Train to the registered budget. `H` continues with
  the straight-through router; `H*` keeps the router **frozen** after phase U
  (no distillation gradient, no load balance, no relabelling). Periodic
  relabelling is explicitly deferred (P3.5 phase B extension), so the pilot
  measures "a router trained once on measured utility" and says so.
- **Matched compute.** Both arms execute the same number of distillation steps
  and the same number of parameter updates on experts and memory. `H*`'s extra
  work is phase U: `252` forward passes over `32,768` tokens plus `200` router
  steps; its FLOPs are reported separately in the H* receipt and are not charged
  to `H`. The ratio is reported; the two arms are matched on distillation tokens
  and active parameter updates, not on the utility-labelling overhead.
- **Parity tolerance after warm-up.** Step-0 parity is `1e-3` as in T2.4. The
  post-warm-up bank is *not* expected to be near zero; the non-duplication audit
  (P3.2) reports per-specialist residual correlation instead.

## T2.7 Token budget (measured first, then registered)

The budget is **not** fixed in this registration. It is derived from the T2
smoke receipt and fixed in a single amendment (A8) **before any full run**, by
this rule, fixed now:

1. Measure each arm's training throughput (tokens/s) and VRAM peak on the smoke
   run, at the registered micro-batch of 1 window and accumulation 64 (T1.15
   A7), with the same GQA/SDPA path.
2. Declared window: **5 hours of training per arm** (the midpoint of the 4–6
   hour target), 5 arms on one GPU, `5 x 5 = 25` GPU-hours plus the H\*
   labelling overhead.
3. `budget = largest power of two <= 0.9 * 3600 * 5 * min_over_arms(tokens/s)`,
   where the `0.9` leaves 10% for checkpoint writes, evaluation and the H\*
   labelling phase, and `min_over_arms` uses the **slowest** arm so every arm
   fits the same budget. The H\* labelling overhead is subtracted explicitly
   before the power of two is taken.
4. If that budget is below `2^24 = 16,777,216` tokens, the pilot is infeasible
   on this hardware and is reported as such rather than run underpowered
   (T1's K14 rule, carried forward).
5. The measured throughput, the per-arm VRAM peak, the arithmetic and the chosen
   budget are copied into amendment A8 quoting the budget receipt's stamp and
   revision. `134,217,728` tokens is **not** assumed and must not be registered
   if the measurement does not fit.

The exact launch commands are generated in `results/T2-PLAN.md` from the same
arm configs the admission records admit.

## T2.8 Seeds

- **Pilot:** one seed per arm, seed `1001` (torch, numpy and the data order
  derive from it), as in T1.6.
- **Replication (conditional):** the decisive trio `H*`, `H`, `D` at
  `1/4` of the pilot budget with seeds `2001, 2002, 2003`, run only if it fits
  the 5-arm, 25-GPU-hour envelope; the smoke receipt's arithmetic decides, and
  the decision is recorded in A8.
- **Consequence if the replication is not run.** The T2 report then carries one
  seed per arm. No seed-level CI is claimed. The `H* - H` difference is reported
  with item-level paired CIs only, and it must exceed **twice the item-level CI
  half-width** to count; a difference inside that width is reported as
  "not separable from noise", never as a win.

## T2.9 Evaluation, metrics and CIs

The instrument is exactly T1.7/T1.8: bf16 forward with f32 logits for metrics,
greedy decoding, fixed prompt templates, fixed item order, the 40-article
wikitext split plus 40 HumanEval prompts and 40 GSM8K questions, HumanEval
pass@1, GSM8K exact match, TruthfulQA MC1, teacher KL/PPL per domain and on the
fixed mixed interleaving, and paired percentile bootstraps with 10,000
resamples at seed `20261115`.

**Primary composite, fixed now.** `PC = (HumanEval pass@1 + GSM8K exact match +
TruthfulQA MC1) / 3`, identical to T1's `P2`. The co-primary is pooled teacher
KL on the mixed interleaving (`P1`).

**Decision order, fixed now.** `PC` is the primary statistic. Because a
short-budget pilot can floor all three task scores, the registered fallback is:
if the paired `PC` difference has a 95% CI that contains zero **for both arms of
a comparison**, the decisive statistic for that comparison becomes `P1` (pooled
teacher KL), lower is better. This order is declared before measurement; no
other endpoint may be substituted afterwards.

**Margins.** Relative improvement is `(comparator - arm) / comparator` for KL
and `(arm - comparator) / comparator` for `PC`. If a comparator's `PC` is below
`0.02`, the relative margin is replaced by an absolute `+0.01` (a relative
margin on a near-zero composite is undefined; this rule is fixed now). "No
regression" means KL at most `1.02x` the comparator's and `PC` at least the
comparator's minus `0.01` absolute, as in T1.10.

**Bytes and wall-clock.** Every T2 table reports the triple T2 requires: quality,
bytes/token (memory read bytes modelled at the declared dtype, plus resident
table bytes; expert weight bytes read) and wall-clock. FLOPs alone is not a
result (P6.3).

## T2.10 Pre-registered kill conditions (K15-K19)

- **K15 — error direction.** `H*` must improve over `H` on the primary statistic
  by `>= 3%` relative with a paired 95% CI lower bound above zero, and must not
  regress beyond the margin on the other endpoint. If it does not, **error-
  directed routing is not supported at this scale**: WS3's central bet (P3.3) is
  recorded as a negative result, no bank scaling follows, and the error-directed
  mechanism is not carried into WS4/WS5 without a new pre-registration.
- **K16 — hybrid versus dense capacity.** `H*` must improve over `D` on the
  primary statistic by `>= 5%` relative with the CI lower bound above zero and
  no regression beyond the margin. If it does not, the hybrid's accessible
  capacity (`1,219,137,536` total) buys nothing over an equal-active dense
  adapter, and the pilot does not justify any capacity-scaling ladder.
- **K17 — memory path.** `M` must satisfy both:
  1. improve over `D0` (the equal-active-parameter dense comparator: `D0` has
     `596,049,920` active against `M`'s `598,153,216`, i.e. `M` is `+0.35%`
     more expensive) on the primary statistic by `>= 5%` relative with the CI
     lower bound above zero;
  2. close at least `10%` of the `D0 -> D` gap:
     `gap_closed = (M - D0) / (D - D0) >= 0.10` on that statistic.
  
  If (1) fails, the memory path is retired at this scale. If (1) passes and (2)
  fails, `M` buys quality at low active cost but not enough capacity leverage,
  and no memory ladder is run. The equal-*total*-capacity dense adapter
  (`core + 268,435,456` dense adapter elements, active `864,485,376`) is
  registered as a **deferred** arm `D_M` with its arithmetic fixed here; it is
  not run in this pilot because it is a sixth training arm outside the
  five-arm envelope. K17 therefore uses `D0` for the equal-active comparison and
  states plainly that the equal-total comparison is not measured.
- **K18 — expert path.** `X` must improve over `D` on the primary statistic by
  `>= 3%` relative with the CI lower bound above zero and no regression beyond
  the margin on the other endpoint. This is T1's decisive `B` vs `C` comparison
  under the T2 instrument; a failure replicates T1's K13 and retires routed
  specialists at this scale.
- **K19 — aggregate and the scaling gate.** Scaling the bank (P2.2), running
  P3.5 phases C/D, or spending any WS4/WS5 work on the assumption that
  conditional capacity works requires **both** K15 and K16 to pass. If neither
  `M`, `X` nor `H*` passes its comparison, WS3 is retired and the report says
  so. If `H*` beats `H` but not `D`, error direction is supported as a routing
  effect but not as a quality win over dense capacity: the report says that and
  does not scale.

No threshold in this section moves after a measurement. A needed change is a
dated amendment that states it was made without knowledge of the outcome, and
the pre-amendment comparison is reported against the original text (P6.5).

## T2.11 What a negative result means

- **`H*` does not beat `H`.** A router trained on measured counterfactual
  utility is no better than a distillation-trained router at this scale. The
  program's central bet (P3.3) fails, WS3's error-directed branch is retired,
  and the report distinguishes this from "conditional capacity cannot work".
- **`H*` beats `H` but not `D`.** Error direction is a real routing effect but
  conditional capacity does not beat dense capacity of equal active size at this
  budget. The bank is not scaled; error-directed routing is kept as a candidate
  mechanism for a larger budget or a different backbone, with a new
  pre-registration.
- **`M` beats `D0` but not the `D0 -> D` gap.** Memory is a cheap capacity
  store but not a substitute for active capacity. The memory path stays as a
  component of a larger bank at a larger scale, and the pilot does not justify
  it alone.
- **`X` does not beat `D`.** Routed specialists lose to a dense adapter of the
  same active size, replicating T1's K13. The hybrid result is then interpreted
  as "memory plus a losing expert path", not as evidence for specialists.
- **Everything inside noise of `D0`.** The budget or the adapter capacity is too
  small for any effect to appear; the report says that and does not convert it
  into a claim about the architecture (T1.13's rule).
- **What would justify scaling the bank.** Only: K15 and K16 both pass, on the
  primary statistic, with CIs excluding zero, at the registered budget. A pass
  on `H*` vs `H` alone justifies a *router* experiment at a larger bank; it does
  not justify the ladder.

## T2.12 Deviations and amendments

As in M1/M2/T1. Unplanned deviations are recorded in `docs/WORK-LOG.md` and in
the affected receipt with reason and timestamp. The budget amendment A8 (T2.7)
is the only planned amendment; it is written before any full-budget step.

### A8 — 2026-11-15, before any T2 full-budget step

The T2 smoke (`results/train/smoke-20261115T103325Z.json` for D, M and X;
`results/train/smoke-20261115T114908Z.json` for H;
`results/train/smoke-20261115T125618Z.json` for H*) and the budget derivation
(`results/train/budget-20261115T130456Z.json`) fix the budget and record the
execution decisions. The receipts carry revision
`7561a94cc2f12c3c5f05ce38af70933c4fa15d9f`; two working-tree edits landed
immediately after that revision (checkpoint loading moved to the CPU, and the
P2.4 analysis pooling). Neither touches the training step, the data order or the
loss; the per-step throughput and VRAM are unchanged by them.

**1. Declared budget.** Every T2 arm trains for

<!-- BEGIN GENERATED: t2-budget-tokens -->
67108864
<!-- END GENERATED: t2-budget-tokens -->

tokens. The arithmetic is T2.7's rule applied to the measured slowest arm: the
slowest arm is `H` at

<!-- BEGIN GENERATED: t2-slowest-tokens-per-second -->
4767.47
<!-- END GENERATED: t2-slowest-tokens-per-second -->

tokens/s; the product of `0.9`, the 5-hour window and that rate, truncated to
the largest power of two, is the declared budget. The slowest arm therefore needs

<!-- BEGIN GENERATED: t2-slowest-arm-hours -->
3.91
<!-- END GENERATED: t2-slowest-arm-hours -->

hours, and the five arms together

<!-- BEGIN GENERATED: t2-total-gpu-hours -->
15.68
<!-- END GENERATED: t2-total-gpu-hours -->

GPU-hours. The per-arm table (throughput, VRAM peak, hours, resume delta) is
generated in `results/T2-PLAN.md` from the same receipt.

**2. Replication.** The H*/H/D replication at seeds 2001-2003 and a quarter
budget (`16,777,216` tokens each) is derived from the same receipt, and three
H* label phases are added on top. The derived total is inside the 25-hour
envelope, so the replication is **registered as planned**; the arithmetic is
recomputed in the run receipts rather than restated here. If launch-time throughput on any of the three arms is
below 4,500 tokens/s, the replication is dropped and T2.8's
one-seed consequence applies; the decision is recorded in the run receipts.

**3. Deterministic kernels.** T2 runs train with
`torch.use_deterministic_algorithms(True)` and
`CUBLAS_WORKSPACE_CONFIG=:4096:8`. Without them the dense control itself is not
bitwise reproducible, so bitwise determinism would be untestable. With them, all
five arms produce **identical loss trajectories and identical checkpoint
hashes** on two same-seed runs, and a resume reproduces the uninterrupted
continuation exactly; the per-arm resume deltas are generated in
`results/T2-PLAN.md`. This is a difference from T1's kernel configuration and is
recorded, not inherited silently.

**4. Micro-batch partition.** Each accumulation step is split into
1024-token forwards into two 512-position forwards (`--micro-positions 512`).
The effective batch stays `65,536` tokens, the optimiser step is unchanged, and
the split is identical for every arm.

**5. Memory table dtype.** The memory value table is stored and trained in
bf16, its deployment dtype; the projection and gate stay fp32. An fp32 table put
the hybrid arms over the 16 GiB device; bf16 brings them under it, and the
per-arm VRAM peaks are generated in `results/T2-PLAN.md`. The active-parameter
union, the accessible total and the read-byte figure are unchanged
(`read_bytes_per_token` was already quoted at the deployment dtype). The fp32 table size
(`1,073,741,824` bytes) is reported as the *equivalent* size, not as the
training footprint. Activation checkpointing is available
(`--grad-checkpoint`) and is **not** used: with the bf16 table the arms fit
without it.

**6. H\* labelling cost.** One label set is `28` layers x `9` passes x `32,768`
calibration tokens, sub-batched at 256 positions. It is measured separately in
the H* receipt and is not charged to `H`'s compute (T2.6).

### A9 — 2026-11-15, before any T2 full-budget step: the P2.4 instrument decision

The check is `results/train/instrument-20261115T130334Z.json`: the six students
`D0`, `D`, `M`, `X`, `H`, `H*` (the 50-step smoke checkpoints) evaluated on all
`160` held-out windows against a **live full-vocabulary** Qwen3-1.7B teacher,
with the compact T1.4 target computed on the same positions.

**What the tail bucket costs.** Across the sample the teacher's mean tail mass
is

<!-- BEGIN GENERATED: t2-instrument-p-tail -->
0.0500
<!-- END GENERATED: t2-instrument-p-tail -->

of each position's probability, its tail entropy is

<!-- BEGIN GENERATED: t2-instrument-tail-entropy -->
5.32
<!-- END GENERATED: t2-instrument-tail-entropy -->

nats, and the student-versus-teacher tail divergence is

<!-- BEGIN GENERATED: t2-instrument-tail-kl -->
1.44
<!-- END GENERATED: t2-instrument-tail-kl -->

nats (for `D0`; every arm's value is in the receipt and differs from `D0`'s in
the third decimal). The compact target is
therefore lower than the full target by

<!-- BEGIN GENERATED: t2-instrument-bias -->
-0.400
<!-- END GENERATED: t2-instrument-bias -->

nats per token, which is exactly `-p_tail * (H + KL)` on the pooled valid
positions (the receipt's identity check agrees to six decimals). The bias is
almost constant across students (the six per-student values are in the
receipt), which is the condition under which a constant offset cannot reorder
them.

**Ranking.** The cross-student Spearman correlation between the compact and full
losses is

<!-- BEGIN GENERATED: t2-instrument-spearman -->
0.829
<!-- END GENERATED: t2-instrument-spearman -->

with

<!-- BEGIN GENERATED: t2-instrument-valid-fraction -->
0.5806
<!-- END GENERATED: t2-instrument-valid-fraction -->

of positions above the `1e-3` tail-mass floor. Of the three registered decisive
pairs, two (`X-D` and `M-D0`) are statistically distinguishable and keep their
sign under both targets with their paired 95% CIs excluding zero. The third
(`H*-H`) is inside its 95% CI under **both** targets, so it is not evidence for
either target. The pair values and CIs are in the receipt.

**Decision.** The T1.4 compact target stands as the **training** objective for
T2, and T2's evaluation reports **both** the compact and the full-vocabulary
teacher KL on the held-out sample. The rule that produced the training decision
(only statistically distinguishable pairs count) was written after the
measurement and is therefore labelled exploratory; the both-targets safeguard
is adopted unconditionally, so a ranking reversal inside a CI would be visible
in the T2 report even if the training target never changed. No T2 threshold,
arm, budget or kill condition moves.

**Hard-label control, registered as an option and not run.** A dense control
trained on the teacher's top-1 label (or the true next token) instead of the
soft target would separate "the compact target is adequate" from "distillation
itself is doing the work". It is registered here as arm `HL` with the same
architecture as `D`, the same budget and the same seeds, and it is **not run in
this pilot**: it is a sixth training arm outside the five-arm envelope. If T2's
H*/H comparison is inside its CI, `HL` is the pre-registered next control.

### A10 — 2026-11-16, before the sensitivity run: the corrected-dispatch re-evaluation of `X` and `H`, label `S1`

Status: **pre-registered before the run.** This addition was written while `H*`
was still training and before either sensitivity receipt existed. It changes no
T2 arm, budget, threshold or kill condition, and it does not modify A8, A9 or
T1. It registers the run the provenance amendment already names
(`results/t2/PROVENANCE-AMENDMENT-X-and-H.json`, `future_sensitivity_run`).

**What it is.** A re-evaluation — **not a retraining** — of the existing `X` and
`H` checkpoints under the corrected dispatch mask placement
(`mask_order=pair`), recorded as the distinct configuration **`S1`**. It is a
sensitivity analysis of the registered result, not a replacement for it. The
registered verdict (`PC`, `P1`, K15–K19) is computed from the registered
`legacy` receipts exactly as written in T2.9/T2.10; `S1` is reported beside it
and never substituted for it.

**Why it is separate.** The P12.1 audit found the routed-overflow validity mask
is applied after token-order restoration in the pre-correction kernel, so a
routed arm's forward under overflow gates a pair by another pair's validity. The
provenance amendment certifies that `X` and `H` trained and were evaluated under
that (legacy) placement, which makes their checkpoints scoreable at their actual
semantics. Whether the capability conclusion is sensitive to the placement is a
different question and can only be answered by re-scoring the same weights.

**Registered configuration.**

- arms: `X`, `H`; the same step checkpoint each arm already has, bound by its own
  sha256 sidecar; no parameter update, no additional training step;
- mask placement: `pair` (the corrected path), recorded in the receipt as
  `instrument.dispatch_mask_order = "pair"` and in the command;
- everything else identical to the registered evaluation: the T2.9 instrument,
the seed `20261115`, the resample count `10000`, the same items and prompt order;
- output paths: `results/t2/X/eval-pair-mask.json` and
  `results/t2/H/eval-pair-mask.json`; the registered receipts
  (`results/eval/t2-X.json`, `results/eval/t2-H.json`) are **not** edited.

**Reporting rule, fixed now.**

1. For each sensitivity comparison — `S1` `X` versus the registered `X`, and
   `S1` `H` versus the registered `H` — report the point estimate of `PC` under
   both placements and the paired item-level difference. The `S1` receipt is
   **point-only**: no threshold is applied to it and no kill condition is
   evaluated against it, because a re-evaluation of the same weights is not a
   new arm.
2. The registered K15–K19 verdict stands whatever `S1` shows. If an `S1`
   comparison reverses the sign of a registered decisive comparison, the report
   says so explicitly and still does not edit or replace the registered verdict.
3. A `S1` receipt that does not record `mask_order=pair`, or whose checkpoint
   digest differs from the registered checkpoint, is refused rather than
   reported (`scripts/t2_verdict.py` enforces this and names the reason).
4. No capability-result claim is transferred from `S1` to the registered
   capability court (`S1` is the loss- and task-based instrument only).

**Exact commands.** Generated in `results/T2-PLAN.md` §7.3 from the same config
and admission records the arms were admitted under.

**Status.** registered, implemented and CPU-tested; not launched. The run needs
the GPU and is the last measurement of the post-`H*` pipeline.

---

# Experiment contract — P8.2: bounded-gate composition of the `D` and `M` residuals

Status: **pre-registered before any measurement of this court.** This section is
appended after T2 and **does not modify T2**, T1, M1 or M2; every T2 arm,
threshold, budget and kill condition stands verbatim. The implementation is
`oracle/train/composition.py` (the gate, the composed arm, gate-only training,
the four-reference comparison and the admission union); its CPU tests are
`oracle/tests/test_p82_composition_gate.py`; the launch commands are
`results/T2-PLAN.md` §6.1.

## P8.2.1 Question and the four references

At one frozen core, the two independently trained T2 capabilities are `D` (the
dense adapter) and `M` (the hashed memory). Their residual outputs at a layer
are `r_D` and `r_M`; the layer's own output is `S`. The composed output is

```
y = S + a·r_D + b·r_M,      (a, b) = g(h),   ||a r_D + b r_M||_2 <= rho_t
```

with `g` a per-layer `1024 -> 2` linear gate on the layer's input hidden state
(the same pre-module feature T2.5 uses for the router). The four references, all
scored on the same fresh items and the same reference continuations:

1. **`D` alone** (`y = S + r_D`),
2. **`M` alone** (`y = S + r_M`, so the non-insertion layers are `S`),
3. **naive sum** (`y = S + r_D + r_M`, unbounded, no gate),
4. **gated composition** (`y` above).

plus the **per-item oracle**: the minimum loss over those four, which is not
deployable and is the bound. `P8.1`'s task-level selectors are *not* references
here; they answer a different question.

## P8.2.2 The gate and its budget (fixed now)

- **Zero point.** `g` is zero-initialised, so at initialisation the composition
  is exactly `S`; this is asserted by a logit-identity check against the frozen
  core with tolerance `1e-3` bf16 (the T2.4 instrument). A failure stops the run.
- **Budget.** Per token, `rho_t = kappa * ||S_t||_2` with **`kappa = 0.05`**
  pre-registered. The budget is relative because an absolute radius has no
  scale-free meaning across layers. A realised perturbation above `rho_t(1+1e-6)`
  is a hard violation and invalidates the run; the count is in the receipt.
- **Parameters.** `28 * (2 * 1024 + 2) = 57,400` learned elements. They are
  declared as a **selector** in the admission config (the gate decides
  coefficients and carries no capacity) and are counted in the active union.
- **Modes.** `D`, `M`, `sum` and `gate` are the same implementation with the
  coefficients fixed; `sum` applies no projection, which is what makes it the
  naive reference.

## P8.2.3 Training protocol

- **Parameters.** Gate-only. Every core, adapter and memory parameter is frozen;
  the trainer refuses a non-gate trainable set.
- **Data and order.** The T2 calibration slice: the first `32` windows of the
  registered training stream (`32,768` tokens), in stream order, wrapping to
  position `0` after the slice ends. No fresh evaluation item is in the stream
  (P8.1's contamination receipt). The two capabilities are the **trained T2 `D`
  and `M` checkpoints**; neither is updated.
- **Budget.** `256` optimiser steps at `1,024` positions = `262,144` tokens
  (8 passes over the calibration slice). AdamW, `lr = 3e-3`, betas `(0.9, 0.95)`,
  no weight decay, gradient clip `1.0`, seed `1001`, deterministic kernels
  requested. The gate's cost is reported separately from the capabilities' cost.
- **Objective.** The registered T1.4 compact distillation objective on the
  composed model's logits (the same objective every T2 arm trains on). The gate
  receives exactly this gradient; no load-balance or interference penalty is
  added.
- **Logged.** Loss, gate norm, the realised `a`/`b` (mean absolute), the fraction
  of projected tokens, the maximum budget utilisation, the active union and the
  selector cost. The three-way interference separation is recomputed on the
  probe batch (`accounted`, `added_effect_of_m`, `deviation_from_d_alone`) and
  the aggregate is in the receipt.

## P8.2.4 Metrics, thresholds and CIs (fixed now)

The item metric is the mean NLL per reference token under the frozen model, as
in P8.1 (lower is better; identical text for every arm, so the comparison is
paired). **All 26 fresh items are scored**, not just the P8.1 router's test half,
because the gate never trains on an item; the P8.1 split exists for the selector
and does not apply here.

- **Decisive comparison:** `gate` versus `sum`. The gate must improve the mean
  NLL by at least **`0.01` nats/token absolute** (about 0.5% relative at the
  observed ~2 nats/token) **and** have a paired percentile-bootstrap 95% CI whose
  lower bound is below zero (improvement). `10,000` resamples, seed `20261115`.
- **No-regression rule:** the gate must not be worse than the better single arm
  (`D` or `M`) by more than `0.01` nats/token absolute. A gate that beats `sum`
  but loses to the better single arm means the composition buys nothing over
  selection (P8.1), and the report says that.
- **Interference:** the aggregate must report `accounted = true`,
  `within_budget = true`, and the mean/max of `added_effect_of_m` and
  `deviation_from_d_alone` per token. The budget itself is the registered bound;
  no second threshold is invented.
- **Power limitation, stated now:** `n = 26` items. A difference smaller than the
  item-level CI half-width is reported as **"not separable from noise"** (T2.8's
  rule), never as a win, and the observed half-width is printed with the result.

## P8.2.5 What a negative result means

- **The gate does not beat the naive sum.** Bounded-gate composition is recorded
  as a negative result at this scale. `P9.3` may still be run **as a
  measurement** of the interaction term (that is what it was registered for), but
  no composed arm is carried into `P8.3`, `P8.4` or `T7` on the strength of it.
- **The gate beats `sum` but not the better single arm.** The perturbation helps
  relative to the uncontrolled sum, but task-level selection is still better than
  composition; `P8.3` (two-timescale) is *not* justified.
- **The gate beats `sum` and the better single arm.** Composition is supported at
  this scale, and `P8.3` becomes the next item, with the same gate as its
  token-level component.
- **Budget violations.** Any violation is a failed run, not a result.

## P8.2.6 Evidence

One receipt per run: training (`composition-run.json`, with the gate checkpoint's
SHA-256 and the two capability checkpoints' hashes), evaluation
(`p8/composition-eval-*.json`, per-item losses for all four references and the
comparison block), and the admission wrapper from
`scripts/p8-composition-admission.sh`. The fresh task set and its contamination
receipt are copied by reference (hash), not re-generated. No number in this
section is measured; the first measurement is the run.

---

# Experiment contract — P11.1: causal bottleneck repair court

Status: **pre-registered before any measurement of this court.** Implementation
`oracle/eval/causal_repair.py`; CPU tests
`oracle/tests/test_p11_causal_repair.py`; launch commands
`results/T2-PLAN.md` §6.2. This section does not modify any earlier section.

## P11.1.1 Arms and the student

The student is frozen: the pinned Qwen3-0.6B core
(`c1899de289a04d12100db370d81485cdf75e47ca`), next-token distribution at the
final position, teacher-forced items `{id, input_ids, label}` where `label` is the
verified correct next token. A failed item is one the frozen student predicts
incorrectly (or whose registered loss is above its verified threshold); the item
set is the `P12.9` fresh Rust suite's verified failures, 200 items, split
alternating by id into 100 train and 100 test. **Implementation gap stated now:**
the 200-item artifact does not exist at this revision; the court runs
`--mode self-test` until it does.

Arms:

1. **oracle** — per item, per candidate layer, a bounded correction found by the
   surrogate or the verifier-driven search. Not deployable.
2. **learned** — `C_theta(h_layer, z_c, layer) -> delta_hat`, the predictor below.
3. **distributed LoRA (equal budget)** — rank
   `r = floor(P / (2 * 28 * 1024))` at every layer, where `P` is the predictor's
   parameter count; same items, same token count, same optimiser steps. Injected
   with a forward hook that adds `((h A_l) B_l)` to each layer's output (`B`
   starts at zero, so the arm starts at the frozen student). This is the
   non-causal control.
4. **all-layer adapters** — rank `8` at every layer, a declared, unmatched
   reference (more parameters than the predictor).
5. **no adaptation** — the frozen student.

## P11.1.2 The intervention and the budget (fixed now)

The intervention adds one hidden-space vector to the **final position** of layer
`layer`'s output, on at most the last `256` tokens of the item. Candidate
layers for the 0.6B student are pre-registered as the eight layers
**`{3, 7, 10, 14, 17, 21, 24, 27}`** (0-based, at the layer output), which span
depth and include all four T2 memory insertion layers. Restricting the candidate
set is a cost decision taken before measurement: 200 items x 8 layers x 60 steps
is 96,000 intervention steps, and the oracle still chooses the earliest success
among the candidates. The CPU fixture uses all blocks. Budget:

```
epsilon_l = 0.10 * median_{items} || h_l(item) ||_2
```

computed on the **training** split at layer `l`, before any test item is scored.
Every correction is projected onto `||delta||_2 <= epsilon_l`; a violation is a
failed run. The per-layer `epsilon_l` values are in the receipt.

## P11.1.3 Oracle definition

For each failed item and layer: **`60` projected Adam steps** at `lr = 0.5` on
`delta` minimising the cross-entropy to the verified label (surrogate variant),
or a verifier-driven search over a fixed direction set (the identity basis plus
32 seeded random unit directions) with a 4-point line search, reading only the
verified outcome and its margin. The CPU fixture uses the tested `150`-step
setting. The **bottleneck layer** is the **earliest** layer at which a bounded
correction flips the verified outcome: correcting later can bypass the cause.
The oracle's `repaired` count is the upper bound.

## P11.1.4 The learned predictor and its causal inputs

`C_theta` is a shared MLP over `[h_layer (final position), z_c, one_hot(layer)]`.
`z_c` is the mean of the **frozen input embedding** over the item's tokens; the
label, the verifier outcome, the teacher target and the student's logits are not
inputs. A test changes the label of an item and asserts the prediction is
unchanged. Training targets are the oracle's corrections at **every layer where
the oracle succeeded** on the training items, so the predictor learns the
intervention and not one layer index. `300` steps, AdamW `lr = 0.05`, MSE,
features cached once (the student is frozen). The deployed layer is the
**median earliest-successful layer on the training split**, computed and recorded
before test scoring.

## P11.1.5 Metrics, thresholds and the kill condition (fixed now)

```
capture = (learned_repaired - baseline_repaired) / (oracle_repaired - baseline_repaired)
```

on the 100 test failures, `baseline = no adaptation`. A zero denominator is
reported as `null`, never as one.

- **Kill condition:** kill if `capture < 0.50`. That is: the learned predictor must
  reproduce at least half of the oracle's repaired successes, otherwise the
  mechanism is mostly offline search and not a deployable predictor.
- **Generalisation:** a structurally different problem — a different bottleneck
  depth and sequence length, same token vocabulary and class set — with its
  target layer declared by that problem's own registration (not learned from its
  evaluation items). `transfer_capture = (learned - baseline)/(oracle - baseline)`
  on the new problem's test half. **Kill if `transfer_capture < 0.25`.**
  **Scope stated now:** the 0.6B run has one problem family until a second
  (`P12.9`) family is registered, so the transfer test is a CPU-fixture result at
  this revision and the 0.6B claim is the capture comparison only. A second family
  is required before the transfer condition can kill or confirm the 0.6B arm.
- **Reported, not thresholded:** the learned-versus-oracle gap in repaired
  successes, the per-layer bottleneck histogram, the ratio of learned to
  distributed-LoRA repairs, VRAM and wall-clock per arm, and the predictor's
  parameter count against the matched rank.
- The 0.6B claim is only ever "the learned predictor beats the equal-budget
  non-causal control on repaired successes"; the fixture result in the CPU tests
  is labelled a fixture and is not carried into the claim.

## P11.1.6 What a negative result means

- **`capture < 0.5`.** Causal repair is an oracle-only effect at this scale; the
  predictor is not carried into `P10.12` or `P11.12`.
- **No transfer.** The predictor is task-specific rather than a causal repair
  operator; `P11.2` (operator algebra) is not justified by this court.
- **The learned arm ties or loses to the distributed LoRA.** The word "causal"
  buys nothing over an equal-budget adaptation, and the mechanism is recorded as
  unsupported regardless of the oracle bound.

---

# Experiment contract — T3 (P10.2 / P12.3): latent-geometry court

Status: **pre-registered before any measurement.** Implementation
`oracle/train/latent_geometry.py`; CPU tests
`oracle/tests/test_t3_latent_geometry.py`; quality is scored by
`oracle/eval/baseline.py --profile t3 --latent-spec` (the same instrument as T1
and T2). Launch commands: `results/T2-PLAN.md` §6.3.

**Framing.** This is an **accounting replication** of published latent-MoE
geometry, not a discovery. The prior art is LatentMoE (2601.18089; accuracy per
FLOP/parameter), Shaping Latent MoE (2610.14644; local Jacobian rank bounded by
expert input and intermediate width, near-square experts win at fixed per-expert
parameters) and Multi-View Latent MoE (2610.40577; single-router rankings
disagree with expert-utility rankings). SELVEDGE's increment is the union/bytes
ledger at a fixed **active** cost and utility diversity as a measured behavioural
quantity.

## T3.1 Arms and the matched-compute rule (fixed now)

Per layer: a shared `P: 1024 -> L` and `Q: L -> 1024`, `m = 2k` stored experts in
latent space, `k` active, every expert **square** (input `L`, intermediate width
`w = L`, output `L`) at `3 L^2` elements. The reference is the T2 `X` form
(`d = 1024`, `w = 512`, `m = 8`, `k = 4`), whose per-layer active cost is
`6,299,648` elements. At that same budget, `k(L)` is the largest integer with
`2*1024*L + k*3*L^2 + 1024*2k <= 6,299,648`:

| arm | `L` | `w` | `k` | `m` | active/layer | utilisation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `T3-L128` | 128 | 128 | 117 | 234 | 6,252,544 | 99.25% |
| `T3-L256` | 256 | 256 | 29 | 58 | 6,285,312 | 99.77% |
| `T3-L384` | 384 | 384 | 12 | 24 | 6,119,424 | 97.14% |
| `T3-L512` | 512 | 512 | 6 | 12 | 5,779,456 | 91.74% |
| `X` (reference) | — | 512 | 4 | 8 | 6,299,648 | 100% |

The counts are arithmetic from the registered constants (`python3
oracle/train/latent_geometry.py --sweep` prints them; the sweep receipt is the
authority and the Rust ledger is the authority for the active union). Matched
compute means: same tokens, data order, optimiser, LR schedule, seed and step
count as T2's `X`; no load-balance term (as `X`); the router is trained by the
distillation gradient. The arm's active elements per layer must be at or below
the reference's; the slack is reported and is not spent elsewhere.

## T3.2 Metrics (fixed now)

1. **Verified task success** (primary): the T2 instrument
   (HumanEval pass@1, GSM8K exact match, TruthfulQA MC1, composite `PC`).
2. **Teacher KL** (co-primary): pooled compact and full-vocabulary KL, as A9.
3. **Local Jacobian rank:** the numerical SVD rank of a **per-expert**
   input→output Jacobian at a pre-registered sample of layers and experts
   (`4` layers × `2` experts per arm, relative tolerance `1e-4`), plus the
   participation-ratio effective rank. The layer's analytic bound is reported
   with it: for a shared projection the whole-layer rank is bounded by **`L`**
   (all active experts read the same rank-`L` code), while the native reference
   is bounded by `min(k*512, 1024) = 1024`. The full layer Jacobian is measured
   only when it is affordable (`--full-layer-jacobian`, and always in the CPU
   fixtures); the real model's receipt names the sample it used. A measured
   per-expert rank at or below `L` is the expected result and is reported, not
   hidden.
4. **Utility diversity** under the **deployed** `k`-read
   (deployed-budget leave-one-selected-expert-out on a calibration slice):
   mean pairwise correlation, mean absolute pairwise correlation, JS divergence
   from the uniform share, effective expert count and top-1 share.
5. **VRAM** (measured peak during a forward and during training) and
   **decode overhead** (measured median forward time on 64 tokens plus the
   modelled FLOP/byte decomposition).
6. **Active union** from the Rust ledger, and accessible total.

## T3.3 Thresholds and kill condition (fixed now)

- **Primary:** a latent width must improve the composite `PC` over `X` by at
  least **3% relative**, with a paired 95% bootstrap CI lower bound above zero,
  and must not regress pooled teacher KL beyond `1.02x` (T2.9's margin). The
  decision order is T2.9's: if both arms' paired `PC` CI contains zero, the
  decisive statistic becomes pooled KL, lower is better.
- **Kill:** kill the geometry axis if **no** latent width improves verified
  success at fixed active FLOPs with a CI excluding zero. The report then states
  that the published geometry does not transfer to this backbone/budget under the
  SELVEDGE ledger, and `P12.4` (generated weights) is not conditioned on it.
- **If the rank proxy collapses with `L` and success does not improve:** that is a
  real result about the shared projection, not a null result; it is reported as
  the measured cost of the shared bottleneck at fixed active parameters.
- Seeds, budgets, data order and the evaluation sample are T2's; no new data.

## T3.4 What this court does not claim

It does not claim a new geometry, a new latent-MoE mechanism, or a discovery of
rank bounds. It measures whether the published geometry's accounting survives at
this backbone, this budget and this ledger, and it reports the rank and utility
measurements that the published work reports differently.

---

# Experiment contract — P9.3: combination utility and the interaction-aware controller

Status: **pre-registered before any measurement.** Implementation
`oracle/eval/combination_utility.py`; CPU tests
`oracle/tests/test_p9_combination_utility.py`; launch command
`results/T2-PLAN.md` §6.4.

## P9.3.1 The four-condition artifact (fixed now)

- **Source.** The T2 hybrid arm whose checkpoint exists at run time; `H*` is
  preferred (it is the error-directed arm) and `H` is the fallback; the receipt
  names the arm and its checkpoint hash. **Implementation gap stated now:** the
  court needs `H*`'s or `H`'s final checkpoint, which the running pilot produces.
- **Items.** `64` calibration windows (`65,536` tokens) from the registered
  training stream, in stream order, disjoint from every evaluation item. An item
  is one window; its causal features are the 8 registered token-id statistics
  (log length, mean id, std id, first id, last id, unique fraction, and two
  length/marker terms) computed from the window's tokens. No label, no teacher
  target and no model output is a controller input.
- **Four conditions.** Both paths toggled by forward hooks that zero the
  module's output: `none`, `m` (memory active), `e` (experts active), `both`.
- **Metric.** `R = -mean` T1.4 per-token distillation loss over the window
  (higher is better). The artifact stores all four rewards, the per-item
  interaction, an input hash per item, the item-id hash, the feature hash and the
  leakage assertions that were executed. The runner receives a `CausalView`
  whose outcome keys raise.
- **Split.** Alternating over the id-sorted set: 32 train / 32 test.

## P9.3.2 Estimator and decision labels (fixed now)

`I = R_both - R_m - R_e + R_none`, per item, summarised with a percentile
bootstrap over items (`10,000` resamples, seed `20261115`) plus the standard
error. Labels: lower bound above zero → **synergy**; upper bound below zero →
**antagonism**; interval containing zero → **no detectable interaction**. The
last is never called "independence".

## P9.3.3 Controller arms (fixed now)

1. **additive-utility** — ridge predictions of `u_m(x)`, `u_e(x)`; subset value
   is the sum; no cross term. This is `H*`'s independent-utility weighting.
2. **interaction-aware** — the same two heads plus `I_hat(x)` from the same
   causal features; the pair subset's value includes `I_hat`.
3. **independent-gates** — one thresholded gate per capability.
4. **load-balanced** — always both, zero parameters.
5. **combinatorial-oracle** — the best of the four measured subsets per item;
   not deployable, reported separately.

All controllers choose a subset; the realised reward is looked up from the
measured four-condition table for that item, so the comparison is paired and
never re-measures. Learned-parameter counts are reported per arm.

## P9.3.4 Thresholds and kill condition (fixed now)

- **If `I` is detectable** (its CI excludes zero): the interaction-aware
  controller must beat the additive controller on realised reward on the test
  half by at least **`0.5%` relative** (or `0.01` absolute when the arm's mean
  reward is below `2.0`) with a paired 95% CI lower bound above zero. If it does
  not, the interaction-aware controller is recorded as **not supported**: the
  cross term is measurable but not usable by this controller class.
- **If `I` is not detectable:** the additive target is adequate for this pair at
  this budget; that is the result, and the interaction controller's extra
  parameters are recorded as unjustified.
- **Oracle reported separately, never as a deployable result.** The gap to the
  oracle is reported, and a controller that matches the oracle exactly is
  checked for leakage before it is believed (the manifest is the evidence).
- `n = 32` test items: any difference smaller than the item-level CI half-width
  is reported as "not separable from noise".

## P9.3.5 What a negative result means

- **`I` detectable but the controller does not use it.** The target is right and
  the controller class is too weak; `P9.3`'s next step is a stronger
  differentiable top-`k` surrogate, and no claim is made that interaction-aware
  routing works.
- **`I` not detectable.** The T2.5 additive utility was adequate for this pair;
  the T2.5/T2.6 risk is not what limited the hybrid, and the miss is elsewhere.
  This is stated as a negative result for the interaction term, not for
  composition (P8.2 answers that).

---

# Dated addition — 2026-11-16 (S21, P14.4): capability suite registration

The P12.9 fresh Rust suite referenced by P8.2.6 and P11.1.1 was expanded to the
size recorded in `docs/CAPABILITY-COURT.md` §2, with a triple-surface trio. This
addition changes no threshold, arm, kill condition or historical measurement.

- The original registration `docs/capability-preregistration.json` remains
  byte-for-byte as written and continues to bind the historical 24-task suite.
- The expanded suite is registered in
  `docs/capability-preregistration-v3.json` with the same 10% relative primary
  threshold, the same paired category-stratified bootstrap (seed `20261115`,
  10,000 resamples), the registered per-category minimum, and the reported trio
  surfaces (`rust`, `numeric-graph`, `logic`). v2 registered the first pass and
  was superseded without being edited after the near-duplicate audit. The final
  suite binding is the hash in `fixtures/capability/MANIFEST.json`.
- P8.2.6's "fresh task set and its contamination receipt are copied by
  reference (hash)" now means the expanded manifest and the contamination
  receipt `results/capability/contamination-20261116T080000Z.json`.
- P11.1.1's 200-item P12.9 failure artifact and its target-layer registration are
  unaffected and still do not exist at this revision; the expansion does not
  supply that artifact.

---

---

# Dated addition — 2026-11-16 (S26, P14.4/P14.7): the doubled suite and the sealed holdback

The P12.9 fresh suite was doubled to the size recorded in
`docs/CAPABILITY-COURT.md` §2 (with the per-category count and the unchanged
trio and probes), and a sealed holdback was frozen. This addition changes **no** registered arm,
threshold, decision rule, bootstrap, seed or historical measurement. It
supersedes `docs/capability-preregistration-v3.json` without editing it.

- **Registration.** `docs/capability-preregistration-v4.json` binds the scored
  suite at `suite_sha256 =
  7347e1e94b7ddcd3d04f1580646b392fc1591a25e71b299a663fc7f837f0a207`. The
  primary endpoint is unchanged in definition — the equal-weight mean of the
  eight category means — with the item count and the per-category minimum
  recorded in the registration. The 10% relative threshold, the paired
  percentile bootstrap (seed `20261115`, 10,000 resamples, level 0.95), the
  0.05 absolute secondary regression limit and the trio rule are unchanged.
- **Power, reported not assumed.** The per-category single-proportion 95% CI
  half-width and the registered paired composite's simulated 80%-power minimum
  detectable effect are emitted from
  `results/capability/power-20261116T090300Z.json` and printed in
  `docs/CAPABILITY-COURT.md` §7.1. The simulated MDE is above the registered 10%
  threshold, so **a 10% relative improvement is not resolvable at 80% power at
  this n.** The threshold is unchanged; a null near the threshold is
  under-powered and must be reported as such.
- **Sealed holdback (P14.7).** Three further tasks per category, frozen at
  `sealed_sha256 =
  7666768f3abdbd93c268cc6d10250ddefa13a3a58c94268863fdcab79e7984bc`, registered
  in `fixtures/capability/SEALED.json` with every item hash, the freeze date,
  the no-tuning rule and the unlock procedure. The harness refuses to score the
  sealed set unless `--unlock-sealed` and the registered hash are supplied
  together, writes an unlock record with the UTC time before checking any item,
  and treats the registration as append-only. The sealed set may not be used for
  tuning, prompt selection, few-shot examples, checkpoint selection or threshold
  selection. A sealed result is published once per revision with its unlock
  record. No arm has been run against it.
- **Checker strengthening.** One task per category declares a **naive fix** — the
  plausible shallow answer — and the self-test requires it to compile and then
  fail a hidden behavioural test. The counts are in
  `results/capability/selftest-20261116T090000Z.json`, whose every declared
  naive fix was rejected. The stdout-forgery probe remains, so a solution that
  prints a passing test summary is recorded as a failure.
- **Near-duplicate screen.** `scripts/capability_similarity.py` compares the
  diagnostic, code and assertion views of every authored item with thresholds
  registered before the run (`results/capability/similarity-20261116T090600Z.json`).
  No pair involving a new item exceeds a threshold. Two pairs from the
  already-registered v3 suite are above a threshold and are reported for a future
  supersession, not rewritten here.
- **Contamination.** `results/capability/contamination-20261116T090400Z.json`
  covers the scored suite, the trio and the sealed set against HumanEval, GSM8K,
  TruthfulQA, the pinned T2 windows and the bounded corpus sample (counts in the
  receipt), and reports `disjoint`. The first expansion run found one coincidental
  13-token overlap in a sealed item; the fixture's literals were changed and the
  run re-passed.

---

# Experiment contract — P13.14: omission-conditioned recovery (dated addition, 2026-11-16, S23)

Status: **pre-registered before any measurement of this court.** Written while
`H*` was still training and before any compensation run existed. It does not
modify T2, T1, M1, M2, P8.2, P9.3, P11.1 or T3: every arm, budget, threshold and
kill condition in those sections stands verbatim. The implementation is
`oracle/train/recovery.py` (the compensator, the alignment baseline, the
accounting and the arm matrix); the runner is `oracle/eval/recovery_court.py`
(`--plan` needs no model, `--smoke` is a tiny-model CPU path); the CPU tests are
`oracle/tests/test_p13_recovery.py`. Nothing here has been run on the GPU.

**Identifier note.** The session brief that produced this work calls the item
`P11.9 omission-conditioned recovery`. In `docs/PROGRAM-PLAN.md` the item is
**P13.14** ("Omission-Conditioned Recovery"); plan item P11.9 is
*Counterexample-Constrained Routing*, a different mechanism that this
registration does not touch. Only P13.14 is registered here.

## P13.14.1 Question

Skipping a group changes the layer's output in a way a fixed model cannot undo.
Is a correction **conditioned on which groups were omitted** better, at matched
bytes, than the strongest set-independent correction — layer-wise output
reconstruction, the step GPTQ (arXiv 2210.17323, VERIFIED) performs for
quantisation error?

## P13.14.2 The corrected layer and its causal inputs (fixed now)

```
h' = F_sparse(h, S) + C_theta(h, S)
```

- `F_sparse` is the arm's own deployed sparse path: the router's top-`k`
  specialists at their softmax weight. It is never modified.
- `S` is that same selection, read from the same router on the same hidden
  state in the same forward pass. The omission set is its complement. The
  compensator cannot see an omission list the router did not produce.
- `C_theta(h, S) = W_up · act(W_down · [h ; enc(S)] + b_down) + b_up`, with
  `enc(S) = sum_{e in S'} E_e` a learned per-expert embedding and `S'` the
  omitted set (`--set-mode omitted`, the registered default) or the selected set
  (`--set-mode selected`, reported as a robustness variant).
- Size: `rank = 8`, `set_dim = 8`, one per routed layer, `silu`. Element count
  `m·d + (hidden+d)·r + r + r·hidden + hidden = 17,544` per layer,
  **491,232** for 28 layers at `hidden = 1024`, `m = 8`. `--plan` prints it from
  the same constants the runner uses, so the registration cannot drift from the
  implementation.
- `W_up` and `b_up` are **zero-initialised**, so at step 0 the corrected layer is
  the sparse layer bit for bit. `test_zero_initialised_compensator_reproduces_the_sparse_layer_exactly`
  is the evidence, and the run's receipt records the zero point.
- **Causal inputs only.** The forward signature is `(hidden_states, selected)`;
  `causal_input_signature()` returns exactly that tuple and the test pins it.
  No label, no teacher logit, no future token, no omitted group's activation and
  no evaluation item reaches it. Training uses the registered distillation
  objective on the training stream, the same objective and stream the base arm
  trained on, so the compensator cannot learn anything the base arm could not.

## P13.14.3 The alignment baseline (the hard one)

A **set-independent affine low-rank map** `A(h) = h·V·Uᵀ + b` fitted in one
closed-form step per layer on calibration activations, then truncated to the
rank the compensator's own element budget allows:

- `mode="l2"` (primary): ridge least squares on `(X, Y_dense − Y_sparse)`, i.e.
  the layer-wise output reconstruction GPTQ performs. The correspondence is a
  **reading**, not an identity: GPTQ alters quantised weights and compensates
  the residual through a Hessian inverse, while this fits a residual map. It is
  recorded as a reading in every receipt.
- `mode="ot"` (reported): the Gaussian moment-matching map, which reproduces the
  target's output mean and covariance, so the baseline cannot be dismissed as
  weak on distribution shape.
- `X` is the bank's input on the same calibration slice the compensator sees;
  `Y_dense` is the unit-weight full bank, the reference T2.5's utility labels
  already use.
- **Matched bytes, fixed now.** An affine low-rank map costs
  `2·hidden·rank + hidden` elements. `matched_alignment_rank` floors the rank so
  the baseline's count is *at or below* the compensator's, and fails closed when
  the budget cannot afford rank one. At the registered size this is rank `8`:
  `17,408` per layer, **487,424** for 28 layers, against the compensator's
  **491,232**. A win therefore cannot come from extra parameters.

## P13.14.4 Arms and the budget (fixed now)

Base arm: **`X`** (`results/artifacts/train/t2/X/step-001024.pt`), because a
specialists-only arm has one path, so "the omitted computation" is unambiguous.
`H` is registered as a secondary replication that is **not decisive** (its
memory path would confound what is being omitted). Five arms, one base, one data
order, one omission rate `(m − k)/m = 0.5`:

| arm | correction | recovery elements | realized active |
| --- | --- | ---: | ---: |
| `sparse-only` | none | 0 | 772,440,064 |
| `sparse+alignment` | set-independent affine low-rank, closed-form | 487,424 | 772,927,488 |
| `sparse+compensator` | `C_theta(h, S)`, trained | 491,232 | 772,931,296 |
| `sparse+alignment+compensator` | both | 978,656 | 773,418,720 |
| `dense-teacher-reference` | none; all `m` groups at unit weight | 0 | not deployable |

The M2 accounting rule (M2.2) applies unchanged: every learned selector or
compensator weight is charged to the arm's active unique element count, and two
arms are compared only at identical realized counts. No arm exceeds the
`1e9` ceiling; the largest is 773,418,720. `--plan` regenerates this table.

## P13.14.5 Data and protocol (fixed now)

- Training stream: `results/artifacts/train/train.bin`, the registered corpus,
  starting after the calibration slice the arm was already labelled on. No
  evaluation item is in it (T1.3/T1.5 contamination filtering).
- Calibration slice: the first 8 windows (8,192 tokens) of the stream, disjoint
  from the evaluation windows. Capture bytes are bounded and recorded:
  `3 · positions · hidden · 4` per layer.
- Compensator training: `512` steps, AdamW `lr = 5e-4`, betas `(0.9, 0.95)`,
  weight decay `0`, gradient clip `1.0`, batch `2 × 1024` tokens, seed `1001`.
  Only the compensator is trainable (`freeze_arm_but_recovery`), which the
  receipt records.
- Alignment fitting: `ridge = 1e-3` relative to the mean feature variance,
  one closed-form solve per layer, no optimisation steps.

## P13.14.6 Metrics, statistics and decision order (fixed now)

Instrument: `oracle/eval/harness.py`, the same primitives `baseline.py` (T2.9)
calls — KL to the distillation teacher, top-1 agreement, per-domain PPL, and the
three task accuracies. The registered instrument is not modified.

- Primary: paired item/document differences in `KL`, `top-1` and per-domain
  `PPL`, percentile paired bootstrap, seed `20261115`, `10,000` resamples,
  stratified by domain exactly as T2.9 does it.
- Secondary: `humaneval pass@1`, `gsm8k exact match`, `truthfulqa mc1`, paired
  by item id.
- Decision order, applied in this order and reported in it:
  1. **Validity.** Ledger equality, the zero point, the causal-input signature
     and the frozen-parameter report. Any failure refuses the arm rather than
     reporting it.
  2. `sparse+compensator` **versus** `sparse+alignment` at matched bytes.
  3. `sparse+compensator` **versus** the shuffled-condition control (below).
  4. **Task relevance** (P13.14.7).
  5. Report the `dense-teacher-reference` gap as the recoverable ceiling; it is
     never a deployable result.

## P13.14.7 The falsifying control and the task-relevance clause (fixed now)

**Shuffled condition.** A compensator of the same size, trained identically, but
conditioned on an independently drawn omitted set of the same size
(`recovery.shuffle_sets`). It knows the omission *rate* and nothing about the
identity. A conditioned compensator that only matches this control has not
learned the omission; it has learned the rate.

**Task relevance — the clause this court exists to enforce.** Recovering
activation statistics is not the claim. The compensator must recover
**task-relevant transformations**, so:

- a `KL`/`PPL`-only win **does not** satisfy this court, whatever its margin;
- at least one registered task accuracy must improve over `sparse-only` with a
  paired CI excluding zero, **or** the *correct-answer recovery rate* — the
  share of positions/items where the sparse arm's wrong answer is replaced by
  the teacher's correct one minus the reverse rate — must improve with a CI
  excluding zero;
- no task may regress by more than the item-level CI half-width.

## P13.14.8 Kill condition (fixed now)

Retire the item if **any** of the following holds at matched bytes:

1. `sparse+compensator` does not beat `sparse+alignment` by at least `5%`
   relative on the mixed stratified `KL`, with a paired CI excluding zero;
2. it does not beat the shuffled-condition control by at least `10%` relative on
   the same metric with a CI excluding zero;
3. the task-relevance clause of P13.14.7 fails.

A retired item is kept as a negative result (P6.5), never deleted, and no
compensator is added to any arm's ledger.

## P13.14.9 What a negative result means, and what this court does not claim

- **Alignment wins.** The omission's effect is well approximated by a
  set-independent map on these weights; "conditioning is unnecessary" is a real
  finding and it retires the mechanism rather than the question.
- **Only the shuffled control is matched.** The gain is the extra capacity, not
  the conditioning; report it as such and do not describe the mechanism as
  omission-conditioned.
- **KL/PPL move and tasks do not.** The layer-level correction does not survive
  to the output; this is exactly the reading the task-relevance clause exists to
  block, and it is reported as a negative result.
- This court makes no claim about MoSE/GMoE/TYLER, no claim about any external
  system, and no capability claim; it is one mechanism's decisive experiment
  inside the P13 gate (`docs/PROGRAM-PLAN.md` §18).

## P13.14.10 Exact commands (not run in this session)

```bash
# (0) The arm matrix and the ledger, from the registered constants. No model.
SELVEDGE_MEM=2g SELVEDGE_CPUS=2 SELVEDGE_TMP=1g \
  scripts/court oracle -- python3 oracle/eval/recovery_court.py \
    --plan --base-active 772440064

# (1) The tiny-model CPU smoke: build, capture, fit, train-side wiring, ledger.
SELVEDGE_MEM=8g SELVEDGE_CPUS=8 SELVEDGE_TMP=2g \
  scripts/court oracle -- python3 oracle/eval/recovery_court.py \
    --smoke --out results/t2/p13-smoke-$(date -u +%Y%m%dT%H%M%SZ).json

# (2) The registered run. GPU court; size the memory from the T2 evaluation
#     instrument's measured peak (11.596 GiB VRAM on the 4080 SUPER) plus the
#     bounded capture (3 * 8192 * 1024 * 4 * 28 = 2.75 GiB of host RAM) plus the
#     default 1.5x headroom rule => SELVEDGE_MEM=32g. Record the limit and this
#     arithmetic in the launch record before starting.
SELVEDGE_MEM=32g SELVEDGE_CPUS=8 SELVEDGE_PIDS=1024 SELVEDGE_TMP=8g \
  scripts/court --gpu oracle -- python3 oracle/eval/recovery_court.py \
    --arm X \
    --checkpoint results/artifacts/train/t2/X/step-001024.pt \
    --config results/train/admission-configs/t2/X.json \
    --admission results/train/admission/t2/X.json \
    --stream results/artifacts/train/train.bin \
    --targets results/artifacts/train \
    --steps 512 --calibration-tokens 8192 \
    --out results/t2/p13/recovery-X-$(date -u +%Y%m%dT%H%M%SZ).json
```

**Preconditions before step (2) may be launched:** the tier gate of §18 has
opened (`H*` reported, first capability measurement exists); the GPU is free;
the item is promoted from `CANDIDATE` and this registration is committed.

**Status.** registered, implemented, CPU-tested (`24` tests in
`oracle/tests/test_p13_recovery.py`, the tiny-model smoke in
`recovery_court.py --smoke`); **not launched**. No measurement of any arm
exists at this revision.

# P18.6 — cut-set discovery and interaction diagnostics from stored omission artifacts

Status: **pre-registered 2026-11-16 (S25), before any statistic in
`results/P18.6-CUTSET.md` was computed.** This section is appended to the M1/M2
contract and does **not** modify any M1 or M2 threshold, arm, budget or kill
condition. It registers a new analysis over artifacts that already exist. It is
the registration P18.6 of `docs/PROGRAM-PLAN.md` §23 requires. The analysis
script is `scripts/p18_6_cutset.py`; its only inputs are the files named in
P18.6.1; it executes no model and uses no GPU.

## P18.6.1 Data sources, with the stamp and revision that produced each

| role | path | stamp | revision | binding hash |
| --- | --- | --- | --- | --- |
| pairwise interaction records (primary) | `results/selectors/interactions-20261114T232338Z.json` | `20261114T232338Z` | `1e623062379239a4bed3337ed2ed91d1a7f5411b` | sha256 `ac160ac774e0264e7e3c303e01b571a6df008a798a12a5bda5013add3bccfb9d` |
| M1 layer-omission frontier (secondary) | `results/frontier/frontier.json` | `20261114T223105Z` | `9ea017fd6071a8fb75e88f4254a9f6cf83b23c30` | sha256 `3407eb4f47ddada9a01b025d16893a2adaa898ffab7a650622d433d859b124a3` |
| M1 published report | `results/M1-FRONTIER.md` | — | reports the frontier above | — |
| M2 published report | `results/M2-SELECTORS.md` | — | reports the interaction receipt above | — |
| T2 single-expert labels (instrument-gap statement only) | `results/artifacts/train/t2/Hstar/utility-labels.pt` + `.sha256` | label stamp `20261116T053240Z` in the label manifest | artifact revision not stored in the manifest | sha256 `9e41d6049766f69ae1483867b5bdc895383be6dfd94993d2c38949caa1b7bf14` |
| T2 diagnostics receipt (instrument-gap statement only) | `results/t2/diagnostics-20261116T070700Z.json` | `20261116T070703Z` | `41e4a7e1bbeaf59c7b75bdcf28b19acbdfcf461e` | — |
| T2 arm state (instrument-gap statement only) | `results/t2/pilot-state.json` | — | — | — |

The interaction receipt records `preregistration_commit`
`a0b107adc23f23cbda43a09cbd6d229b0ea024a1` (the M2 registration) and
`data.split_sha256`
`902cc178dc3e6f5921c4b134d9a081adebde120433a917b3cf76125cf1853366`, which
binds it to `results/splits/split-20261114T223127Z.json` (sha256
`693a55437d2c6377bea9a3cca44c2ba7ac14599e5b8281a6cc0dd2bfc1a5d7b9`).

**What the primary artifact contains.** 160 contexts. Each context is one
(window, probed layer, fixed group pair) on calibration text. The probed layer
is one of layers `{0, 7, 14, 21, 27}`; the window is one of eight calibration
windows; the pair is one of four fixed pairs per layer. In the probed layer the
retained group mask takes the four values `S`, `S+g`, `S+j`, `S+g+j`, where `S`
is the static top-4 magnitude group set of that layer; every other layer keeps
its static top-5 magnitude mask. For each mask the receipt stores the exact
teacher KL in nats/token. The 160 contexts cost 640 forwards and
`49.596` GPU-seconds at the receipt's stamp. **The receipt measures retained
sets, not trained banks, and every window is calibration text.**

## P18.6.2 Definitions, fixed now

For a context `c`, write `a = KL(S)`, `b = KL(S+g)`, `c_j = KL(S+j)`,
`d = KL(S+g+j)`.

- **Unary addition effect** `m_g = b - a`, `m_j = c_j - a`. Positive means
  *adding* `g` (restoring it to the retained set) raises loss at this context.
  This is the measured non-monotonicity direction; it is not forced to be
  negative.
- **Joint addition effect** `Delta = d - a`.
- **Pairwise non-additivity**
  `E = d - b - c_j + a = Delta - m_g - m_j`. This equals `-I` where `I` is the
  receipt's `interaction` field (`I = b + c_j - a - d`), because the receipt's
  sign convention is the opposite of P16.5's. `E > 0` means the two additions
  cost **more** than the sum of their unary effects (super-additive loss
  increase; a submodularity violation). In omission terms, with `O` the
  complement of `S` in the probed layer, the same four-point quantity is the
  interaction of the *omission* damages, so `E > 0` is P16.5's positive `b_ij`
  (joint removal worse than additive). `E < 0` is sub-additive (redundancy).
- **Far-worse interaction** (pre-registered): `E > max(0.01 nats,
  0.25 * (|m_g| + |m_j|))`.
- **Measured-minimal destructive omission set.** Fix the primary threshold
  `epsilon = 0.25` nats, the M2 registered usable-tier KL bound. The measured
  nested family is `{O, O\{g}, O\{j}, O\{g,j}}`, with `L(O) = a`,
  `L(O\{g}) = b`, `L(O\{j}) = c_j`, `L(O\{g,j}) = d`.
  - `O` is measured-minimal destructive iff `a > epsilon`, `b <= epsilon` and
    `c_j <= epsilon`.
  - `O\{g}` is measured-minimal destructive iff `b > epsilon` and
    `d <= epsilon` (its only measured immediate subset is `O\{g,j}`);
    `O\{j}` is symmetric.
  - `O\{g,j}` is reported destructive or not, but cannot be classified
    minimal because none of its immediate subsets was measured.
  A distinct configuration is `(layer, g, j, kind)` with `kind` in
  `{O, O\g, O\j}`. `S` is static per layer, so there are at most 60 distinct
  configurations, each replicated over the eight windows.
  The sensitivity values `epsilon` in `{0.05, 0.10, 0.25, 0.50, 1.00}` are
  reported descriptively; only `0.25` is decisive.

**Every statistic is computed from these stored numbers. No missing
measurement is imputed, and no model is executed.**

## P18.6.3 Split, fixed before the numbers are read

The eight calibration windows are ranked by the SHA-256 of the source article
recorded in the split receipt. The four lowest hashes are the **discovery**
half; the four highest are the **replication** half. The ranking is independent
of every measured KL. The assignment is:

| half | windows (documents) |
| --- | --- |
| discovery | `wikitext-article-020`, `wikitext-article-055`, `wikitext-article-042`, `wikitext-article-039` |
| replication | `wikitext-article-038`, `wikitext-article-019`, `wikitext-article-050`, `wikitext-article-044` |

Each half has 4 windows x 5 layers x 4 pairs = 80 contexts. **This is an
internal discovery/replication split inside the calibration corpus. It is not a
held-out-corpus split.** The original M1/M2 held-out articles, the code slice
and the math slice contain no interaction measurement of any kind. A result
that replicates here has replicated across calibration documents, not out of
the corpus.

The M1 layer-omission check is out-of-sample by construction: its unary model
uses only the 28 single-layer omission arms, while the two 20-layer joint arms
are separate measurements the unary model never sees. No split is applied to
that check.

## P18.6.4 Statistics, fixed now

**S1 — interaction distribution and sign asymmetry.** For all 160 contexts and
per layer: `n`, mean, median, IQR, sd, min, max, positive/negative/zero
fractions of `E`; mean and median `|E|`; the fraction of contexts with
far-worse `E`. Sign asymmetry is reported as (a) the distribution of the
marginal-magnitude asymmetry `|m_g| - |m_j|` (mean, median, IQR) and (b) the
number of sign changes of `E` across the eight windows for each of the 20 fixed
`(layer, g, j)` cells (min, median, max; fraction of cells with at least one
flip). `E` is algebraically symmetric in `g` and `j`; no directional asymmetry
is claimed from swapping them.

**S2 — unary-only versus pairwise-additive explanation.** In-sample: mean
`|Delta|`, mean `|E|`, the ratio `mean|E| / mean|Delta|`, and
`R2_unary = 1 - sum(E^2) / sum((Delta - mean Delta)^2)`. The measured pair
interaction is exact in-sample by construction, so an in-sample pairwise `R2`
of one is **not** evidence and is labelled as such. Out-of-sample is the
decisive form: for each of the 20 `(layer, g, j)` cells estimate
`E_D(cell)` as the discovery mean, then on the 80 replication contexts compare
the actual `Delta` against three predictions — **(U)** unary only
`m_g + m_j`; **(I)** unary plus the discovery cell interaction;
**(G)** unary plus the global discovery mean interaction. Report MAE, RMSE,
`R2` against the constant-mean predictor, and the replication sign agreement of
`E` with `sign(E_D(cell))`. Report the Spearman correlation across the 20 cells
between `E_D` and the replication mean `E` descriptively.

**S3 — monotonicity and submodularity violations.** Count contexts with
`E > 0` (super-additive), `E < 0` (sub-additive) and `E = 0`; count far-worse
contexts. Report the sequential marginal comparison explicitly:
`Delta(j | S+g) = d - b` versus `Delta(j | S) = c_j - a`; the event
`Delta(j|S+g) > Delta(j|S)` is algebraically identical to `E > 0` and the
identity is reported as a correctness check, not as a second finding.

**S4 — measured-minimal destructive sets and replication.** At
`epsilon = 0.25`: the number of contexts containing a measured-minimal
destructive configuration, the number of distinct `(layer, g, j, kind)`
configurations, and the number that **replicate**, where a configuration
replicates iff it is measured-minimal destructive in at least 3 of the 4
discovery windows **and** at least 3 of the 4 replication windows. A weaker
"destructive direction" count is also reported: `L > epsilon` in at least 3 of
4 windows in both halves, regardless of subset safety. The `epsilon`
sensitivity list in P18.6.2 is reported descriptively.

**S5 — coverage, using the stored design as the population.** The population is
the 160 cells `(layer, pair, window)`. A cell is **damaging** iff `E > 0.01`
nats (the pre-registered effect floor: one fifth of the acceptable-tier KL
bound `0.05`). Secondary labels `E > 0.1` and "measured-minimal destructive at
`epsilon = 0.25`" are reported descriptively. At equal evaluation counts
`N` in `{20, 40, 80, 120}` compare:

- **random design:** 10,000 uniform draws without replacement, seed `20261116`;
- **covering design:** a strength-2 covering array over the three factors
  `layer` (5 levels), `pair` (4), `window` (8), constructed greedily to maximize
  newly covered factor-level pairs; deterministic tie-break by the SHA-256 of
  the cell key, plus 1,000 randomized tie-break variants (seed `20261116` plus
the draw index), because the damage label is never used in construction.

Metrics: recall of damaging cells, precision, and the number of the 20
`(layer, pair)` strata that contain at least one damaging cell and are touched
by the design. The covering design cannot reach full pairwise coverage below
`N` = 31; the achieved fraction is reported at every `N`. **This simulation is
over the stored 5-layer x 4-pair design, not over the group-pair space: only 20
of the 3,360 possible `(layer, group-pair)` cells were ever measured.**

**S6 — M1 layer-omission additivity, secondary and descriptive.** From
`frontier.json`, take the 28 single-layer omission KL means and the two joint
arms `layer-omit-first-20` (layers 0-19) and `layer-omit-last-20` (layers
8-27). Compare each measured joint KL against the naive additive prediction
`sum_{l in T} KL_single(l)`. Report the ratio and absolute residual. This is a
**different instrument** (whole-layer FFN omission on a dense teacher, joint
arms above the 1e9 ceiling), so it is never pooled with S1-S5 and never used to
set a threshold. It is a sanity check on whether the program's stored artifacts
contain any visible non-additivity at another granularity.

## P18.6.5 Decision rules and kill conditions, fixed now

These are evaluated once, on the primary `epsilon = 0.25` and `tau_E = 0.01`
registration. The sensitivity values cannot change a verdict.

- **D1 — the interaction is non-negligible.** At least 10% of contexts are
  far-worse **and** mean `|E| >= 0.05` nats. Otherwise the pairwise term is
  second-order on this sample and the diagnostic is negative for structure.
- **D2 — the interaction replicates out of sample.** On the replication half,
  `sign(E)` agrees with `sign(E_D(cell))` on at least 70% of contexts **and** the
  cell-interaction predictor **(I)** reduces MAE by at least 10% relative to
  **(U)** with an absolute improvement of at least `0.005` nats. The 10,000-
  resample paired bootstrap over the four replication windows reports the CI of
  the MAE difference; it is reported, not required, because four units are few.
  If D2 fails, P16.5's own kill condition applies: the sign is not stable across
  splits, the interactions are not established, and no constrained selection
  may be built on them.
- **D3 — a minimal destructive set replicates.** At least one distinct
  measured-minimal destructive configuration replicates under the
  3-of-4 + 3-of-4 rule. **If D3 fails, P18.6's kill criterion fires: the result
  is reported negative and nothing is fed to P16.12.** This is the plan's own
  condition and is not renegotiated here.
- **D4 — covering arrays beat random sampling at equal evaluations.** At
  `N = 80`, the covering design's median damaging-cell recall is at least 1.5x
  the random design's median **and** the covering 5th percentile exceeds the
  random 95th percentile. Otherwise the stored population provides no evidence
  that covering arrays are the better use of evaluations.

**What justifies a GPU covering-array experiment.** Only `D1 and D2 and D4`
justify the P16.12-style covering-array search on trained banks. If `D1 and D2`
hold but D4 fails, the justified GPU step is a **full-factorial interaction
census** on the trained banks, not a covering-array search, because no
stored-data evidence favours the array. If D1 or D2 fails, no GPU interaction
experiment is justified by this evidence; the next step is to fix the
instrument (a trained-bank combination label, not another analysis of these
artifacts). D3 is deliberately **not** part of the GPU-justification rule: the
stored `epsilon` is the M2 usable-tier bound on a degraded instrument, and a
real cut set is a property of the trained model that this instrument cannot
decide. D3 still decides whether anything is fed to P16.12.

## P18.6.6 The GPU experiment that would be registered after the gate opens

This is a **plan and a cost estimate**, not a measurement, and it may not run
before the §20.0 gate opens. It is registered now so the sign of the stored
diagnostic cannot be used to substitute a different experiment later.

- **Arms (minimal):** `X` and `H` (trained, admitted; `results/t2/pilot-state.json`),
  plus `H*` once its training and the registered reader finish. `D` and `M` are
  optional controls. Every arm is evaluated with its own trained bank and its
  own router; no post-hoc omission is applied to a dense teacher.
- **Question:** for each routed layer, measure `L(T)` for the full bank and for
  every **2-omission** and **3-omission** set `T` over the 8-experts-per-layer
  bank (`C(8,2)=28` and `C(8,3)=56` sets), on a discovery split and a
  replication split. That is `1 + 28 + 56 = 85` set evaluations per routed
  layer, using the measured pairs so a triple costs one new forward.
- **Splits:** 32 calibration windows (discovery) and 32 held-out general
  windows (replication), 512 tokens each; code and math remain untouched until
  the capability court uses them.
- **Forwards per arm:** `28 layers x 85 sets x 64 windows = 152,320` (full
  factorial). At the interaction receipt's measured `49.596 / 640 = 0.0775`
  GPU-seconds per 512-token forward, that is an **estimated 3.3 GPU-hours per
  arm**, or **~6.6 GPU-hours for X and H** before overhead. This is an estimate
  from a different instrument and is labelled as one; the launch record must
  replace it with a smoke-measured peak and its own `SELVEDGE_MEM` derived by
  the §0.2 rule. The interaction receipt's own `49.596` GPU-seconds is the only
  measured cost quoted here (stamp `20261114T232338Z`, revision
  `1e623062379239a4bed3337ed2ed91d1a7f5411b`).
- **Controls:** (i) random omission sets of equal size, matched in number to the
  discovered damaging sets; (ii) a `b_ij := 0` additive selector, which must
  lose if the interactions are real; (iii) the replication half, which the
  discovery half never sees.
- **Covering-array design (only if D4 holds):** a strength-2 covering array over
  the factors `(routed layer, group-pair index, window)` at equal evaluation
  count to the random control. The stored design has only four pairs per layer,
  so the array cannot be designed from it; it must be generated over the real
  `C(8,2)=28` group pairs and committed before the pass.
- **Prerequisite instrument gap this experiment closes:** `X` and `H` have no
  utility-label artifact at all, and `H*`'s only label artifact
  (`utility-labels.pt`, sha256 `9e41d604...`) is a **single-expert, full-bank
  leave-one-out** label (`u_e = L(+all) - L(-e)`); it contains no pair or triple
  omission. The GPU experiment above is the minimal pass that creates the
  missing combination measurements.

## P18.6.7 What this diagnostic can and cannot conclude

- **Can:** describe the distribution and sign behaviour of the measured
  pairwise non-additivity on 160 stored calibration contexts; test whether a
  discovery-half interaction estimate predicts a replication-half interaction;
  count measured-minimal destructive configurations and test their replication;
  compare covering-array and random designs **over the stored design**; and
  state exactly which measurements are missing.
- **Cannot:** establish a cut set for a trained model. The M2 interactions come
  from a **degraded, non-deployed selection regime**: a dense teacher with
  static magnitude masks in 27 layers and a top-4/top-5 magnitude mask in the
  probed layer. The stored KL values are far outside the usable tier in early
  layers, so the `epsilon = 0.25` test is not a capability test. The interaction
  sample is 8 calibration windows x 5 layers x 4 fixed pairs; it is not a census
  and it cannot bound unmeasured pairs. No code or math item was measured. The
  trained banks `X` and `H` have **no** combination measurements, and `H*` has
  only single-expert labels. Nothing here may be extrapolated to the trained
  models, and no result here may be presented as a capability result.
- **If the evidence is too thin:** the report says so in plain words, reports
  D1-D4 and the kill criterion exactly as measured, and recommends the GPU
  experiment only under the rule in P18.6.5.

## P18.6.8 Reproduction

```bash
SELVEDGE_MEM=8g SELVEDGE_CPUS=8 SELVEDGE_PIDS=1024 SELVEDGE_TMP=2g \
  scripts/court python -- python3 scripts/p18_6_cutset.py \
    --out-json results/p18.6/cutset-$(date -u +%Y%m%dT%H%M%SZ).json \
    --out-md results/P18.6-CUTSET.md
```

The script is deterministic (bootstrap seed `20261116`), uses only the Python
standard library, reads only the paths in P18.6.1, and writes the JSON receipt
and the generated Markdown. It refuses to run if any input hash differs from
the hash recorded above. The court enforces the hard memory, swap, pids, CPU
and `/tmp` limits; the host is never used to execute the analysis.
