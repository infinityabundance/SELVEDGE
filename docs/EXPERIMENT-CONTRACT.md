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

The objective is the exact top-k cross-entropy with a tail bucket:
`L = -sum_j p_t(j) log p_s(j) - p_tail log p_s(tail)`, where `p_t` is the
teacher distribution restricted to its top-16 support plus a tail of mass
`1 - sum_j p_t(j)`, and `p_s` is the student's probability on the same support
plus the sum of the remaining student mass as the tail. This is exact given the
stored quantities and identical for every arm. No hard-label cross-entropy term
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
