# M2 selectors — per-token causal selection, oracle potential and the matched-budget verdict

Status: **measured, negative result for the inference-only branch**. The M2
contract was pre-registered before measurement
(`docs/EXPERIMENT-CONTRACT.md`, M2 section, commit
`a0b107adc23f23cbda43a09cbd6d229b0ea024a1`). The prepare, full-evaluation and interaction
receipts were produced from the clean harness revision
`1e623062379239a4bed3337ed2ed91d1a7f5411b`; the fixtures were regenerated from
the reporting revision `362d2c7156feff66d55a04d84e2aff548c48e95f`; the oracle
(with the cross-window stability field) and the analysis were regenerated from
`40f19dbf62de8395cd55ebf27f5685d71769347c`. Every receipt names its revision
and the pre-registration commit. The later revisions add reporting code and
tests; they do not touch any measured quantity. The development run from the uncommitted worktree is
retained under `results/selectors/superseded-dirty-a0b107a/`; its aggregates
are numerically identical to the clean run on every deterministic arm.

This is a distributional-fidelity result, not a capability benchmark. A KL of
6 means the sparse model's next-token distribution is a different distribution
from the dense teacher's on the measured text; it does not measure code
generation, arithmetic or reasoning.

> **Correction of record (2026-11-15).** A formal review found that the M2
> instrument reported **logical selection counts**, not physically read
> parameters: `dense_group_norms` computed every gate/up channel to rank groups,
> and `MaskedMlp.forward` computed dense projections and applied a mask. It also
> used one `deployable` boolean for two independent questions. The corrected
> measurement, the two typed verdicts (`causal` and `physical`), the sparse
> execution path and its tensor-access audit, and the one-pass versus
> teacher-state comparison are in `docs/M2-CORRECTION.md`. The table below is
> regenerated under the corrected accounting. The historical M2 numbers are
> retained, but the word "deployable" no longer appears as an evaluated field:
> most arms that carried it are physically non-compliant. The oracle bound on
> **logical** group omission survives as a counterfactual result.

## Registered question

M1 rejected static, token-independent group omission at the 1e9 ceiling. M2
asked whether **per-token causal selection** can escape that cliff, and first
whether a causally unavailable per-token **oracle** can escape it at all. The oracle
question is decisive: if even an oracle that chooses a different group subset
for every token, at the same realized budget, cannot reach the usable tier,
then no causally available selector of this family can, and the program must move to
the trained/upcycled branch.

## Registered budgets and thresholds

- **B142 (primary matched table).** 142 whole groups per token, the largest
  whole-group allocation at or below the ceiling: 335,020,032 FFN elements,
  998,630,400 active unique elements. Five groups per layer plus two extras
  chosen by the arm.
- **B140 (5/16 per layer).** 140 groups, 330,301,440 FFN elements,
  993,911,808 active unique elements.
- **Granularity budget G.** 330,301,440 FFN elements at 16, 64 and 4 groups
  per layer (140 groups at width 384; 560 groups at width 96; 35 groups at
  width 1536).
- **Usable tier.** KL `<= 0.25`, PPL ratio `<= 1.10`, top-1 `>= 0.85`.
- **Acceptable tier.** KL `<= 0.05`, PPL ratio `<= 1.02`, top-1 `>= 0.95`.
- **K6.** A causally available arm must beat `token-magnitude` at matched
  realized active unique parameters by at least 5% relative mean KL on general
  held-out text, with a paired-bootstrap 95% CI lower bound above zero. The
  physical verdict is reported separately: an arm that is causally available
  but physically non-compliant cannot be executed under the ceiling.
- **K7.** The best oracle must be within the usable tier.
- **K8.** Granularity is binding only if the 64-group oracle reaches the usable
  tier while the 16-group oracle does not.
- **K9.** No leakage, non-finite value, over-ceiling arm or Python/Rust parity
  failure.

## Oracle potential (causally unavailable, physically non-compliant)

The oracle sample is the first 512-token window of the first eight held-out
general articles. Predicting positions:
<!-- BEGIN GENERATED: m2-oracle-positions -->
4096
<!-- END GENERATED: m2-oracle-positions -->

The oracle chooses an exact
local minimiser of the layer-output squared error for every token and every
layer, applies the map, measures the exact teacher KL, and repeats for three
passes so the local decisions see the hidden states produced by the arm's own
masks. It uses no held-out text in any causally available selector. Its search cost is
recorded; it is not a runtime.

<!-- BEGIN M2 ORACLE TABLE -->

| oracle | budget | active unique | KL mean | KL 95% CI | PPL ratio | top-1 | frac KL<=0.05 | frac KL<=0.25 | forwards | GPU seconds |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| oracle-local-64 | G64 | 993911808 | 3.665418 | [3.289385, 4.034953] | 25.4515 | 0.2693 | 0.0169 | 0.0550 | 24 | 1.84 |
| oracle-final-refine | B142 | 998630400 | 5.785897 | [5.159938, 6.411855] | 177.664 | 0.1018 | 0.0000 | 0.0029 | 2982 | 135.49 |
| oracle-local-16-B142 | B142 | 998630400 | 5.952733 | [5.644695, 6.224299] | 229.896 | 0.0866 | 0.0005 | 0.0046 | 24 | 1.87 |
| oracle-local-16-B140 | B140 | 993911808 | 6.273271 | [5.980226, 6.535457] | 327.762 | 0.0697 | 0.0005 | 0.0042 | 24 | 1.85 |
| oracle-local-4 | G4 | 993911808 | 7.833459 | [7.600261, 8.046133] | 1550.59 | 0.0560 | 0.0000 | 0.0000 | 24 | 1.86 |
<!-- END M2 ORACLE TABLE -->

The best oracle KL:
<!-- BEGIN GENERATED: m2-oracle-best-kl -->
3.6654
<!-- END GENERATED: m2-oracle-best-kl -->
with PPL ratio
<!-- BEGIN GENERATED: m2-oracle-best-ppl -->
25.45
<!-- END GENERATED: m2-oracle-best-ppl -->
and top-1 agreement
<!-- BEGIN GENERATED: m2-oracle-best-top1 -->
0.2693
<!-- END GENERATED: m2-oracle-best-top1 -->
The registered usable tier is KL 0.25, PPL ratio 1.10, top-1 0.85. The best
oracle is outside all three by a wide margin.

At the primary B142 budget the 16-group oracle reaches KL
<!-- BEGIN GENERATED: m2-oracle-16-b142-kl -->
5.9527
<!-- END GENERATED: m2-oracle-16-b142-kl -->
and the 64-group oracle reaches KL
<!-- BEGIN GENERATED: m2-oracle-64-kl -->
3.6654
<!-- END GENERATED: m2-oracle-64-kl -->
The 64-group oracle is much better than 16 groups (finer fragments matter) but
still more than an order of magnitude above the usable KL bound. The bounded
exact-KL refinement improved some tokens (the receipt records the accepted
swaps and prefix forwards) but its 2-window mean is still far outside the tier;
it is a 2-window subset, not an 8-window comparison.

The oracle table records the fraction of tokens whose best subset meets the
acceptable and usable per-token bounds. Even the oracle cannot make most
individual tokens acceptable.

**Oracle cost.** The oracle table records the forwards and GPU seconds per
config; the final-refinement line is a two-window subset and its forwarding
cost is in the receipt. None of this is causally available or physically
compliant, and none of it is charged as a runtime budget.

## Granularity

At identical active FFN elements (330,301,440), 64 groups per layer is better
than 16, and 4 groups is worse. So granularity is a real constraint on the
achievable fidelity, but it is not the binding constraint on the usable tier:
the 64-group oracle is still far outside it. The registered K8 rule (64-group
oracle inside the tier while 16-group is not) is not met.

## Cross-token consistency

Per-layer selected subsets change almost every token at fine granularity:

<!-- BEGIN M2 CONSISTENCY TABLE -->

| oracle | adjacent equal | distinct subsets / layer | modal fraction | cross-window Jaccard | groups / token |
| --- | ---: | ---: | ---: | ---: | --- |
| oracle-local-64 | 0.0002 | 509.8 | 0.0031 | 0.2263 | 560-560 |
| oracle-local-16-B142 | 0.0312 | 387.4 | 0.0439 | 0.2522 | 142-142 |
| oracle-local-16-B140 | 0.0537 | 372.0 | 0.0693 | 0.2638 | 140-140 |
| oracle-local-4 | 0.4224 | 5.0 | 0.4419 | 0.3566 | 35-35 |
<!-- END M2 CONSISTENCY TABLE -->

At 16 and 64 groups there is almost no adjacent-token persistence and hundreds
of distinct subsets per layer in a 512-token window. The cross-window column
compares the subsets at the same position index in two disjoint articles; the
mean Jaccard at the 16-group B142 budget is
<!-- BEGIN GENERATED: m2-cross-window-jaccard -->
0.2522
<!-- END GENERATED: m2-cross-window-jaccard -->
and it is lower at 64 groups. The subsets are not stable within a sequence or
across samples. This is the input to the M3 execution-envelope question: a
stable superset envelope cannot be built from subsets that change on every
token. Coarse 4-group selection is more stable but far worse in quality, so
stability and fidelity trade against each other at this granularity.

## Pairwise interaction (calibration only)

Measured on 160 contexts: 8 calibration windows x layers {0, 7, 14, 21, 27} x
4 fixed pairs (two low-co-activation, two high-co-activation). With the static
magnitude map in every other layer, the probed layer's mask was `S`,
`S+g`, `S+j`, `S+g+j`; each context cost four forwards.

- Mean interaction:
  <!-- BEGIN GENERATED: m2-interactions-mean -->
-0.0175
  <!-- END GENERATED: m2-interactions-mean -->
  with a 95% article-bootstrap CI that includes zero.
- The positive and negative context fractions are close to balanced; the
  receipt's `distribution` block records them exactly.
- Every fixed pair we measured changes sign in at least one context. The
  interaction is context-dependent, not a stable pair property.
- The two marginal gains differ in magnitude. The algebraic `I(g,j|S)` is
  symmetric in g and j; the measurable asymmetry is the sign change across
  contexts and the magnitude difference of the two marginals.
- The graph surrogate (co-activation, log mean-norm product, layer fraction)
  has leave-one-window-out Spearman
  <!-- BEGIN GENERATED: m2-interactions-spearman -->
-0.2783
  <!-- END GENERATED: m2-interactions-spearman -->
  and a sign accuracy near chance (the receipt records it). It anti-predicts
  held-out interaction on this sample; H2.5 fails.
- The receipt records the five largest positive and five largest negative
  contexts. Even the largest measured interaction is a small fraction of the
  base KL, so the pairwise term is a second-order effect at this granularity.

The interaction contexts count:
<!-- BEGIN GENERATED: m2-interaction-contexts -->
160
<!-- END GENERATED: m2-interaction-contexts -->

## Matched-budget selector table

All B142 arms are at 998,630,400 active unique elements except the two learned
arms, which add their
<!-- BEGIN GENERATED: m2-weights-parameter-elements -->
868
<!-- END GENERATED: m2-weights-parameter-elements -->
learned controller elements (amendment A3) and are therefore that many elements
above the baseline. They are strictly more expensive, never less. The B140 and
M1-K3 arms are flagged and excluded from K6. CIs are paired percentile
bootstraps over 40 held-out articles, 10,000 resamples, seed 20261114. The
table is generated by `oracle/m2_selectors.py analyze` from
`results/selectors/selectors.json`; do not edit it by hand.

The `causal` and `physical` columns are independent verdicts defined in
`crates/selvedge-core/src/verdict.rs`. `causal` asks whether the selector read
only pre-decision information; `physical` asks whether selection **and**
execution read only counted parameters, with a measured sparse execution. The
`-sparse` rows execute the same selection through the sliced path; their KL
equals the dense row to the printed precision. `static-*`, `token-random` and
their `-sparse` variants are the only physically compliant rows; none has
usable quality.

<!-- BEGIN M2 TABLE -->

| arm | causal | physical | budget | active unique | KL mean | KL 95% CI | PPL ratio | top-1 |
| --- | --- | --- | --- | ---: | ---: | --- | ---: | ---: |
| teacher-dense | unavailable | non_compliant | noncompliant-dense | 1720574976 | 0.000000 | [0.000000, 0.000000] | 1 | 1.0000 |
| masked-all | available | non_compliant | noncompliant-all-groups | 1720574976 | 0.000000 | [0.000000, 0.000000] | 1 | 1.0000 |
| token-magnitude-sparse | available | non_compliant | B142 | 998630400 | 6.676772 | [6.541269, 6.813145] | 457.211 | 0.0706 |
| token-magnitude | available | non_compliant | B142 | 998630400 | 6.676772 | [6.541269, 6.813145] | 457.212 | 0.0706 |
| token-frequency-sparse | available | non_compliant | B142 | 998630400 | 7.034329 | [6.889674, 7.182299] | 696.72 | 0.0616 |
| token-frequency | available | non_compliant | B142 | 998630400 | 7.034329 | [6.889674, 7.182299] | 696.72 | 0.0616 |
| static-greedy-uniform5 | available | non_compliant | B142 | 998630400 | 7.706657 | [7.512359, 7.922955] | 1196.79 | 0.0239 |
| static-greedy-uniform5-sparse | available | compliant | B142 | 998630400 | 7.706658 | [7.512359, 7.922955] | 1196.79 | 0.0239 |
| token-task-static-nn | unavailable | non_compliant | B142 | 998630400 | 9.580137 | [9.284461, 9.897374] | 7896.43 | 0.0095 |
| static-greedy-swap4-sparse | available | compliant | M1-K3-112 | 927851520 | 9.895816 | [9.643577, 10.153090] | 11033.7 | 0.0081 |
| static-greedy-swap4 | available | non_compliant | M1-K3-112 | 927851520 | 9.895818 | [9.643579, 10.153094] | 11033.7 | 0.0081 |
| static-magnitude-k5-sparse | available | compliant | B140 | 993911808 | 10.019480 | [9.779426, 10.270834] | 11772.9 | 0.0080 |
| static-magnitude-k5 | available | non_compliant | B140 | 993911808 | 10.019482 | [9.779430, 10.270835] | 11772.9 | 0.0080 |
| static-frequency-k5 | available | non_compliant | B140 | 993911808 | 10.309631 | [10.053809, 10.574238] | 16580 | 0.0085 |
| static-frequency-k5-sparse | available | compliant | B140 | 993911808 | 10.309633 | [10.053808, 10.574243] | 16580 | 0.0085 |
| token-learned-unary | available | non_compliant | B142 | 998631268 | 10.474509 | [10.180010, 10.792109] | 18318.3 | 0.0074 |
| token-learned-interaction | available | non_compliant | B142 | 998631268 | 10.482465 | [10.240315, 10.734217] | 18070.7 | 0.0072 |
| token-random-sparse | available | compliant | B142 | 998630400 | 10.745491 | [10.447647, 11.025083] | 27897.5 | 0.0111 |
| token-random | available | non_compliant | B142 | 998630400 | 10.745493 | [10.447650, 11.025086] | 27897.6 | 0.0111 |
| token-gxo | unavailable | non_compliant | B142 | 998630400 | 11.137447 | [10.902189, 11.376774] | 58540.8 | 0.0101 |
| token-cor-gxo | unavailable | non_compliant | B142 | 998630400 | 11.322907 | [11.028465, 11.641592] | 75719.9 | 0.0111 |
| static-random-k5-sparse | available | compliant | B140 | 993911808 | 11.782509 | [11.507477, 12.071168] | 57418.1 | 0.0015 |
| static-random-k5 | available | non_compliant | B140 | 993911808 | 11.782510 | [11.507479, 12.071168] | 57418.1 | 0.0015 |

Oracle (causal unavailable, physical non-compliant):

| arm | budget | KL mean | KL 95% CI | PPL ratio | top-1 | frac KL<=0.05 | frac KL<=0.25 |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: |
| oracle-local-64 | G64 | 3.665418 | [3.289385, 4.034953] | 25.4515 | 0.2693 | 0.0169 | 0.0550 |
| oracle-final-refine | B142 | 5.785897 | [5.159938, 6.411855] | 177.664 | 0.1018 | 0.0000 | 0.0029 |
| oracle-local-16-B142 | B142 | 5.952733 | [5.644695, 6.224299] | 229.896 | 0.0866 | 0.0005 | 0.0046 |
| oracle-local-16-B140 | B140 | 6.273271 | [5.980226, 6.535457] | 327.762 | 0.0697 | 0.0005 | 0.0042 |
| oracle-local-4 | G4 | 7.833459 | [7.600261, 8.046133] | 1550.59 | 0.0560 | 0.0000 | 0.0000 |
<!-- END M2 TABLE -->

The best causally available arm by KL is `token-magnitude` at KL
<!-- BEGIN GENERATED: m2-token-magnitude-kl -->
6.6768
<!-- END GENERATED: m2-token-magnitude-kl -->
and the table records its PPL ratio and top-1. It is physically non-compliant:
the selector reads all gate/up channels to rank groups and the execution is
masked dense. The M1 best static arm at the same B142 budget,
`static-greedy-uniform5`, is at KL
<!-- BEGIN GENERATED: m2-static-uniform5-kl -->
7.7067
<!-- END GENERATED: m2-static-uniform5-kl -->
Per-token magnitude is a real improvement over the best static arm, visible in
the table, and it is still catastrophic. Every other causally available arm is
worse than per-token magnitude. The best relative reduction versus
`token-magnitude` among matched arms is
<!-- BEGIN GENERATED: m2-k6-best-relative -->
-0.0536
<!-- END GENERATED: m2-k6-best-relative -->
which is negative: the closest arm is worse, not better.

The learned interaction-aware arm does not beat the learned unary arm. Its
relative KL change versus the unary arm is
<!-- BEGIN GENERATED: m2-learned-interaction-relative -->
-0.0053
<!-- END GENERATED: m2-learned-interaction-relative -->
with a CI that includes zero. The registered interaction-aware hypothesis H2.4
is not supported. The learned unary model has a positive validation rank
correlation against the local intervention labels:
<!-- BEGIN GENERATED: m2-learned-validation-spearman -->
0.3915
<!-- END GENERATED: m2-learned-validation-spearman -->
so it learns something real about the one-layer local objective; that local
objective simply does not transfer to final-KL quality at this budget.

The gradient x activation and corrected gradient x activation arms are
causally unavailable (`backward_target`): their backward target uses
teacher-forced tokens. They are
also worse than per-token magnitude. The published corrective term is
implemented exactly as stated in section 5.2 / Theorem 5.3 of Song et al.,
Proc. MLSys 2026; at group granularity it does not rescue the attribution.

## Verdicts

- **K6: failed for causally available arms; the physical question is separate.**
  No causally available arm beats `token-magnitude` by 5% at matched realized
  active unique parameters with a CI excluding zero. The best matched relative
  reduction is the negative sentinel above. Every per-token arm that could
  improve on magnitude (magnitude itself, frequency, both learned arms) is
  physically non-compliant: its selector reads all gate/up channels. The six
  physically compliant arms are static or random masks and are far worse. K6 is
  therefore failed on both readings: no arm is better, and no better arm could
  be executed compliantly.
- **K7: failed, decisively, for logical omission.** The best per-token oracle is
  the first row of the oracle table. The usable tier is 0.25 / 1.10 / 0.85. Even
  a causally unavailable per-token oracle at the compliant budget cannot reach the
  usable tier on this teacher at this granularity. **The inference-only,
  post-hoc selection branch is dead for Qwen3-1.7B at 16-64 equal contiguous
  groups.** The oracle is causally unavailable and physically non-compliant; its
  bound is a counterfactual about omission quality, not a runtime result. The
  trained/upcycled branch becomes the primary path.
- **K8: failed.** Granularity matters (64 groups is better than 16, and 4 is
  worse) but the 64-group oracle is still far outside the usable tier. Finer
  fragments alone do not rescue the budget.
- **K9: passed.** No non-finite values, no over-ceiling arm, no leakage into a
  causally available selector, and exact Python/Rust parity on the exported
  fixtures. The one-pass audit found `token-task-static-nn` reads whole-window
  shingles and it is now labelled `future_tokens`. The instrument gate
  `masked-all` is bitwise equal to the teacher, and the M1 static arms reproduce
  their M1 KL values exactly.
- **Physical compliance (audit finding, not a registered kill condition).** Six
  arms are physically compliant with measured sparse execution: the static
  magnitude/frequency/random masks, the two static greedy masks, and
  `token-random`. All are at KL above 7.7. No physically compliant arm is within
  an order of magnitude of the usable tier.

## Rust `selvedge-atlas`

The crate is implemented at `crates/selvedge-atlas`:

- `QualityObservation` records a measured intervention and names its receipt;
  validation rejects out-of-range indices, non-finite values and a base set
  containing the measured group.
- `InteractionGraph` stores unary Welford statistics and canonical
  `low < high` pairwise edges, with typed invariants (no self-loops, finite
  statistics, existing endpoints).
- `SelectorInput` is the only record a selector may read. It carries the
  pre-MLP hidden state, group features, layer, position, sequence key and an
  optional raw hidden vector. It has no field for future tokens, teacher
  logits, gold labels or the realized mask, and `serde(deny_unknown_fields)`
  makes a record that tries to carry one fail to deserialize (Court B test).
- `Selector` implementations reproduce the Python arms: magnitude, frequency,
  deterministic random, static magnitude/frequency, learned unary and learned
  interaction-aware greedy. Decisions are deterministic with ties broken by
  ascending group index.
- `atlas fit` solves the exported normal equations in f64 by Gaussian
  elimination with partial pivoting; `atlas select` replays a fixture set;
  `atlas compare` re-derives every arm's active unique parameters and runs the
  paired bootstrap in Rust.
- The two-view contract is fixed: `ValueEstimate` plus `CostEstimate` with
  `DecisionStatus::{Ready, CostUnavailable}`. The cost head is explicitly
  unavailable until M3; a missing cost model is never reported as zero.

**Parity evidence.** The exported fixture set contains
<!-- BEGIN GENERATED: m2-fixture-records -->
256
<!-- END GENERATED: m2-fixture-records -->
records drawn from real calibration contexts. The Court B parity test replays
all seven selectors on every record and requires exact equality with the Python
decisions; it passes. The fit test reproduces the Python unary weights and
interaction matrices from the exported sufficient statistics within 1e-9 and
replays the learned decisions on the fixtures. The no-leakage, graph-invariant,
calibration-monotonicity and matched-budget tests pass. The Rust
`atlas compare` report independently reproduces the Python relative reductions
and verifies the realized active counts.

**Selector runtime cost.** `atlas cost` measures release-mode inference on the
court CPU. Every selector is sub-microsecond per decision; the measured
per-decision times and derived FLOPs are in
`results/selectors/rust-selector-cost.json`. The controller is negligible
against the FFN forward, and its 868 learned elements are counted in the
budget.

## What is proven

1. **Per-token selection helps logically but cannot escape the cliff, and the
   best logical arms are physically non-compliant.** At exactly matched
   realized active unique parameters, per-token activation magnitude beats the
   best static M1 allocation by about 13% relative KL. That is the only
   positive direction in the M2 table. It is still two orders of magnitude
   outside the acceptable tier, and its selection path reads every gate/up
   channel, so it could not be executed under the ceiling
   (`docs/M2-CORRECTION.md`).
2. **The oracle bound is decisive for logical omission.** Even an exact local
   per-token oracle at the compliant budget, including a 3-pass fixed point and
   a bounded exact-KL refinement, cannot reach the usable tier. The 16-group
   best is KL 5.95; the 64-group best is KL 3.67. This is not a search failure;
   it is the budget and the teacher. The oracle itself is causally unavailable
   and physically non-compliant, so the bound is a counterfactual about
   omission quality, not a runtime result.
3. **Granularity is not the binding constraint.** 64 fragments per layer is
   better than 16, and 4 is worse, but the 64-group oracle is still far above
   the usable KL bound. Finer fragments alone are not a path to compliance.
4. **The local intervention label does not transfer.** The learned unary
   predictor has a positive validation rank correlation against its exact
   one-layer intervention labels (in the prepare receipt), yet its final-KL arm
   is worse than magnitude. The one-layer local objective is not a sufficient
   statistic for final-KL quality at this budget.
5. **Pairwise interaction is context-dependent and not predictable from the
   graph features tested.** The mean is near zero, every fixed pair changes
   sign across contexts, and the co-activation surrogate anti-predicts held-out
   interaction. This is a negative result for the interaction-aware hypothesis
   at this granularity.
6. **The Atlas infrastructure is real and parity-tested.** The Rust crate, the
   CLI, the accounting and the leakage boundary are implemented, not
   scaffolded.

## What is not proven

- It is not a capability benchmark. KL on Wikipedia text does not measure code,
  math, reasoning or factuality.
- It does not bound trained sparse models, upcycling, MoE conversion,
  quantization, a different teacher, or a different grouping. The result is
  about **post-hoc, inference-only selection on a dense checkpoint** at
  16-64 equal contiguous groups per layer.
- It does not prove that no per-token selector can work. It proves that the
  best oracle we could build at this budget and granularity cannot reach the
  usable tier, and that the causally available approximations we tested are worse than
  activation magnitude.
- The interaction sample is small (160 contexts, 20 fixed pairs). It is enough
  to reject the simple graph surrogate and to show context dependence; it is
  not a complete interaction census.
- The learned selectors use one label definition (marginal value given a
  four-group magnitude base). A different label or a stronger model could
  learn more. The oracle bound says it would still face the same budget and
  teacher ceiling.
- Absolute PPL is optimistic because the evaluation text is plausibly in the
  teacher's pretraining corpus; the KL is relative to the same teacher on the
  same tokens.

## Failures and deviations

1. The registered exact-element budget match could not be satisfied for the
   learned arms at whole-group granularity. Amendment A3 in
   `docs/EXPERIMENT-CONTRACT.md` records the 868-element controller difference,
   its direction (learned arms are more expensive), and the rule. No threshold
   or kill condition changed.
2. The 64-group oracle uses bounded greedy (beam width 1) because the
   registered text said "beam search"; exact `C(64,20)` enumeration is
   infinite. The objective is the exact incremental local error, so each
   greedy step is exact; the beam is not widened. The receipt records this.
3. The final refinement was run on 2 windows x 16 positions x 1 pass (32
   positions), matching the registered 32-position sample, not on all eight
   windows.
4. The development run from the uncommitted worktree is retained under
   `results/selectors/superseded-dirty-a0b107a/`. Its deterministic arms are
   numerically identical to the clean run; the gradient arms differ only in the
   last printed digit because the backward pass is not bitwise reproducible.
5. No NaN, Inf, OOM or timeout occurred in the clean M2 runs. Every receipt
   records `nonfinite: 0` for the evaluated arms.
6. **Correction (2026-11-15).** The original M2 instrument reported logical
   selection counts and a single `deployable` boolean. The corrected accounting
   is in `docs/M2-CORRECTION.md`; the regenerated table has `causal` and
   `physical` verdicts and `-sparse` execution rows. The pre-correction
   receipts are retained under their original stamps; they are not comparable
   on the physical axis.
7. **One-pass versus teacher-state iteration.** The old per-token arms used a
   dense state capture and a second forward. A strictly one-pass arm was added
   and is better than the two-pass arm on the eight-window oracle sample (paired
   difference excludes zero). See `docs/M2-CORRECTION.md`; it is not a
   replacement for the full-split table.
8. **`token-task-static-nn` is causally unavailable.** Its nearest-document
   assignment hashes 5-shingles of the whole evaluation window, including
   tokens after a decision position. It is labelled `future_tokens`, not
   deployable and not a per-token selector.

## Artifacts

- Contract: `docs/EXPERIMENT-CONTRACT.md` (M2 section, amendment A3).
- Correction of record: `docs/M2-CORRECTION.md`.
- Prepare receipt and weights: `results/selectors/prepare-Qwen-Qwen3-1.7B-20261114T231248Z.json`,
  `results/selectors/m2-weights.json`, `results/selectors/m2-fit-data.json`.
- Oracle: `results/selectors/oracle-20261114T233601Z.json`.
- Corrected full evaluation: `results/selectors/eval-full-20261115T005000Z.json`.
- Superseded pre-correction evaluations: `results/selectors/eval-full-20261114T232206Z.json`,
  `results/selectors/eval-oracle-20261114T232422Z.json`.
- One-pass comparison: `results/selectors/onepass-20261115T012700Z.json`.
- Sparse-read audit: `results/selectors/sparse-audit-20261115T002900Z.json`.
- Interactions: `results/selectors/interactions-20261114T232338Z.json`.
- Analysis, table, plot: `results/selectors/selectors.json`,
  `results/selectors/table.md`, `results/selectors/selectors.svg`.
- Rust parity fixtures: `results/selectors/fixtures/selector-inputs.json`.
- Rust fit / cost / compare: `results/selectors/atlas-fit-weights.json`,
  `results/selectors/rust-selector-cost.json`,
  `results/selectors/atlas-compare.json`.
- Court receipts: `results/courts/` (oracle, rust, commit_identity,
  perf_numbers).
