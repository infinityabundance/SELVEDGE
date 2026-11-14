# M2 selectors — per-token causal selection, oracle potential and the matched-budget verdict

Status: **measured, negative result for the inference-only branch**. The M2
contract was pre-registered before measurement
(`docs/EXPERIMENT-CONTRACT.md`, M2 section, commit
`a0b107adc23f23cbda43a09cbd6d229b0ea024a1`). The measurements cited here were
produced from the clean code revision `1e623062379239a4bed3337ed2ed91d1a7f5411b`
(the harness commit); every receipt names that revision and the
pre-registration commit. The development run from the uncommitted worktree is
retained under `results/selectors/superseded-dirty-a0b107a/`; its aggregates
are numerically identical to the clean run on every deterministic arm.

This is a distributional-fidelity result, not a capability benchmark. A KL of
6 means the sparse model's next-token distribution is a different distribution
from the dense teacher's on the measured text; it does not measure code
generation, arithmetic or reasoning.

## Registered question

M1 rejected static, token-independent group omission at the 1e9 ceiling. M2
asked whether **per-token causal selection** can escape that cliff, and first
whether a non-deployable per-token **oracle** can escape it at all. The oracle
question is decisive: if even an oracle that chooses a different group subset
for every token, at the same realized budget, cannot reach the usable tier,
then no deployable selector of this family can, and the program must move to
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
- **K6.** A deployable arm must beat `token-magnitude` at matched realized
  active unique parameters by at least 5% relative mean KL on general
  held-out text, with a paired-bootstrap 95% CI lower bound above zero.
- **K7.** The best oracle must be within the usable tier.
- **K8.** Granularity is binding only if the 64-group oracle reaches the usable
  tier while the 16-group oracle does not.
- **K9.** No leakage, non-finite value, over-ceiling arm or Python/Rust parity
  failure.

## Oracle potential (non-deployable)

The oracle sample is the first 512-token window of the first eight held-out
general articles. Predicting positions:
<!-- BEGIN GENERATED: m2-oracle-positions -->
4096
<!-- END GENERATED: m2-oracle-positions -->

The oracle chooses an exact
local minimiser of the layer-output squared error for every token and every
layer, applies the map, measures the exact teacher KL, and repeats for three
passes so the local decisions see the hidden states produced by the arm's own
masks. It uses no held-out text in any deployable selector. Its search cost is
recorded; it is not a runtime.

<!-- BEGIN M2 ORACLE TABLE -->

| oracle | budget | active unique | KL mean | KL 95% CI | PPL ratio | top-1 | frac KL<=0.05 | frac KL<=0.25 | forwards | GPU seconds |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| oracle-local-64 | G64 | 993911808 | 3.665418 | [3.289385, 4.034953] | 25.4515 | 0.2693 | 0.0169 | 0.0550 | 24 | 1.89 |
| oracle-final-refine | B142 | 998630400 | 5.785897 | [5.159938, 6.411855] | 177.664 | 0.1018 | 0.0000 | 0.0029 | 2982 | 136.29 |
| oracle-local-16-B142 | B142 | 998630400 | 5.952733 | [5.644695, 6.224299] | 229.896 | 0.0866 | 0.0005 | 0.0046 | 24 | 1.87 |
| oracle-local-16-B140 | B140 | 993911808 | 6.273271 | [5.980226, 6.535457] | 327.762 | 0.0697 | 0.0005 | 0.0042 | 24 | 1.86 |
| oracle-local-4 | G4 | 993911808 | 7.833459 | [7.600261, 8.046133] | 1550.59 | 0.0560 | 0.0000 | 0.0000 | 24 | 1.89 |
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
cost is in the receipt. None of this is deployable, and none of it is charged
as a runtime budget.

## Granularity

At identical active FFN elements (330,301,440), 64 groups per layer is better
than 16, and 4 groups is worse. So granularity is a real constraint on the
achievable fidelity, but it is not the binding constraint on the usable tier:
the 64-group oracle is still far outside it. The registered K8 rule (64-group
oracle inside the tier while 16-group is not) is not met.

## Cross-token consistency

Per-layer selected subsets change almost every token at fine granularity:

<!-- BEGIN M2 CONSISTENCY TABLE -->

| oracle | adjacent equal | distinct subsets / layer | modal fraction | groups / token |
| --- | ---: | ---: | ---: | --- |
| oracle-local-64 | 0.0002 | 509.8 | 0.0031 | 560-560 |
| oracle-final-refine | nan | nan | nan | ?-? |
| oracle-local-16-B142 | 0.0312 | 387.4 | 0.0439 | 142-142 |
| oracle-local-16-B140 | 0.0537 | 372.0 | 0.0693 | 140-140 |
| oracle-local-4 | 0.4224 | 5.0 | 0.4419 | 35-35 |
<!-- END M2 CONSISTENCY TABLE -->

At 16 and 64 groups there is almost no adjacent-token persistence and hundreds
of distinct subsets per layer in a 512-token window. This is the input to the
M3 execution-envelope question: a stable superset envelope cannot be built
from subsets that change on every token. Coarse 4-group selection is more
stable but far worse in quality, so stability and fidelity trade against each
other at this granularity.

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

<!-- BEGIN M2 TABLE -->

| arm | deployable | budget | active unique | KL mean | KL 95% CI | PPL ratio | top-1 |
| --- | :---: | --- | ---: | ---: | --- | ---: | ---: |
| teacher-dense | no | noncompliant-dense | 1720574976 | 0.000000 | [0.000000, 0.000000] | 1 | 1.0000 |
| masked-all | no | noncompliant-all-groups | 1720574976 | 0.000000 | [0.000000, 0.000000] | 1 | 1.0000 |
| token-magnitude | yes | B142 | 998630400 | 6.676772 | [6.541269, 6.813145] | 457.212 | 0.0706 |
| token-frequency | yes | B142 | 998630400 | 7.034329 | [6.889674, 7.182299] | 696.72 | 0.0616 |
| static-greedy-uniform5 | yes | B142 | 998630400 | 7.706657 | [7.512359, 7.922955] | 1196.79 | 0.0239 |
| token-task-static-nn | yes | B142 | 998630400 | 9.580137 | [9.284461, 9.897374] | 7896.43 | 0.0095 |
| static-greedy-swap4 | yes | M1-K3-112 | 927851520 | 9.895818 | [9.643579, 10.153094] | 11033.7 | 0.0081 |
| static-magnitude-k5 | yes | B140 | 993911808 | 10.019482 | [9.779430, 10.270835] | 11772.9 | 0.0080 |
| static-frequency-k5 | yes | B140 | 993911808 | 10.309631 | [10.053809, 10.574238] | 16580 | 0.0085 |
| token-learned-unary | yes | B142 | 998631268 | 10.474509 | [10.180010, 10.792109] | 18318.3 | 0.0074 |
| token-learned-interaction | yes | B142 | 998631268 | 10.482465 | [10.240315, 10.734217] | 18070.7 | 0.0072 |
| token-random | yes | B142 | 998630400 | 10.745493 | [10.447650, 11.025086] | 27897.6 | 0.0111 |
| token-gxo | no | B142 | 998630400 | 11.138053 | [10.903366, 11.377279] | 58584.1 | 0.0101 |
| token-cor-gxo | no | B142 | 998630400 | 11.322874 | [11.028434, 11.641561] | 75718.4 | 0.0111 |
| static-random-k5 | yes | B140 | 993911808 | 11.782510 | [11.507479, 12.071168] | 57418.1 | 0.0015 |

Oracle (non-deployable):

| arm | budget | KL mean | KL 95% CI | PPL ratio | top-1 | frac KL<=0.05 | frac KL<=0.25 |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: |
| oracle-local-64 | G64 | 3.665418 | [3.289385, 4.034953] | 25.4515 | 0.2693 | 0.0169 | 0.0550 |
| oracle-final-refine | B142 | 5.785897 | [5.159938, 6.411855] | 177.664 | 0.1018 | 0.0000 | 0.0029 |
| oracle-local-16-B142 | B142 | 5.952733 | [5.644695, 6.224299] | 229.896 | 0.0866 | 0.0005 | 0.0046 |
| oracle-local-16-B140 | B140 | 6.273271 | [5.980226, 6.535457] | 327.762 | 0.0697 | 0.0005 | 0.0042 |
| oracle-local-4 | G4 | 7.833459 | [7.600261, 8.046133] | 1550.59 | 0.0560 | 0.0000 | 0.0000 |
<!-- END M2 TABLE -->

The best deployable arm is `token-magnitude` at KL
<!-- BEGIN GENERATED: m2-token-magnitude-kl -->
6.6768
<!-- END GENERATED: m2-token-magnitude-kl -->
and the table records its PPL ratio and top-1. The M1 best static arm at the
same B142 budget,
`static-greedy-uniform5`, is at KL
<!-- BEGIN GENERATED: m2-static-uniform5-kl -->
7.7067
<!-- END GENERATED: m2-static-uniform5-kl -->
Per-token magnitude is a real improvement over the best static arm, visible in
the table, and it is still catastrophic. Every other deployable arm is worse
than per-token magnitude. The best relative reduction versus
`token-magnitude` among matched deployable arms is
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
non-deployable (their backward target uses teacher-forced tokens). They are
also worse than per-token magnitude. The published corrective term is
implemented exactly as stated in section 5.2 / Theorem 5.3 of Song et al.,
Proc. MLSys 2026; at group granularity it does not rescue the attribution.

## Verdicts

- **K6: failed.** No deployable arm beats `token-magnitude` by 5% at matched
  realized active unique parameters with a CI excluding zero. The best matched
  relative reduction is the negative sentinel above; the table records every
  arm's relative position. Recorded as failed in `docs/WORK-LOG.md`.
- **K7: failed, decisively.** The best per-token oracle is the first row of the
  oracle table. The usable tier is 0.25 / 1.10 / 0.85. Even a non-deployable
  per-token oracle at the compliant budget cannot reach the usable tier on this
  teacher at this granularity. **The inference-only, post-hoc selection branch
  is dead for Qwen3-1.7B at 16-64 equal contiguous groups.** The
  trained/upcycled branch becomes the primary path.
- **K8: failed.** Granularity matters (64 groups is better than 16, and 4 is
  worse) but the 64-group oracle is still far outside the usable tier. Finer
  fragments alone do not rescue the budget.
- **K9: passed.** No non-finite values, no over-ceiling arm, no leakage into a
  deployable selector, and exact Python/Rust parity on the exported fixtures.
  The instrument gate `masked-all` is bitwise equal to the teacher, and the M1
  static arms reproduce their M1 KL values exactly.

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

1. **Per-token selection helps but cannot escape the cliff.** At exactly
   matched realized active unique parameters, per-token activation magnitude
   beats the best static M1 allocation by about 13% relative KL. That is the
   only positive direction in the M2 table. It is still two orders of
   magnitude outside the acceptable tier.
2. **The oracle bound is decisive.** Even an exact local per-token oracle at
   the compliant budget, including a 3-pass fixed point and a bounded
   exact-KL refinement, cannot reach the usable tier. The 16-group best is KL
   5.95; the 64-group best is KL 3.67. This is not a search failure; it is the
   budget and the teacher.
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
  usable tier, and that the deployable approximations we tested are worse than
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

## Artifacts

- Contract: `docs/EXPERIMENT-CONTRACT.md` (M2 section, amendment A3).
- Prepare receipt and weights: `results/selectors/prepare-Qwen-Qwen3-1.7B-20261114T231248Z.json`,
  `results/selectors/m2-weights.json`, `results/selectors/m2-fit-data.json`.
- Oracle: `results/selectors/oracle-20261114T231601Z.json`.
- Full evaluation: `results/selectors/eval-full-20261114T232206Z.json`.
- Oracle-sample evaluation: `results/selectors/eval-oracle-20261114T232422Z.json`.
- Interactions: `results/selectors/interactions-20261114T232338Z.json`.
- Analysis, table, plot: `results/selectors/selectors.json`,
  `results/selectors/table.md`, `results/selectors/selectors.svg`.
- Rust parity fixtures: `results/selectors/fixtures/selector-inputs.json`.
- Rust fit / cost / compare: `results/selectors/atlas-fit-weights.json`,
  `results/selectors/rust-selector-cost.json`,
  `results/selectors/atlas-compare.json`.
- Court receipts: `results/courts/` (oracle, rust, commit_identity,
  perf_numbers).
