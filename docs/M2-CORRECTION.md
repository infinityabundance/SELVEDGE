# M2 correction of record — exactness, dual verdicts and measured accounting

Status: **correction, measured**. This document corrects `results/M2-SELECTORS.md`
without deleting its history. It was prompted by a formal review that found
four defects in the M2 instrument and in the core types:

1. an approximate selection could be relabelled exact from a string
   justification;
2. a single `deployable` boolean conflated selection causality with physical
   compliance;
3. the reported "active" parameter counts were logical selection counts, not
   the parameter elements the selector and the execution physically read;
4. the M2 per-token arms used captured dense hidden states and a second
   forward, so their numbers were not from a one-pass online selector.

The type-level fixes live in `crates/selvedge-core/src/selection.rs`,
`crates/selvedge-core/src/verdict.rs` and
`crates/selvedge-core/src/ledger.rs`. The measurement fixes live in
`oracle/masked_mlp.py` and `oracle/m2_selectors.py`. Every measured number in
this document is generated from a receipt; the receipt stamp and revision are
recorded in `docs/perf-numbers.json`.

## 1. What `token-magnitude` actually did

`token-magnitude` used `dense_group_norms` (`oracle/m2_selectors.py`) to rank
decision. That function called `masked_mlp.dense_channels`, which computes
**every** gate and up channel of the layer (`SiLU(W_gate x) ⊙ (W_up x)`) before
taking each group's norm. The selection decision therefore read
`2 * 2048 * 6144` parameter elements per token per layer, whether or not those
groups were selected. At 28 layers this is `28 * 2 * 2048 * 6144` elements for
one token, outside every claimed active set.

`MaskedMlp.forward` then computed the full gate/up projections again and
multiplied the full `[T, intermediate]` activation by a 0/1 mask before the full
down projection. PyTorch executed dense matrix multiplies; excluded channels
were computed and discarded. The reported `active_ffn_elements` was the count
of logically selected channels, not the count of weights whose bytes were read.

The corrected accounting for the primary baseline on the full held-out split is:

- realized active unique elements (logical selection plus controller):
  <!-- BEGIN GENERATED: m2c-baseline-active -->
998630400
  <!-- END GENERATED: m2c-baseline-active -->
- selector parameter elements read to make the decision:
  <!-- BEGIN GENERATED: m2c-baseline-selector-reads -->
225485783040
  <!-- END GENERATED: m2c-baseline-selector-reads -->
  (summed over 160 windows; the per-token union is in the receipt)
- execution parameter elements read:
  <!-- BEGIN GENERATED: m2c-baseline-execution-reads -->
169114337280
  <!-- END GENERATED: m2c-baseline-execution-reads -->
- union of selector and execution elements for one output token, plus the
  backbone:
  <!-- BEGIN GENERATED: m2c-baseline-unique -->
1479926784
  <!-- END GENERATED: m2c-baseline-unique -->

That union is above the `1,000,000,000` ceiling. The arm was never physically
compliant, and the old table's `active unique` column could not show it.

## 2. Corrected M2 table (general held-out, compact)

The full table is generated in `results/M2-SELECTORS.md` and
`results/selectors/table.md`. The two verdicts are independent:

- `causal`: did the selector read only information available before its
  decision?
- `physical`: did selection **and** execution read only counted elements, and
  was the executed computation genuinely sparse and measured?

Of
<!-- BEGIN GENERATED: m2c-arm-count -->
23
<!-- END GENERATED: m2c-arm-count -->
evaluated arms,
<!-- BEGIN GENERATED: m2c-causal-count -->
19
<!-- END GENERATED: m2c-causal-count -->
are causally available and
<!-- BEGIN GENERATED: m2c-compliance-count -->
6
<!-- END GENERATED: m2c-compliance-count -->
are physically compliant. The six compliant arms are the static and random
masks executed through the sliced sparse path; none of them is useful quality.

| arm | causal | physical | why it fails |
| --- | --- | --- | --- |
| token-magnitude | available | non_compliant | dense gate/up selector reads outside the active set; dense masked execution |
| token-frequency | available | non_compliant | same |
| token-learned-unary | available | non_compliant | features require every gate/up channel |
| token-learned-interaction | available | non_compliant | same |
| token-gxo / token-cor-gxo | unavailable | non_compliant | backward target is the gold token |
| token-task-static-nn | unavailable | non_compliant | document 5-shingles include tokens after the decision |
| static-magnitude-k5-sparse | available | compliant | quality is unusable |
| static-greedy-uniform5-sparse | available | compliant | quality is unusable |
| token-random-sparse | available | compliant | chance baseline |

The measured KL for every row, with its paired-bootstrap CI, is the generated
table in `results/M2-SELECTORS.md` and `results/selectors/table.md`; the receipt
is the corrected full evaluation named at the end. This document deliberately
does not repeat those numbers outside a generated region.

`static-greedy-uniform5` at the matched B142 allocation is the M1 best static
arm; its KL is
<!-- BEGIN GENERATED: m2c-static-kl -->
7.706657
<!-- END GENERATED: m2c-static-kl -->
and its sparse variant is physically compliant. Per-token magnitude is better:
<!-- BEGIN GENERATED: m2c-baseline-kl-2 -->
6.676772
<!-- END GENERATED: m2c-baseline-kl-2 -->
but not physically compliant.

**Which arms cannot be made physically compliant at all.** Any arm that needs
the full gate/up projection to score groups (`token-magnitude`,
`token-frequency`, both learned arms, both GxO arms) cannot be compliant at this
budget: the selector's own read is `28 * 2 * 2048 * 6144` elements, which
alone is most of the ceiling and leaves no room for the backbone
and the selected FFN. The oracle arms cannot be compliant either; their search
evaluates dense group contributions. No arm in the measured family is both
causally available, physically compliant and better than `token-magnitude`.
The compliant static/greedy masks were selected offline with teacher KL on
calibration text; that one-time search is not a per-token runtime process and
is not charged as per-token work, but it means those masks are oracle-derived
constants rather than a learned compliant controller.

## 3. The oracle bound survives as a counterfactual result

The oracle result in `results/M2-SELECTORS.md` is a bound on **logical group
omission**, measured with the same masked-dense instrument. It is not a claim
about a physically compliant execution and it is not a claim that a runtime
exists. It survives as a counterfactual: even if the selection cost were free
and the mask were applied exactly, the best per-token oracle reaches KL

<!-- BEGIN GENERATED: m2c-oracle-best-kl -->
3.665418
<!-- END GENERATED: m2c-oracle-best-kl -->

on the oracle sample, far outside the registered usable tier. The conclusion
"the inference-only post-hoc selection branch is dead for this teacher at this
granularity" stands for the *logical omission* question. The word "deployable"
does not: the old M2 arms that carried it were never physically compliant.

## 4. One-pass online selection versus teacher-state iteration

`online-one-pass-magnitude` was added. Its hook runs immediately before each
wrapped MLP and selects from the hidden state that layer actually receives under
the masks already applied in the same forward. It captures no dense state and
performs no second forward. `offline-iterated-magnitude` is the old two-pass
arm: a dense first pass, captured states, then a second forward.

Both are causally available. Both are physically non-compliant for the reason in
section 2. On the same eight-window oracle sample:

| arm | budget |
| --- | --- |
| online-one-pass | B140 |
| offline-iterated | B140 |
| online-one-pass | B142 |
| offline-iterated | B142 |

Top-1 agreement, per-row accounting and measured times for each row are in
`results/selectors/onepass-20261115T044200Z.json`; the KL values and the paired
differences are generated below.

The online KL is
<!-- BEGIN GENERATED: m2c-onepass-b140-kl -->
6.529975
<!-- END GENERATED: m2c-onepass-b140-kl -->
at B140 and
<!-- BEGIN GENERATED: m2c-onepass-b142-kl -->
6.423737
<!-- END GENERATED: m2c-onepass-b142-kl -->
at B142. The offline-iterated KL is
<!-- BEGIN GENERATED: m2c-offline-b140-kl -->
6.955762
<!-- END GENERATED: m2c-offline-b140-kl -->
at B140 and
<!-- BEGIN GENERATED: m2c-offline-b142-kl -->
6.611430
<!-- END GENERATED: m2c-offline-b142-kl -->
at B142.

Paired difference (online minus offline), per window, 10,000-resample
percentile bootstrap over the eight windows:

- B140:
<!-- BEGIN GENERATED: m2c-onepass-minus-offline-b140 -->
-0.425787
<!-- END GENERATED: m2c-onepass-minus-offline-b140 -->
  with 95% CI
  [
<!-- BEGIN GENERATED: m2c-onepass-minus-offline-b140-low -->
-0.557538
<!-- END GENERATED: m2c-onepass-minus-offline-b140-low -->
  ,
<!-- BEGIN GENERATED: m2c-onepass-minus-offline-b140-high -->
-0.309288
<!-- END GENERATED: m2c-onepass-minus-offline-b140-high -->
  ];
- B142:
<!-- BEGIN GENERATED: m2c-onepass-minus-offline-b142 -->
-0.187693
<!-- END GENERATED: m2c-onepass-minus-offline-b142 -->
  with 95% CI
  [
<!-- BEGIN GENERATED: m2c-onepass-minus-offline-b142-low -->
-0.281567
<!-- END GENERATED: m2c-onepass-minus-offline-b142-low -->
  ,
<!-- BEGIN GENERATED: m2c-onepass-minus-offline-b142-high -->
-0.101138
<!-- END GENERATED: m2c-onepass-minus-offline-b142-high -->
  ].

The earlier M2 numbers depended on iteration: the two-pass arm is worse in both
budgets. The difference is negative (one-pass is better) and excludes zero. The
plausible reading is that the first mask pass changes the states the second pass
selects from, and the second pass is not self-consistent with its own execution;
the one-pass arm decides from a state it will actually execute. This is one
eight-window sample and one teacher, so it is a measured comparison, not a
general claim.

Controller cost: the one-pass controller has zero learned parameters; the
controller is the hand-written norm/top-k rule. Its parameter reads are the same
dense gate/up reads that make the arm non-compliant. The timing fields are
separated so two conventions cannot be confused: for the online arm the selector
hooks run inside the timed forward and synchronise the device, so the recorded
forward time already contains the selector; for the offline arm the selector
runs outside the timed forward, so the two add. At B140 over the eight-window
sample on the RTX 4080 SUPER the online arm spends
<!-- BEGIN GENERATED: m2c-onepass-b140-seconds-2 -->
0.739
<!-- END GENERATED: m2c-onepass-b140-seconds-2 -->
seconds inside the selection hooks and
<!-- BEGIN GENERATED: m2c-onepass-b140-total-seconds -->
1.242
<!-- END GENERATED: m2c-onepass-b140-total-seconds -->
seconds in the whole forward. The remainder after subtracting the hook time is
<!-- BEGIN GENERATED: m2c-onepass-b140-execution-seconds -->
0.502
<!-- END GENERATED: m2c-onepass-b140-execution-seconds -->
seconds, but it is **not** a clean execution measurement: the per-layer
device synchronisation inside the hook also waits for forward work, so the
subtraction can absorb it. The defensible online quantity is the total. The
offline arm spends
<!-- BEGIN GENERATED: m2c-offline-b140-execution-seconds -->
1.240
<!-- END GENERATED: m2c-offline-b140-execution-seconds -->
seconds in execution over its two forwards and
<!-- BEGIN GENERATED: m2c-offline-b140-total-seconds -->
1.307
<!-- END GENERATED: m2c-offline-b140-total-seconds -->
seconds in total, with its selector time outside the timed forward. The online
selector time is conservative because it includes that synchronisation; the
offline selector is measured on already-captured states in a batch. The random
baseline is not a controller-cost claim: its selector is a naive scalar splitmix
generator and reads no learned parameters (its measured time is
<!-- BEGIN GENERATED: m2c-random-b140-selector-seconds -->
6.320
<!-- END GENERATED: m2c-random-b140-selector-seconds -->
seconds), so its physical failure is the dense masked execution, not a selector
read outside the active set. Its corrected receipt records that reason
explicitly instead of copying the magnitude arm's reason.

## 5. Tensor-access audit of the sparse path

A genuinely sparse path was implemented in `masked_mlp.sparse_grouped_forward`:
it groups tokens by identical subset, slices gate/up rows and down columns with
`index_select` before every matmul, and never computes an unselected channel.
Court B checks it against the masked-dense path with `atol=rtol=1e-5` on
synthetic layers, including rows that select nothing and rows that select every
group, and against a float64 hand computation. On the full split it reproduces
the masked-dense KL of every evaluated arm to the printed precision: for
`token-magnitude` the dense KL is
<!-- BEGIN GENERATED: m2c-token-magnitude-dense-kl -->
6.676772
<!-- END GENERATED: m2c-token-magnitude-dense-kl -->
and the sparse KL is
<!-- BEGIN GENERATED: m2c-token-magnitude-sparse-kl -->
6.676772
<!-- END GENERATED: m2c-token-magnitude-sparse-kl -->
and the full JSON records the per-window equality.

The by-construction read count is cross-checked with
`torch.profiler(record_shapes=True)`: every `aten::mm` contributes `k*n`
weight elements and `2*m*n*k` FLOPs. On real Qwen3-1.7B weights, eight tokens
through five layers, all five layers matched exactly:

- layer 0 counted:
  <!-- BEGIN GENERATED: m2c-audit-layer0 -->
94371840
  <!-- END GENERATED: m2c-audit-layer0 -->
- layer 0 profiled:
  <!-- BEGIN GENERATED: m2c-audit-layer0-profiled -->
94371840
  <!-- END GENERATED: m2c-audit-layer0-profiled -->
- all audited layers: `matches: true` in
  `results/selectors/sparse-audit-20261115T002900Z.json`.

That file is the audit receipt; the per-layer rows for layers 0, 7, 14, 21 and
27 are printed there with counted elements, profiled elements, counted FLOPs and
profiled FLOPs.

## 6. Residual gaps

- The six compliant arms are static or random masks; there is no compliant
  learned selector in this family at this budget. A compliant controller would
  need a scoring projection small enough to fit the B142 headroom; no such
  controller was trained here.
- The one-pass comparison uses the oracle sample (eight windows), because the
  full split would multiply the controller cost measurement by the number of
  windows. It is one seed and one teacher.
- `token-task-static-nn` was found to read whole-window shingles, including
  future tokens. It is now labelled `future_tokens`; the old prose that called it
  a deployable per-token arm is not retained as a claim.
- The M2 selection traces come from a degraded, untrained-router setting: the
  selected subsets are the output of a post-hoc statistic on an unmodified dense
  checkpoint, not a trained router. Their instability (and the one-pass
  difference) should not be extrapolated to a trained router.
- Exact neural execution remains future work: the type-level witness exists, but
  no end-to-end exact-mode runtime consumes it yet (Court E parity is still
  future work).

## Receipts

- Corrected full evaluation:
  `results/selectors/eval-full-20261115T005000Z.json`
- Analysis and corrected table: `results/selectors/selectors.json`,
  `results/selectors/table.md`
- One-pass comparison: `results/selectors/onepass-20261115T044200Z.json`
  (supersedes `onepass-20261115T012700Z.json`, whose random-arm failing reason
  contradicted its zero selector reads and whose `seconds` field mixed two
  timing conventions; the KL values are unchanged and reproduce exactly)
- Sparse-read audit: `results/selectors/sparse-audit-20261115T002900Z.json`
- Rust parity/accounting: `results/selectors/atlas-compare.json`
- Court receipts: `results/courts/` (oracle, rust, io, commit_identity,
  perf_numbers)
