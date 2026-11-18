# S52 repair — the P19.9 screen's comparisons, ledger and statistics

Status: **repaired and re-registered before any screen run**. The screen itself
is still **not run**; nothing in this page is an accuracy or byte measurement.

The S52 registration (`docs/s52-p19.9-router-screen-registration-v1.json`) is
retained for provenance. Its pre-run defects and their corrections are recorded
in `docs/s52-p19.9-router-screen-registration-v2.json`. The v2 file is the
authority for the screen.

## 1. The two controls are not interchangeable

The v1 driver scored the `gated-dense` control and then copied the memory arm's
read counts onto it:

```text
summary["gated-dense"]["read_requests"] = summary["memory"]["read_requests"]
summary["gated-dense"]["read_bytes"]    = summary["memory"]["read_bytes"]
```

That made a control with no external store look like a memory reader and let the
win bar compare a routed arm's bytes against a number that belonged to a
different arm.

The v2 separation:

- **accuracy** is compared against `gated-dense`, the same zero-initialised gate
  over a dense read. It has no external store; its `read_requests` and
  `read_bytes` are `null`, and the receipt says why.
- **bytes** are compared against `memory`, the dense memory condition in which
  every token reads the table and no router is used. Its own gather ledger is
  the always-read byte count.
- if the memory control is absent, `fewer_bytes_than_dense` is `null` and no arm
  can win on bytes.

`oracle/frontier/routing_runner.py:evaluate_arms` now reports
`accuracy_control`, `byte_control` and `gated_dense_read_claim` explicitly, and
the driver no longer assigns any read number to `gated-dense`.

## 2. The ledger reports addresses and a cache model

`StoreLedger` (in `oracle/frontier/routing_runner.py`) now records every gather
call and every generation:

- per call: requested, unique, repeated, cumulative unique and cache hits;
- per generation: the same totals, grouped by a generation marker the driver
  sets before each sample;
- per run: requested, unique, repeated and the table's own row size.

The row size is derived from the table, not typed: `table.element_size() *
table.shape[1]`. A table that changes row size mid-run fails closed.

The byte bases are explicit:

- `bytes` is the **request-based physical count** under the stated cache model.
  With the default `cache_model="none"`, a repeated address is a repeated
  fetch.
- `unique_bytes_modelled` is the globally deduplicated row count times the row
  size and is labelled `modelled`. It is never a physical-byte claim.
- `bytes_from_source` and `bytes_from_cache` are reported separately once a
  cache model is declared.

`oracle/frontier/utility_routing.py` carries the same correction into
`ReadAccounting`: `addresses_supplied`, `cache_model`, `bytes_physical`,
`bytes_unique_modelled` and the one-row-per-token fallback are separate fields.
`cost_from_accounting` uses the physical count and labels the selection audit
`modelled` because that audit does not supply row addresses. The S50 fields are
kept unchanged for overlap with earlier receipts; they were never a physical
byte claim.

## 3. The statistics unit is the problem

`prompt_paired_bootstrap` resamples distinct problems, not continuations. The
driver now emits one row per holdout problem with that problem's fraction
correct for every arm, and `evaluate_arms` requires the prompt-level paired 95%
interval to exclude zero in addition to the registered combined standard error.
The number of distinct problems is reported beside every interval.

The holdout is expanded from the v1 eight problems to **24 distinct problems**
in `docs/s52-p19.9-calibration-prompts-v2.json`. The same 64 committed prompts
are re-graded by file order (fit 32 / calib 8 / holdout 24); no question,
answer, id or sampling parameter changed. The prompt set remains disjoint from
GSM8K test and MATH-500. The screen is labelled
**mechanism-validation only** in the registration, the eval receipt and the
screen receipt; a 64-prompt set is not a capability claim.

## 4. What the screen can and cannot say

It can say whether the pre-read route beats base and the gate control on a small
problem set and whether it reads fewer physical bytes than the always-read
memory. It cannot support a capability claim, a 20M route registration, or a
nats-per-byte price. The registered decision rules are unchanged.

## 5. Stage 3's FLOPs clause is a guard, not the evidence

The stage-3 registration's `R=4` greedy FLOPs bound (≤ 0.30 of stage-2 maj@16)
is close to automatic when the comparison is four greedy passes against sixteen
sampled answers: a ratio near 0.125 is arithmetic, not an experimental result.
`docs/ws19-post-trained-route-stage3-amendment-s54-v1.json` records that the
accuracy clause (closing at least two thirds of the stage-2 greedy-to-maj@16 gap
on both tasks) is the evidence, and the FLOPs bound is retained as a guard. The
executed-set FLOPs are still reported; satisfying the guard is not a win.

## 6. Tests

The S52 runner tests, the S50 pre-read mask tests and the S53 optimiser
comparison tests pass in the oracle court for this revision. The pinned claims
include per-call/per-generation ledger rows, the table-derived row size, the
modelled unique-byte label, the accuracy/byte control separation, a deliberately
wrong gated-dense read count that is never used, the prompt-level bootstrap, and
the refusal path when no dense memory control exists. The court receipts under
`results/courts/` name the revision and the exact command.
