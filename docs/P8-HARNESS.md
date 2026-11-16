# P8 harness — fresh tasks, three selectors, and bounded composition

Status: **harness implemented and CPU-tested; nothing has been measured on a
real arm.** The GPU belongs to the running T2 pilot, so this session builds and
checks the instrument and records the exact launch commands. Three receipts back what
is here: the contamination receipt `results/p8/contamination-20261116T000311Z.json`,
the selector-harness self-test `results/p8/selector-harness-selftest-20261116T000332Z.json`
and the composition admission `results/p8/composition-admission-20261116T000322Z.json`,
all at revision `8baa139c42ef3538e777ff0220f3ceb424014ab9`.

This is the preparation for `docs/PROGRAM-PLAN.md` P8.1 (task-level `D`/`M`
selection on fresh held-out tasks) and P8.2 (compositional residuals with a
bounded learned gate). It changes no registered arm, threshold or verdict.

## 1. The fresh task set

`oracle/eval/fresh_tasks.py` authors every item here; no item text, answer or
test is copied from an evaluation source. Items by domain (generated from the
contamination receipt):

<!-- BEGIN GENERATED: p8-task-items -->
26
<!-- END GENERATED: p8-task-items -->

- rust:
<!-- BEGIN GENERATED: p8-task-rust -->
10
<!-- END GENERATED: p8-task-rust -->
- code:
<!-- BEGIN GENERATED: p8-task-code -->
6
<!-- END GENERATED: p8-task-code -->
- math:
<!-- BEGIN GENERATED: p8-task-math -->
5
<!-- END GENERATED: p8-task-math -->
- factual:
<!-- BEGIN GENERATED: p8-task-factual -->
5
<!-- END GENERATED: p8-task-factual -->

Answer kinds, and what "checkable" means for each:

| kind | check | limitation |
| --- | --- | --- |
| `diagnostic` (rust) | the expected `rustc` error code(s) must appear and every required phrase must occur | textual; compiling the snippets with `rustc` in the rust court is the next strengthening, and the receipt says so |
| `code` | the answer is appended to the prompt and run against embedded `assert`s in the same sandbox limits the registered HumanEval harness uses | one file, no imports outside the stdlib |
| `numeric` | the last number in the answer, normalised, must equal the gold value | — |
| `choice` | the last `A`–`D` letter must equal the gold letter | — |

The Rust items cover lifetime and borrow errors (`E0515`, `E0502`, `E0499`,
`E0597`, `E0507`), non-exhaustive matching (`E0004`), `Send` violations in
threads and across `await` (`E0277`), a trait object without `dyn` (`E0782`), and
debug-build overflow behaviour.

## 2. The contamination receipt

Disjointness is checked, not asserted. Every fresh item is hashed and shingled
and compared against every registered evaluation source, in two units:

* **8-word shingles** over lower-cased, whitespace-collapsed text against
  HumanEval (prompt, canonical solution and tests), GSM8K (question and answer)
  and TruthfulQA (question and every choice);
* **13-token shingles** between the tokenised item text and the pinned held-out
  windows that the KL metric uses — the same tokenizer, so the comparison is in
  the model's own units.

Sources compared against, from the receipt:

- HumanEval items:
<!-- BEGIN GENERATED: p8-contam-humaneval -->
164
<!-- END GENERATED: p8-contam-humaneval -->
- GSM8K items:
<!-- BEGIN GENERATED: p8-contam-gsm8k -->
1319
<!-- END GENERATED: p8-contam-gsm8k -->
- TruthfulQA items:
<!-- BEGIN GENERATED: p8-contam-truthfulqa -->
817
<!-- END GENERATED: p8-contam-truthfulqa -->
- held-out windows:
<!-- BEGIN GENERATED: p8-contam-windows -->
160
<!-- END GENERATED: p8-contam-windows -->
  covering
<!-- BEGIN GENERATED: p8-contam-tokens -->
47468
<!-- END GENERATED: p8-contam-tokens -->
  tokens

Verdict, as the receipt records it:

<!-- BEGIN GENERATED: p8-contam-verdict -->
disjoint
<!-- END GENERATED: p8-contam-verdict -->

The check fails closed: `--require-disjoint` (the default) exits non-zero on any
overlap, so a run with a contaminated set cannot be registered. Regenerate the
receipt before registering P8.1 — the set is small and hand-authored, and a later
edit must re-pass the same gate.

## 3. The three selectors

`oracle/eval/complementarity.py`. The metric is the per-item loss of the
*selected* capability (lower is better); every comparison is paired on the same
items with a percentile paired bootstrap, seed fixed. The router is trained on
one half and **scored only on the other half**, so no selector is scored on an
item it was fitted to.

1. **`fixed-rule`** — pre-registered, no learned parameters, no outcome:
   pick the first capability when the prefix's code-marker or Rust-keyword score
   reaches the threshold fixed in the module, otherwise the second. It is not
   tuned on the data; changing the threshold is a new registration.
2. **`trained-router`** — a logistic head on standardised task features (prefix
   statistics, plus a cheap hidden-state summary when the caller supplies one:
   norm, mean absolute, last-token norm). Labels are the measured per-item
   outcome on the training half. It reads no gold answer at decision time, and
   its learned-parameter count is reported per decision.
3. **`oracle`** — the per-item better capability. It needs the outcome, so it is
   recorded as `deployable: false`; it is the bound that separates "the selector
   is bad" from "the capabilities are not complementary".

Each row reports: mean loss, gap to the per-item oracle, selection accuracy, pick
counts, per-domain means, the paired differences against `always-D` and
`always-M` with their intervals, and the selector's own cost. The cost is
reported separately from the capabilities' cost so a selector win cannot be
bought with an unaccounted read.

Capability losses come from a language-model instrument, not from a rubric:
every item carries a **reference continuation** (its gold answer text or code
body) and `completion_loss` scores the mean NLL of that continuation under an
arm, given the prompt. Both capabilities are scored on the same text, so the
paired comparison is exact. The instrument is covered by the contamination check
because the reference is part of the item's text.

The harness is capability-agnostic. `--mode self-test` uses two synthetic
complementary capabilities whose answer is known by construction, and the test
suite asserts the orderings that follow from it. `--mode checkpoint` loads the
trained `D` and `M` arms, scores the fresh set, and feeds the measured losses
through the same selector path; it refuses to run without the GPU court. Both
paths and the instrument are CPU-tested — the checkpoint path with stub models,
because the frozen 0.6B core does not fit the CPU court's memory limit.

## 4. The bounded-gate composition (P8.2)

`oracle/train/composition.py` implements

```
y = S + a r_D + b r_M,   (a, b) = g(feature),   ||a r_D + b r_M||_2 <= rho
```

with the gate's weights zero-initialised (so the composition is exactly `S` at
step 0, matching the zero point every registered arm satisfies) and the realised
perturbation projected onto an L2 ball of radius `rho` per token. `interference_report`
separates three quantities, because "adding `M` changed the output" can mean
three things and only one of them is bounded:

* `accounted` — the composed output equals `S + a r_D + b r_M` for the gate's own
  coefficients, recomputed from the residuals. If this is false the difference is
  not the gate's doing and no interference claim is valid.
* `added_effect_of_m` — `||b r_M||`, what `M` adds at the chosen coefficients.
* `deviation_from_d_alone` — `||(a-1) r_D + b r_M||`, the plain-sense
  interference, which is **not** bounded by `rho` because the gate may also turn
  `D` down. Reporting both is what keeps the bound honest.
* `within_budget` — the perturbation norm against `rho`. This is the declared
  bound, and it is the only one that is guaranteed by construction.

### The fresh union, through the Rust ledger

`scripts/p8-composition-admission.sh` emits the composed config in the Python
court, admits it in the Rust court, and wraps the record with a stamp and
revision. The ledger unions the two arms' execution slices **by identity**, which
is why the union is not a sum: both arms re-declare the frozen core.

- active unique parameters, admitted:
<!-- BEGIN GENERATED: p8-compose-active -->
774371384
<!-- END GENERATED: p8-compose-active -->
- of which the gate (declared as a selector, since it decides coefficients
  rather than carrying capacity):
<!-- BEGIN GENERATED: p8-compose-selector -->
57400
<!-- END GENERATED: p8-compose-selector -->
- execution union:
<!-- BEGIN GENERATED: p8-compose-execution -->
774313984
<!-- END GENERATED: p8-compose-execution -->
- total *accessible* (above the ceiling on purpose — the ceiling is a per-token
  execution budget, not a storage budget, exactly as `H` shows):
<!-- BEGIN GENERATED: p8-compose-accessible -->
1042804792
<!-- END GENERATED: p8-compose-accessible -->

The Python mirror prints the same active union before admission and the wrapper
records `mirror_agrees`, so a disagreement between the ledger and the mirror is a
failing receipt rather than a footnote.

## 5. What is established, and what is not

Established (CPU, receipts above):

* the task set is disjoint from every evaluation source in both units, with the
  counts and the hashes in the receipt;
* the three selectors and the paired bootstrap run, are deterministic, and are
  tested against synthetic capabilities whose answer is arithmetic;
* the composition's zero point is exact, its perturbation never exceeds the
  declared budget, its interference is accounted term by term, and its union is
  admitted by the Rust ledger under the ceiling.

Not established, and deliberately not claimed:

* **no arm has been evaluated on the fresh set.** Every number the harness
  reports today describes synthetic fixtures.
* **the Rust answer check is textual.** It does not compile the snippets.
* **the tasks are few.** One seed, one set, no power analysis; a
  complementarity difference smaller than the task set can resolve will not be
  detectable, and the set is not a benchmark.
* **the gate is untrained.** Its zero point and its bound are tested; nothing
  says a trained gate is useful.

## 6. Exact commands for when the GPU is free

Run these after the pilot publishes `results/T2-PILOT.md` and after P12.1a's
evaluation-only step (they load the trained `D` and `M` checkpoints).

```bash
# 0. re-clear the task set (cheap, CPU) and keep the receipt
scripts/court oracle -- python3 oracle/eval/fresh_contamination.py \
    --out results/p8/contamination-$(date -u +%Y%m%dT%H%M%SZ).json

# 1. the three selectors over the trained capabilities (GPU)
scripts/court --gpu oracle -- python3 oracle/eval/complementarity.py \
    --mode checkpoint \
    --arm-d results/artifacts/train/t2/D/step-001024.pt \
    --arm-m results/artifacts/train/t2/M/step-001024.pt \
    --config-d results/train/admission-configs/t2/D.json \
    --config-m results/train/admission-configs/t2/M.json \
    --admission-d results/train/admission/t2/D.json \
    --admission-m results/train/admission/t2/M.json \
    --out results/p8/selectors-$(date -u +%Y%m%dT%H%M%SZ).json

# 2. the four composition references (GPU), after the gate is trained
scripts/court --gpu oracle -- python3 oracle/train/composition.py \
    --run-composition --budget 0.5 \
    --config results/train/admission-configs/p8/composition.json \
    --admission results/train/admission/p8/composition.json \
    --out results/p8/composition-$(date -u +%Y%m%dT%H%M%SZ).json

# 3. the union, re-admitted at the current revision
scripts/p8-composition-admission.sh
```

`--mode checkpoint` exists and is the command in step 1. It has not been run:
the GPU belongs to the pilot, and the frozen core does not fit the CPU court, so
its checkpoint loader is exercised only by the stub-model tests.

`--run-composition` in step 2 does **not** exist yet. A composed arm needs a
trained gate, which needs a training run, which needs the GPU; until then the
composition is implemented, bounded, admitted and tested, and its four
references are the four capabilities the selector harness already accepts. The
command above is the interface that step will take, recorded now so the step is
not redesigned after seeing a result.
