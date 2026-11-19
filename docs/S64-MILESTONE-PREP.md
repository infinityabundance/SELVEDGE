# S64 — milestone scoring preparation (0.5M / 1M / 2M / 5M)

Status: **prepared, not run.** No corrected-training checkpoint exists yet. This
file records the instrument, the cadence and the selection discipline so the
scoring cannot drift once the arm trains.

Registration: `docs/ws19-stage2-correction-s63-preregistration-v1.json`
(corrected stage 2, cadence and win bar inherited from
`docs/ws19-post-trained-route-stage2-revised-preregistration-s60-v1.json`).

## Instrument

The collector is `scripts/s62_score_milestones.py`. It does not invent a
number; every figure in its summary comes from a receipt a command wrote.

* `--plan` prints one court command per milestone and task and records whether
  the milestone checkpoint exists and whether its step is on the run's own
  checkpoint grid. A missing checkpoint is printed as `MISSING` and must not be
  run.
* `--collect` reads those receipts and emits, per milestone:
  * GSM8K and MATH-500 flexible accuracy (greedy, non-thinking, zero-shot,
    `\boxed{}`, the S54 instrument);
  * the paired problem-level bootstrap delta against the post-trained 0.8B and
    its 95% interval (the unit is the problem, 4,000 resamples);
  * the termination fractions — truncation rate, loop-classified fraction and
    the cross-tab — from the same stage-1 receipt;
  * the MMLU / MMLU-Pro retention deltas and their paired intervals;
  * the registered win decision.
* `--self-test` exercises the summary, retention verdicts, win logic and
  milestone parsing on synthetic receipts. Its court receipt is
  `results/courts/oracle-20261119T052011Z-be4462.json`.

## Cadence and command template

The four cadence points are **processed tokens**, not wall-clock steps. The
launcher writes one merged full-weight checkpoint at each point and prints the
actual step numbers; the plan must use those numbers, not a guessed grid:

```sh
# steps come from the launcher receipt / its printed milestone line
SELVEDGE_MEM=4g scripts/court oracle -- python3 scripts/s62_score_milestones.py \
  --plan \
  --run-dir results/stage2/runs/s64-preference \
  --milestones 0.5M=<step>,1M=<step>,2M=<step>,5M=<step> \
  --out-dir results/stage2/s64/score

# after the plan's stage-1, retention and pairing commands have run serially,
# one GPU job at a time:
SELVEDGE_MEM=4g scripts/court oracle -- python3 scripts/s62_score_milestones.py \
  --collect \
  --run-dir results/stage2/runs/s64-preference \
  --milestones 0.5M=<step>,1M=<step>,2M=<step>,5M=<step> \
  --out-dir results/stage2/s64/score \
  --out results/stage2/s64/score/summary.json
```

The launcher's receipt records `checkpoints[].cadence_tokens`,
`checkpoints[].step` and `checkpoints[].processed_tokens`, so the mapping is
read, never typed.

## Selection discipline (normative)

1. **No test split is a selection input.** The interpolation coefficient is
   chosen on GSM8K-cal / MATH-cal (train split only) under the retention bound;
   the preference and control arms are compared on the registered win bar and
   its paired intervals. GSM8K test and MATH-500 test are evaluation
   instruments, run after a checkpoint is fixed.
2. **The milestone summary is not a selection rule.** It reports the registered
   win decision at the cadence points. Choosing a checkpoint because it scored
   best on a test split would be selection on the test set and is not done.
3. **An unresolved retention clause blocks a win; it is not a retention
   failure.** The collector already distinguishes the two.
4. **Termination and loop fractions are reported beside accuracy**, because a
   capability gain that is bought by more truncation or more looping is not the
   registered result.

## What this file does not establish

Nothing here is a measurement. It records the instrument, the cadence, the
command shape and the rule that selection never reads a test split.
