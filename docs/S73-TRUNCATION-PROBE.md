# S73 — the truncation ceiling (cap-ladder probe)

Status: **registered, not run.** The registration is
`docs/s73-truncation-ceiling-preregistration-v1.json`. It registers a
diagnostic cap ladder on **MATH-500 only**. It authorises no training, no trace
generation, no new claim protocol and no new split. Execution is blocked on the
GPU driver (§10).

This is the human-readable companion to that JSON. A measured number names the
receipt, stamp and revision that produced it; an arithmetic number names its
inputs and formula; a number that was not measured or derived is **unknown**.
The classification numbers in §2 come from
`results/stage2/s73-truncation/*.json` (revision
`d1283db96b26fe8d7d4d732187ba08d0e5bf7870`) and are labelled **pre-run
estimate** throughout: they describe stored 4,096 completions, not a result of
this probe.

## 1. The finding this turns into an experiment

All rows are our S54 stage-1 protocol — greedy, non-thinking, zero-shot,
`max_new_tokens = 4096`, flexible basis, strict and exact beside it — replayed
from stored completions by `results/stage2/s72-recon/teacher-recon.json`
(sha256 `a7bb2812…`, stamp `20261122T130947Z`). No model was reloaded.

| model | MATH-500 flexible | strict | exact | truncated at 4,096 | receipt | stamp |
|---|---|---|---|---|---|---|
| S67 soup | **203/500 = 40.60%** | 40.60% | 40.60% | **158/500 = 31.6%** | `results/stage2/s67-final/soup-stage1.json` | `20261120T225912Z` |
| post-trained Qwen3.5-0.8B | 191/500 = 38.20% | 38.20% | 40.60% | 160/500 = 32.0% | `results/stage1/s54-stage1-post-0.8b-nonthinking-greedy.json` | `20261118T033841Z` |
| Qwen/Qwen3-0.6B | **244/500 = 48.80%** | 48.80% | 52.00% | **18/500 = 3.6%** | `results/stage2/s68-frontier/qwen3-0.6b.json` | `20261121T192906Z` |
| Qwen/Qwen3.5-2B (teacher) | 329/500 = 65.80% | 65.80% | 70.20% | 115/500 = 23.0% | `results/stage1/s54-stage1-post-2b-nonthinking-greedy.json` | `20261118T045605Z` |

The gap is 244 − 203 = 41 problems on the identical 500 problems, 48.80% −
40.60% = 8.20 points. The leader truncates one ninth as often as we do, has
flexible == strict == 48.80 with zero non-boxed fallbacks, and its exact number
(52.00% at `math_verify 0.9.0`) is higher still.

The hypothesis under test has two parts, and they are not the same claim:

* **H-a, the cap converts would-be-correct answers into failures.** A truncated
  completion already contained the right answer, and the cap turned a correct
  answer into a failure.
* **H-b, the model does not terminate.** The model's learned length/stop
  behaviour, not its reasoning, is the deficit.

H-a is the stronger and more useful claim, because it would make a longer cap
(or an extraction at the answer) recover problems at no reasoning cost.

## 2. What the stored completions already say (pre-run estimate)

`oracle/frontier/truncation.py` (sha256 `f2b1c07c…`, version
`s73-truncation-v1`) partitions each stored completion exclusively into
`loop_degenerate` > `no_box` > `boxed_then_continued` > `other` (the declared
precedence; a looping row that also carries a box keeps `boxed_present` true and
is reported in the overlap counts). It was run through the `oracle` court over
the stored 4,096 completions of all four models. Receipts:
`results/stage2/s73-truncation/soup-math500.json` (sha256 `fe57385d…`),
`…/baseline-08b-math500.json` (`2f93a491…`), `…/qwen3-06b-math500.json`
(`0d9980ae…`), `…/teacher-2b-math500.json` (`64baa08d…`).

For the 158 soup rows truncated at 4,096, rows in the pool:
<!-- BEGIN GENERATED: s73-soup-rows-truncated -->
158
<!-- END GENERATED: s73-soup-rows-truncated -->

`loop_degenerate`:
<!-- BEGIN GENERATED: s73-soup-class-loop-degenerate -->
42
<!-- END GENERATED: s73-soup-class-loop-degenerate -->

`no_box`:
<!-- BEGIN GENERATED: s73-soup-class-no-box -->
116
<!-- END GENERATED: s73-soup-class-no-box -->

`boxed_then_continued`:
<!-- BEGIN GENERATED: s73-soup-class-boxed-then-continued -->
0
<!-- END GENERATED: s73-soup-class-boxed-then-continued -->

`other`:
<!-- BEGIN GENERATED: s73-soup-class-other -->
0
<!-- END GENERATED: s73-soup-class-other -->

rows that already contain a correct boxed answer anywhere before the cap:
<!-- BEGIN GENERATED: s73-soup-truncated-correct-box-anywhere -->
0
<!-- END GENERATED: s73-soup-truncated-correct-box-anywhere -->

The same measurement for every model (counts from the same four receipts):

| model | truncated | loop | no box | box then continued | correct box anywhere |
|---|---|---|---|---|---|
| S67 soup | 158 | 42 | 116 | 0 | **0** |
| post-trained 0.8B | 160 | 42 | 115 | 3 | **0** |
| Qwen3-0.6B | 18 | 6 | 11 | 1 | **0** |
| Qwen3.5-2B | 115 | 16 | 96 | 3 | **4** |

Four consistency checks pass, which is why the estimator is trusted as a
description of the stored data: the exclusive classes sum to each truncated
count; `all_rows.flexible_correct_rows` equals the stored receipt's flexible
correct count for every model (203, 191, 244, 329); the soup's boxed rows (343)
and Qwen3-0.6B's (484) equal the recon's last-boxed method counts; and the soup's
157 rows with no box at all match the recon's flexible method `none` of 157.

**What this means before a single new token is generated.** Not one of the
soup's 158 truncated rows contains a correct answer, and only one contains a box
at all. So H-a, read as "the 4,096 prefix already held the right answer and
extraction threw it away", is **false at 4,096 by construction**. The dominant
fault in the soup's truncations is that no answer is emitted (116 rows, 73.4%),
with degenerate loops second (42 rows, 26.6%); "answered and then did not stop"
is zero rows. The leader's 18 truncated rows and the baseline's 160 behave the
same way (zero correct boxes). The teacher is the exception: 4 of its 115
truncated rows already contain a correct box and are already scored correct, so
part of the teacher's truncation is a last-box/extraction effect rather than a
length effect.

This does not falsify H-b. It sharpens it: the soup's deficit at 4,096 is
"does not reach an answer", not "will not stop after an answer". A higher cap can
therefore only help through tokens 4,096 and later, and that is exactly what the
ladder measures.

## 3. What is registered

**Caps:** 4,096 (existing; the registered primary), 8,192 and 16,384. All four
models already have stored 4,096 MATH-500 completions, so the 4,096 rung is a
replay. The ladder adds two generations per model.

**Models:** S67 soup; post-trained Qwen3.5-0.8B (weights snapshot
`2fc06364715b967f1860aea9cf38778875588b17`); Qwen/Qwen3-0.6B (`c1899de2…`);
Qwen/Qwen3.5-2B (`15852e8c…`).

**Instrument, frozen:** identical zero-shot user template
(`{question}\n\nSolve the problem step by step. Put the final answer in
\boxed{}.`), no system message, greedy `temperature = 0.0` / `top_p = 1.0`,
non-thinking, EOS-only stop, the S54 extractor. Only `max_new_tokens` changes.

**Validity check:** for every model and cap, the first 4,096 generated tokens
must reproduce the stored 4,096 completion token-for-token. Greedy decoding is
deterministic only in exact arithmetic, so a reimplemented or batched engine can
drift; a rung whose prefix differs is receipted as void for that model and is not
quoted, because then the difference is not attributable to the cap.

## 4. The statistics, fixed before the run

* **Primary statistic (unchanged protocol):** flexible accuracy on the 500
  MATH-500 problems at `max_new_tokens = 4096`. The soup's value stays 203/500 =
  40.60% (`soup-stage1.json`, stamp `20261120T225912Z`); the leader's stays
  244/500 = 48.80% (`qwen3-0.6b.json`, stamp `20261121T192906Z`). **The
  registered primary protocol stays 4,096.**
* **Diagnostic statistic:** `recovered_flexible_correct`, the number of the 158
  soup rows that were not flexible-correct at 4,096 and are flexible-correct at
  the diagnostic cap; strict is reported beside it (on this instrument soup
  flexible == strict, so any divergence is itself a finding).
* **Per-model delta:** for each compared model, flexible-correct at cap *c* minus
  flexible-correct at 4,096, on the identical 500 problems.
* The 8,192 and 16,384 generations are a **diagnostic instrument**, not a new
  claim protocol. No statement of the form "the soup scores X at 16,384" may be
  made as a capability claim.

## 5. The classifier

`oracle/frontier/truncation.py`, sha256
`f2b1c07cbd65125d3e16882a8adf9e74b47e2d7456fcf3067513f33d058e19f9`, version
`s73-truncation-v1`, `BOX_THEN_CONTINUED_MIN_NONSPACE = 20`. It reuses the
registered extraction (`frontier.rft._boxed_answers`, `answers_equal`, `extract`)
and the S56 loop detector (`frontier.termination.classify`, calibrated in
`results/stage1/s56-termination-calibration.json`); it does not fork either. The
strict wrapper mirrors `extract_strict` in `scripts/s54_post_trained_eval.py`,
and a test pins the two to agreement on a synthetic battery so the mirror cannot
drift. The hash is pinned by test to the registration; changing the module, the
threshold or the class order requires a new registration.

Per-row output also carries `boxed_present`, `boxed_count`,
`tail_after_last_box_nonspace`, the loop verdict and reason, both extraction
verdicts, and `correct_box_anywhere`.

## 6. Falsification: X = 22

The pre-declared condition is:

> If fewer than **22** of the 158 soup rows truncated at 4,096 become
> flexible-correct at 16,384, termination is not the binding constraint, and the
> next experiment returns to the teacher/data route. C12 is not opened.

Arithmetic, all from measured inputs:

* gap = 244 − 203 = **41 problems** on the identical 500.
* noise floor = 1.96 × √(0.406 × 0.594 / 500) × 500 = 21.52 rows → **22**.
* half the gap = ⌈41 / 2⌉ = **21** rows.
* **X = max(22, 21) = 22 rows**, which is 22/158 = 13.9% of the truncated pool.
  The pool is 7.18× X, so the threshold is reachable and not trivially met.
* For scale: full-gap recovery is 41/158 = 26.0%, giving 203 + 41 = 244/500 =
  48.80%, an exact tie with Qwen3-0.6B. Recovering X = 22 gives 225/500 = 45.00%,
  still 19 problems below the leader.

The threshold is the stricter of the instrument's own 95% noise floor and half
the measured gap; it is fixed before the ladder runs, and passing it still does
not produce a win.

## 7. Decision rule (fixed before any score)

1. **Validity first.** A rung whose 4,096-token prefix does not reproduce the
   stored completion is void for that model; receipt and re-run. Do not quote it.
2. **Recovery < 22.** Termination is not the binding constraint. Return to the
   teacher/data route (C9/C10 or a stronger teacher). C12 stays closed.
3. **22 ≤ recovery ≤ 40.** Termination is a material but not sole constraint
   (it closes at least half the gap, not all of it). Open C12 as the primary
   target; keep the teacher/data route open as a separate, non-competing item.
4. **Recovery ≥ 41.** Termination explains the observed 4,096 gap to
   Qwen3-0.6B. C12 is the primary target.
5. **The leader gains evenly.** If the soup's gain at the diagnostic cap minus
   Qwen3-0.6B's gain at the same cap is ≤ 0, record "the cap helps both, no
   gain, no claim". The gap is not a termination artefact, whatever the absolute
   recovery, and termination is not the target.
6. **The leader gains far less.** If the differential is ≥ 22 problems (X), the
   cap helps us far more than the leader: termination is the target and C12
   opens even when the absolute recovery is below 22, because the differential,
   not the absolute, is the evidence of a termination-specific deficit.
7. **A rung affordable for some models but not others.** Do not compare across
   caps. Report only the caps at which every compared model ran, and name the
   missing rung as unaffordable with its arithmetic.
8. **No post-hoc retuning.** A changed cap, threshold, class definition or
   statistic is a new dated registration, never an edit.

## 8. Fairness rule

Any future claim at a higher cap must be evaluated at that cap for **every**
compared model, on the identical problems, with the same prompt, decoding and
extractor, and must state the cap. A 16,384 row for one model beside a 4,096 row
for another is not a comparison. A model added later must be run at every cap the
comparison quotes.

## 9. Cost, memory and disk

**Throughput inputs** (per-task `tokens_per_second` recorded in the 4,096
receipts): soup 726.05, post-trained 0.8B 703.17, Qwen3-0.6B 194.41,
Qwen3.5-2B 528.79. Receipts and stamps as in §1.

**Token arithmetic.** For each model, keep the finished rows at their measured
length and run every truncated row to the new cap:
`tokens(cap) = generated_tokens(4096) − truncated × 4096 + truncated × cap`.
At 8,192 the four models total 4,971,661 tokens; at 16,384, 8,666,253.
At the measured rates that is **2.574 and 4.292 GPU-hours**, a point estimate of
**6.866**. If every row ran to the cap the bound would be 11.189 and 22.379
GPU-hours, **33.568** together. The 4,096 rates are optimistic at longer
contexts, so the real rate must come from a 3-item smoke at each cap and never be
assumed. Registered ceilings: **16 h** for the 8,192 rung, **32 h** for the
16,384 rung, **48 h** for the ladder. A rung whose smoke-projected time exceeds
its ceiling is not launched for anyone; the ladder ends at the last completed cap
and the missing rung is recorded.

**Memory.** Rule: `max(1.5 × measured peak, measured peak + 1 GiB)` from a
3-item smoke at that cap, per `AGENTS.md`; the 2g CPU-court default is never
inherited. Prior reserved peaks at 4,096: soup 6.05 GiB, Qwen3-0.6B 5.53 GiB,
2B 8.31 GiB (receipts as in §1). KV-cache growth at 16,384 from the on-disk
configs, assuming batch 1: the hybrid Qwen3.5 models keep 6 of 24 layers in full
attention (2 KV heads × 256 × 2 × 2 B = 12,288 B/token), 0.19 GiB at 16,384;
Qwen3-0.6B (28 layers × 8 KV heads × 128 × 2 × 2 B = 114,688 B/token) needs
1.75 GiB. Both stay far below the 15.5 GiB bound that triggers a batch-size
reduction. If the launcher batches prompts the KV term multiplies and the smoke,
not this arithmetic, sets the limit.

**Disk.** `df -h /work` through the `io` court reports 36G available
(`results/courts/io-20261122T131807Z-e2ca1c.json`). The stored soup dump is
2,546,347 bytes for 959,895 generated tokens, 2.653 bytes/token, so the raw
outputs for both rungs and all four models bound at 130.38 MB; the registered
ceiling is 250 MB.

## 10. What does not change, and what is blocked

* **No win bar moves.** MATH-500 ≥ 50.0 and full MATH ≥ 50.0 with paired
  intervals above zero, GSM8K ≥ 62.38 and MMLU ≥ 51.07 (`docs/S71-LITERATURE.md`
  H.4) stay as registered.
* **MATH-500 and full MATH are two scales of one test set**, not independent
  replication: MATH-500 is a 500-problem subset of the 5,000-problem Hendrycks
  MATH test split, and `docs/S67-RESULT.md` §3 records that the full-corpus gold
  agrees with the MATH-500 answer field on all 500 overlapping problems. This
  probe reads MATH-500 only.
* **No new split is touched.** MATH-500 was already spent by the S67 and S68
  receipts; GSM8K test, MMLU, full MATH, the S69/S66 calibration splits and
  MATH-train are not read.
* **Blocked.** The GPU driver is broken: kernel module 620.70.06 against
  userspace 620.73, `nvidia-smi` fails with a driver/library version mismatch,
  and there is no root to reload it. No GPU court, CUDA call or model load may
  run until the operator reloads the driver. Everything in §2, §5, §6 and §9 was
  computed on CPU through the courts; no model was loaded.

## 11. Why the termination hypothesis may be wrong

1. **The stored data already points the other way for the simplest version.**
   Zero of 158 truncated soup rows contain a correct answer and only one
   contains a box, so "it answered and would not stop" is not the observed fault.
   If the extra tokens at 8,192 and 16,384 are more of the same, the ladder will
   recover few rows and correctly falsify the story.
2. **"Needs more room" and "cannot finish" are observationally similar at
   4,096.** The 0.6B finishes in ≤ 4,096 on 96.4% of the same problems, so our
   non-finishing is a learned behaviour, but the ladder alone cannot tell a model
   that is slow from one that is lost: both are truncated.
3. **The differential is partly arithmetic.** Qwen3-0.6B can gain at most 18
   problems from a higher cap (its entire truncated set), so a favourable
   differential is bounded by the soup's 158-row pool. The decision rule requires
   a differential of at least 22, but a reader should not read the bound as a
   mechanism.
4. **Truncation and wrongness are different failures.** The teacher's 4
   truncated-but-correct rows show that truncation can change the *reported*
   score by changing which box is last, with no change in reasoning. A cap
   increase that improves accuracy through last-box selection is not evidence for
   a termination deficit.
5. **Length-control training may miss the dominant class.** The fault is mostly
   "never reaches an answer" (116/158), not "does not stop after an answer"
   (0/158). A stop-after-box objective would not obviously fix never-reach; a
   length/budget objective might, which is why C12 is specified as
   termination/length control rather than stop-token training alone.
