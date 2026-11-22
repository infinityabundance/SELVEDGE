# S69 — four LoRA arms from the S67 soup champion: measured outcome

Status: **calibration-level failure of the preference arm, small degradation of
both completion-CE controls, and a cadence probe that partly rescues the
controls but not the arm.** No test split was read. Nothing was adopted.
The failure is attributed to the objective and the update regime, not to corpus
construction. This document is a report of measurement; it registers no arm.

Every number below is a quotation from the receipt named beside it, with that
receipt's stamp and the revision recorded inside it. Nothing here is
hand-computed from a log. `results/stage2/s71-overshoot/analysis.json` (stamp
`20261122T124848Z`, revision `d9f08ba`) is the single receipt that carries all
the derived statistics; the raw rows are in the calibration receipts it reads.

| artifact | receipt | stamp | revision |
|---|---|---|---|
| probe registration (committed before any score) | `docs/s71-overshoot-probe-preregistration-v1.json` | registered `2026-11-22T08:55:00Z`, first add `15fe95f` | `9c7e6c0` |
| soup calibration re-score | `results/stage2/s71-overshoot/soup-calibration.json` | `20261122T093710Z` | `15fe95f` |
| matched-sft 500k/1M | `results/stage2/s69-calibration/matched-sft.json` | `20261122T105819Z` | `8ffa1e6` |
| matched-sft-weighted 500k/1M | `results/stage2/s69-calibration/matched-sft-weighted.json` | `20261122T123744Z` | `0b44f9f` |
| preference 500k/1M/2M (already scored) | `results/stage2/s69-calibration/preference.json` | `20261122T062926Z` | `bdd8ae7` |
| arm receipts and update traces | `results/stage2/s69-preference/<arm>/receipt-<arm>.json`, `trace-<arm>.json` | arm-local stamps | `bdd8ae7` |
| analysis, falsification, training statistics | `results/stage2/s71-overshoot/analysis.json` | `20261122T124848Z` | `d9f08ba` |
| S69 selection and verdict | `results/stage2/s69-calibration/selection.json`, `results/stage2/s69-final/verdict.json` | `20261122T063627Z`, `20261122T075921Z` | `bdd8ae7` |

## 1. What ran

All four arms start from the S67 soup champion
(`results/stage2/s67-soup/interp-0.375-arm.pt`, sha256 `072b8946…a1d6ab`,
step 4000), install rank-64 / alpha-32 LoRA on every text-tower linear layer
except the head, train with `batch_pairs = 1` at lr `1e-4` and `grad_clip = 1.0`,
and are scored after merging. The calibration corpus is 214 positive-weight
pairs (263 loaded, 49 zero-weight dropped) for the preference and both CE arms;
the recovery arm uses the 49 zero-weight pairs. The preference cadences sit at
326/649/1294 steps and both CE controls at 742/1468/2935 steps over that pair
set.

The preference objective is `-log sigmoid(beta * margin)` with `beta = 0.1` on
mean-normalised sequence log-probabilities. The two CE controls use the same
chosen continuations: `matched-sft` with uniform weight, `matched-sft-weighted`
with the corpus-mean-normalised `w = p(1-p)` weight.

## 2. Training trajectory, clipping and gradient scale

All fields are from `analysis.json` §`training`, recomputed from the tracked
`trace-*.json` rows; `beta = 0.1` and the implied margin is
`m = log((1 - loss) / loss) / beta`.

| arm | steps | epochs over 214 pairs | first-step loss | first step below 0.05 | final loss | implied margin at final step | clipping-active fraction | pre-clip grad-norm p50 / p90 / max |
|---|---|---|---|---|---|---|---|---|
| preference | 1294 | 6.05 | 0.6285 | 61 | 5.85e-05 | 97.5 | 0.209 | 0.022 / 4.24 / 95.5 |
| matched-sft | 2935 | 13.71 | 0.1567 | 186 | 0.01334 | 43.0 | 0.024 | 0.340 / 0.727 / 1.67 |
| matched-sft-weighted | 2935 | 13.71 | 0.1421 | 186 | 0.00722 | 49.2 | 0.033 | 0.336 / 0.762 / 1.95 |
| recovery | 909 | 18.55 | 0.6931 | 70 | 1.24e-04 | 89.9 | 0.061 | 0.0028 / 0.194 / 28.1 |

Two things are visible before any capability score. First, the preference loss
falls through 0.05 by step 61 — inside the first third of the first epoch over
the 214 pairs — and ends at 5.85e-05, which under the registered loss implies a
mean-normalised margin of about 97 nats on the training pairs. Second, the
preference run clips on 20.9% of steps with pre-clip norms reaching 95.5, while
the CE controls clip on 2.4-3.3% of steps and never exceed 2.0. The preference
objective is producing a much larger, much spikier update than CE on the same
pairs at the same budget.

## 3. Calibration by cadence

Flexible accuracy on the registered 1,000-item splits; deltas are per-problem
paired bootstraps (4,000 resamples, seed 53) against the soup re-score, which
reproduced the stored S67 row exactly (`gsm8k-cal` 0.609, `math-cal` 0.326,
1,806,060 generated tokens), so arm and reference are the same measurement on
the same problems. Intervals are 95% percentile intervals on the paired delta.

| arm | cadence | GSM8K-cal | Δ vs soup [95%] | MATH-cal | Δ vs soup [95%] | Δ MATH vs own 2M [95%] |
|---|---|---|---|---|---|---|
| soup | — | 0.609 | — | 0.326 | — | — |
| matched-sft | 500k | 0.596 | −0.013 [−0.044, +0.017] | 0.300 | −0.026 [−0.053, +0.002] | +0.035 [+0.009, +0.061] |
| matched-sft | 1M | 0.592 | −0.017 [−0.047, +0.012] | 0.291 | −0.035 [−0.062, −0.010] | +0.026 [−0.000, +0.051] |
| matched-sft | 2M | 0.591 | −0.018 [−0.048, +0.012] | 0.265 | −0.061 [−0.088, −0.035] | — |
| matched-sft-weighted | 500k | 0.595 | −0.014 [−0.044, +0.016] | 0.302 | −0.024 [−0.051, +0.002] | +0.033 [+0.007, +0.058] |
| matched-sft-weighted | 1M | 0.542 | −0.067 [−0.098, −0.037] | 0.286 | −0.040 [−0.067, −0.014] | +0.017 [−0.008, +0.043] |
| matched-sft-weighted | 2M | 0.573 | −0.036 [−0.067, −0.004] | 0.269 | −0.057 [−0.084, −0.032] | — |
| preference | 500k | 0.000 | −0.609 [−0.639, −0.579] | 0.000 | −0.326 [−0.355, −0.298] | −0.006 |
| preference | 1M | 0.000 | −0.609 [−0.639, −0.579] | 0.000 | −0.326 [−0.355, −0.298] | −0.006 |
| preference | 2M | 0.043 | −0.566 [−0.597, −0.534] | 0.006 | −0.320 [−0.349, −0.291] | — |

MMLU-cal is not re-scored for the new cadences: no MMLU row exists for them and
this probe does not create one. The 2M rows remain the only MMLU evidence
(`matched-sft` 0.5185, `matched-sft-weighted` 0.514, `preference` 0.470, soup
0.5215; `results/stage2/s69-calibration/table.json`, stamp
`20261122T084104Z`, revision `9c7e6c0`, reading the `mmlu-*.json` receipts in the
same directory).

## 4. Generation behaviour

Counts and cap/EOS fractions are from the stored per-problem rows
(`samples`, `truncated_samples`) in `analysis.json` §`results.*.generation`;
`truncated` means the sequence reached the 2048-token cap without EOS, greedy
decoding contributes one sample per problem, so EOS presence is `1 − cap`.

| arm | cadence | generated tokens | GSM8K cap | GSM8K EOS | MATH cap | MATH EOS |
|---|---|---|---|---|---|---|
| soup | — | 1,806,060 | 0.050 | 0.950 | 0.403 | 0.597 |
| matched-sft | 500k | 2,106,574 | 0.088 | 0.912 | 0.518 | 0.482 |
| matched-sft | 1M | 2,059,068 | 0.090 | 0.910 | 0.487 | 0.513 |
| matched-sft | 2M | 2,084,944 | 0.072 | 0.928 | 0.522 | 0.478 |
| matched-sft-weighted | 500k | 2,077,051 | 0.103 | 0.897 | 0.493 | 0.507 |
| matched-sft-weighted | 1M | 2,005,251 | 0.082 | 0.918 | 0.464 | 0.536 |
| matched-sft-weighted | 2M | 2,206,286 | 0.118 | 0.882 | 0.556 | 0.444 |
| preference | 500k | 4,096,000 | 1.000 | 0.000 | 1.000 | 0.000 |
| preference | 1M | 4,096,000 | 1.000 | 0.000 | 1.000 | 0.000 |
| preference | 2M | 4,096,000 | 1.000 | 0.000 | 1.000 | 0.000 |

The controls' cap and EOS rates are in the same regime as the soup. The
preference checkpoints are not: every one of the 2,000 sequences at every
cadence runs to the 2048-token cap and none emits EOS.

**Loop status was not measured.** The calibration scorer stores no generated
ids, so `loop_cut_index` needs a fresh generation pass, and that pass did not
run: the GPU became unusable mid-session with a host driver/library version
mismatch (`nvidia-smi`: "Failed to initialize NVML: Driver/library version
mismatch"; running module 620.70.06 against userspace 620.73.04), which is a
host condition, not a property of the model. `scripts/s71_generation_diagnostics.py`
is written, registered and import-verified for that pass; it fails closed unless
every sampled problem reproduces the stored `flexible_correct`, so it cannot
report a diagnostic that disagrees with the score. No loop number is claimed
here. The cap/EOS rows above are the substitute evidence and they are exact,
not sampled.

## 5. Falsification outcome

The registration's condition: *if every earlier cadence is worse than its 2M
endpoint, the magnitude hypothesis is not supported by this corpus.*

**The condition is not met.** All four earlier cadences beat their own 2M
endpoint on MATH-cal (`matched-sft` +0.035 and +0.026; `matched-sft-weighted`
+0.033 and +0.017), so the magnitude hypothesis survives on this corpus. It is
weakened, not confirmed: two of the four intervals include zero, no earlier
cadence reaches the soup's 0.326, and the arm the hypothesis was meant to
explain moves in the opposite direction (§6).

## 6. The collapse: what the data supports and what it cannot distinguish

**Supported.** The preference arm's failure is a collapse into degenerate
generation, not a noisy null: 100% cap hit and 0% EOS at all three cadences, a
loss that reaches
5.85e-05 (margin ≈ 97 nats on the training pairs) after 6.05 epochs over 214
pairs, and a clipping rate six to nine times the CE controls' with pre-clip
norms up to 95.5. The signature is consistent with margin saturation of
`-log sigmoid(beta * margin)` on a small pair set: the objective has no bound
on the margin, the pair set is exhausted in 6 epochs, and the loss can be driven
arbitrarily low by separating the same 214 chosen/rejected pairs further at each
step. Nothing in the traces contradicts that reading.

**Not distinguishable from this data.** There is no held-out pair set: the 214
pairs used for training are the pairs the loss is measured on. So this report
cannot say whether the model learned a transferable correction or merely
suppressed the 214 specific rejected sequences — a large margin on the training
pairs is equally consistent with both, and the collapse to 0% EOS with
unbounded repetition on *unseen* calibration prompts is the signature of a
policy that has left the language manifold rather than acquired a rule. Nor can
it separate saturation from the gradient-spike regime: clipping at 20.9% of
steps and pre-clip norms of order 10² mean the optimizer took many large,
clipped steps in a direction selected by a saturated contrastive gradient. Both
readings predict the observed loss curve; only a held-out-pair margin
measurement, or a training run with the margin bounded, would separate them.

**Contradiction with the magnitude hypothesis.** The hypothesis predicts that
smaller magnitude is better. For the two CE arms it is: MATH-cal rises
monotonically as cadence falls, and at 500k the degradation against the soup
shrinks from −0.061 to −0.026 (`matched-sft`) and from −0.057 to −0.024
(`matched-sft-weighted`). For the preference arm it is the reverse: 500k and 1M
score 0.000/0.000 against 0.043/0.006 at 2M. The collapsing arm is *worse* at
every earlier cadence. A pure magnitude story therefore does not cover the arm
it was proposed to explain; the ordering that holds for CE does not hold for the
saturated contrastive objective.

One further non-monotonicity is recorded rather than smoothed: `matched-sft-
weighted` at 1M drops GSM8K-cal to 0.542, below both its 500k (0.595) and its
2M (0.573) endpoints, with a paired interval that includes zero against 2M. A
cadence index is not a clean magnitude dial — pair order is shuffled and the
weight realises per batch — so cadence ordering should not be read as a
controlled dose-response curve.

## 7. Failure classification

The failure is **objective and update-regime instability**, not corpus
construction:

* the two CE controls use the identical chosen-side corpus, the identical 214
  pairs, the identical cadence and budget, and change the objective only. They
  keep GSM8K-cal within 3.6 points and MATH-cal within 6.1 points of the soup at
  2M, and improve with earlier stopping. A corrupt or mismatched corpus would
  not spare them.
* the preference arm's distinguishing quantities are objective-side: unbounded
  margin, 20.9% clipping, pre-clip norms two orders of magnitude above the
  controls, and a final margin of about 97 nats on the training pairs.
* the collapse appears at all three cadences, so it is not an end-of-training
  artefact and not fixed by stopping earlier.

`docs/TODO.md` records this as B2's calibration-level failure and opens B5
("magnitude-controlled training from the champion") as the candidate that would
test a bounded-margin / controlled-norm regime.

## 8. Outcome flags

* `test_read: false` — no test split was read at any point in this probe or in
  the S69 verdict it continues (`results/stage2/s69-final/verdict.json`).
* `adopted: false` — no checkpoint is selected, promoted or compared on a test
  instrument. The four new rows are diagnostic; the S69 selection rule and
  adoption gate are unchanged and remain unmet.
* `diagnostic: true` — the cadence rows and the soup re-score exist to answer
  the registered question, not to select.
