# S69 — the preference mechanism initialised from the champion

Status: **registered, not run.** This file records the experiment, the
instrument and the selection discipline before any score exists.

Registration: `docs/s69-preference-from-champion-preregistration-v1.json`.

## 1. Question

The S68 depth ramp did not beat the S67 soup champion. The next registered
route to a MATH gain is the reference-anchored preference mechanism of S63/S64,
run for the first time **from the champion itself**:

* rollouts are sampled from the soup champion's own policy,
* the chosen side is a verified-correct teacher continuation on the student's
  exact token prefix,
* the rejected side is the student's own failed continuation from that prefix,
* the frozen reference is the soup champion, so the student starts at margin
  zero against the policy it is being nudged away from.

The S64 arm started from the post-trained baseline and its loss collapsed on
degenerate pairs. That is not this experiment. The fixed trainer drops
zero-weight pairs from the normal arm and the matched SFT control, trains on the
identical effective problem set, and exports merged full-weight checkpoints on
the CPU side.

## 2. Prompt set

GSM8K-train 400 prompts and MATH-train 250 prompts, selected by
`frontier.rft.stratified_select` with seed 6900 after removing:

* the frozen S58 stage-2a hold-out;
* the registered S66 GSM8K-cal and MATH-cal splits.

Removing the calibration ids keeps the selection instrument unseen by the
training arm. No test split is read at any point before the one-shot test.

## 3. Rollouts

K = 4, temperature 0.8, top-p 0.95, cap 2048, EOS plus the registered per-row
loop cut, batch 16 prompts, seed 6900. Every completion is verified with
`frontier.rft.verify` against the train label. The per-problem ledger records
attempts, `p = verified_correct / attempts`, truncation, loop-cut and duplicate
counters, so the pass rate survives the discard of failed text. One failed
continuation per problem is written as token ids only.

## 4. Pairs and reference

The teacher corpus (`s58-combined`) is reused only where its prompt ids are
token-identical to the rollout prefix; this was checked on a 200-row sample
before the run (200/200 exact matches). Missing coverage is generated with the
2B teacher from the student prefix and re-verified. Identity pairs are dropped.
The normal arm keeps `w = p(1-p)` pairs; `p = 0` problems form the
teacher-recovery arm at the matched mean positive weight.

The frozen-reference cache is the soup's per-token log-probabilities for the
fixed chosen and rejected actions, with the registered mean normalisation.

## 5. Training

Rank-64 / alpha-32 LoRA on every linear layer of the text tower except the
output head, base frozen, merged into a full-weight checkpoint. Two million
processed tokens, checkpoints at 0.5M / 1M / 2M. Arms:

* `preference` — the normal arm;
* `matched-sft` — completion CE on the chosen continuations of the identical
  effective problem set, same budget;
* `recovery` — the `p = 0` teacher pairs (trained only if at least 8 exist).

## 6. Selection, adoption and the one-shot test

Milestones are scored on the 1,000-item GSM8K-cal and MATH-cal splits, greedy,
non-thinking, cap 2048. The selection rule is highest MATH-cal, tie-broken by
higher GSM8K-cal and then the earlier cadence. The retention filter is MMLU-cal
within 1.0 point of the soup's calibration MMLU.

Adoption requires MATH-cal strictly above the soup's and GSM8K-cal within 1.0
point. If the gate passes, the selected checkpoint is scored **once** on GSM8K
test, MATH-500, full MATH (the S67 5,000-problem instrument), full MMLU and
MMLU-Pro 4,000. If it fails, no test split is read and the result is a
calibration-level failure.

The win bar is a paired comparison against the stored soup raw dumps: MATH-500
and full MATH intervals above zero, and GSM8K and MMLU not regressing beyond the
1.0-point bound. MMLU-Pro is secondary.

## 7. What this file does not establish

Nothing here is a measurement.
