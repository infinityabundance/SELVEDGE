# S69 — Amendment 1: how the registered difficulty weight is realized

Status: **registered before any calibration or test score, not run.** This
amendment amends `docs/s69-preference-from-champion-preregistration-v1.json`
without editing it. The machine-readable form is
`docs/s69-amendment-1-weight-realization-v1.json`.

Registration of record: `docs/s69-preference-from-champion-preregistration-v1.json`
(committed `9bd2a2894930a296813eb15a19641042d702a7cd`).
Experiment plan: `docs/S69-PREFERENCE-PLAN.md`.

The S69 corpus now exists and is receipted. What this amendment records is that
the corpus's registered weight `w = p(1-p)` was, until commit `aed2b56`, inert
in the trainer, and how it is made real. No selection input, threshold, budget
or scoring rule changes.

## 1. What changes and what does not

### Changed — the realization of the registered weight

1. **Loss denominator: corpus mean.** The registered per-pair weight
   `w_i = p_i (1 - p_i)` is divided by the corpus mean weight computed once
   from the filtered pair file, not by the current batch's weight sum
   (`--weight-normalization corpus`, the launcher default). The pre-S70
   denominator was the batch weight sum; at the registered
   `--batch-pairs 1` that sum is the single pair's own weight, so the ratio was
   `w_i L_i / w_i = L_i` and the weight cancelled exactly. Because the launcher
   drops `w <= 0` before budgeting, `w(p) = p(1-p)` had **no effect on the
   loss, the gradient or the update** in any pre-S70 run.
2. **Pair order stays `shuffled`.** `--pair-order shuffled` is the default and
   remains the registered primary order; it reproduces the pre-S70 stream from
   the seed, so the only realized difference in the primary arm is the
   denominator. `--pair-order weighted` (a deterministic
   difficulty-proportional permutation per epoch) is implemented and tested but
   is **not** part of the primary run: it is recorded as a sensitivity knob and
   is not used to select a checkpoint, a cadence or an arm.
3. **Realized-update telemetry.** `--trace-out` writes one row per optimizer
   step with the batch problem ids, the pair weights, the corpus mean weight,
   the normalized weight, the pre-clip and post-clip gradient norms, the L2 norm
   of the parameter delta the step actually produced, and whether the clip was
   active. The arm receipt records `corpus_mean_weight`,
   `weight_normalization`, `weight_scale_basis`, `realized_mean_weight` and
   `realized_mean_weight_over_mean`. This is the evidence that the weight is
   realized; a loss value alone is not, because AdamW's per-parameter
   normalization cancels a constant rescale of the gradient.

### Unchanged by this amendment

* The objective functional form: `-log sigmoid(beta * [(s+ - s-) - (r+ - r-)])`
  with mean-normalised sequence log-probabilities.
* `beta = 0.1`, `lr = 1e-4`, `--grad-clip 1.0`, LoRA rank 64 / alpha 32 on every
  text-tower linear layer except the output head, base frozen.
* The processed-token budget (2,000,000 forward-token positions) and the
  cadence (500,000 / 1,000,000 / 2,000,000).
* The effective problem set rule: the preference arm and the matched SFT control
  drop `w <= 0` pairs before budgeting; the recovery arm keeps the `p = 0`
  pairs at the matched mean positive weight.
* The calibration-only selection protocol (highest MATH-cal, tie-break higher
  GSM8K-cal then earlier cadence; MMLU-cal retention filter; adoption gate).
* The win bar (MATH-500 and full MATH paired 95% intervals above zero; GSM8K and
  MMLU point delta and interval lower bound not worse than -1.0).
* The one-shot test scoring rule and the no-enlargement clause.
* The frozen reference identity (`pi_ref` is the S67 soup champion).
* The fallback route.

## 2. Why the change is pre-score and legitimate

**The defect is an implementation defect, not a threshold.** The registered
objective says each pair contributes `w_i` times its loss. The trainer divided
by the batch sum. At one pair per optimizer step those are the same number, so
the registered weight never entered the update. Correcting this makes the run
implement the registration; it does not move the registration.

**Nothing scored existed at the fix.** The receipts in the tree at the time are
data-construction and instrument receipts: rollouts
(`results/artifacts/stage2/s69-rollouts/soup-receipt.json`), pairs and recovery
(`results/artifacts/stage2/s69-preference/soup-receipt.json`), and the frozen
reference caches
(`results/artifacts/stage2/s69-preference/pairs-with-reference-receipt.json`,
`results/artifacts/stage2/s69-preference/recovery-with-reference-receipt.json`).
They fix the corpus; they contain no calibration or test score, and no S69 arm
receipt exists. The first S69 score — calibration or test — is still in the
future, so there is no score to have tuned against.

**Measured under a real optimiser, not asserted.** The fix's effect on realized
updates is measured on a two-parameter toy language model trained with
`torch.optim.AdamW` and `clip_grad_norm_(1.0)`, in
`oracle/tests/test_s70_preference_weighting.py`:

* `test_both_knobs_change_the_realized_adamw_update` — compares the pre-fix
  configuration `("shuffled", "batch")` with the primary post-fix
  configuration `("shuffled", "corpus")` and with `("weighted", "batch")` and
  `("weighted", "corpus")`. It asserts that each arm's mean and final parameter
  trajectory distance from the pre-fix baseline exceeds the test's stated
  tolerance, and that neither knob alone reproduces the combination. The
  `("shuffled", "corpus")` arm is exactly the primary configuration this
  amendment registers.
* `test_adamw_trajectory_differs_between_uniform_and_weighted_training` —
  compares unweighted/uniform with weighted/normalized over a full epoch.
* `test_ordering_matters_when_the_budget_does_not_complete_a_whole_epoch` —
  pins the mid-epoch coverage effect that the sensitivity knob exercises.

The tests and their measurement ran as the `oracle` court:
`results/courts/oracle-20261122T023758Z-3450fa.json` (fourteen passed, revision
`40b2a11d12beaa8d57aea4bdbb6260bdec039fa7`). No trajectory number is restated
here; the receipt and the test module carry them.

**One precision, recorded rather than glossed.** A constant rescale of the
gradient is cancelled by AdamW, so "divide by a constant" alone would not be
evidence of anything. It matters here because the weight is *not* constant
across steps: the pair sampled at step `t` carries its own `w_i`, so the
gradient magnitude varies step to step and the optimiser's moments mix
differently-scaled gradients. The test's `("shuffled", "corpus")` arm measures
that configuration directly. The `weighted` order is the second, independent
knob; it is implemented, tested and not used in the primary run.

**Commit range.** `aed2b56` (corpus normalizer, weighted ordering),
`40b2a11` (per-step realized-update telemetry and launcher flags), ending at
`cdf4385`. `5f78e58` and `cdf4385` also carry the matched-compute control basis
(A2); that basis is adjacent bookkeeping, not part of the weight fix.

## 3. What would have been wrong to do

Any of the following would have turned a defect correction into a threshold
move or a selection change, and none was done:

* re-selecting any threshold, win-bar clause, retention bound, sample limit or
  cadence;
* touching the calibration adoption gate or the tie-break rule;
* reading any test split, or any calibration item, before the corruption was
  corrected and this amendment committed;
* adopting or re-selecting the soup coefficient alpha (0.375) after seeing test
  results;
* silently switching the primary order to `weighted`, which changes which pair
  each step sees and is therefore a different experiment;
* running the arms with `--weight-normalization batch` and calling the result
  the registered objective — `batch` exists only as the pre-S70 comparator for
  reproduction;
* rebuilding the corpus with different filters or weights to make the fix look
  cleaner;
* presenting the fix as a change to the hypothesis, or as a result.

## 4. The corpus's statistical limitation, recorded before the run

Read from the receipts, not from memory:

* The rollout ledger covers 650 problems at K = 4
  (`results/artifacts/stage2/s69-rollouts/soup-receipt.json`,
  stamp `20261122T011302Z`). Its `p_histogram` records 155 problems at `p = 0`
  and 226 at `p = 1`.
* The positive-weight pair file has 263 rows and the recovery file 49 rows
  (`sha256 ef32d4de…012fc` and `sha256 726b0323…f0495`;
  `results/artifacts/stage2/s69-preference/soup-receipt.json`,
  stamp `20261122T022911Z`).

**263 positive-weight pairs and 49 recovery pairs is a small corpus.** The
registered win bar is demanding relative to it: it requires two independent MATH
instruments (MATH-500 and the 5,000-problem full MATH) to have paired 95%
intervals above zero, plus no GSM8K or MMLU regression beyond the 1.0-point
bound. At this corpus size a real improvement of the size this program would
care about can easily be indistinguishable from zero on both instruments.

**No power calculation supports an expectation of clearing the bar.** None is
offered and none should be inferred. The run is registered to produce evidence,
not to confirm a prior. If the result is null, it will be reported as a null
with its interval, the interval's width and the corpus size that produced it —
not re-framed as "underpowered", not rescued by re-running on a different split,
and not replaced by a favourable secondary instrument. A `p = 1`-heavy
histogram also means much of the rollout sample contributes no preference signal
at all; the effective problem set is smaller than 650 by construction.

## 5. The three arms, as they will be launched

One arm per invocation, one GPU job at a time, all through `scripts/court` with
`--gpu oracle`. The frozen-reference caches are already built; the arms read
`pairs-with-reference.jsonl` and `recovery-with-reference.jsonl`.

Memory is sized from each arm's own smoke, per `AGENTS.md` §2. The smoke's
`20g` is a **bound, not a measurement**; the identical-geometry S64 preference
receipt is the pre-smoke prior
(`results/stage2/runs/s64-preference/receipt-preference.json`). The arm's limit
is computed from its own smoke receipt and stated in the launch record.

```sh
SMOKE_LIMIT=20g

# ---- arm 1: preference (the registered primary) ----
SELVEDGE_MEM=$SMOKE_LIMIT SELVEDGE_CPUS=8 SELVEDGE_PIDS=1024 SELVEDGE_TMP=2g \
  scripts/court --gpu oracle -- python3 scripts/s64_preference_launch.py \
    --arm preference \
    --pairs results/artifacts/stage2/s69-preference/pairs-with-reference.jsonl \
    --student-dir /hf-cache/models/Qwen/Qwen3.5-0.8B/2fc06364715b967f1860aea9cf38778875588b17 \
    --init-checkpoint results/stage2/s67-soup/interp-0.375-arm.pt \
    --budget-tokens 2000000 --cadence 500000,1000000,2000000 \
    --batch-pairs 1 --lr 1e-4 --beta 0.1 --grad-clip 1.0 \
    --rank 64 --alpha 32 --seed 6900 \
    --pair-order shuffled --weight-normalization corpus \
    --max-steps 20 \
    --out-dir results/stage2/s69-preference/preference/smoke

# size the arm from its own smoke (max(1.5 x peak, peak + 1 GiB); the cgroup
# peak is included because the court's limit must cover it)
SMOKE=results/stage2/s69-preference/preference/smoke/receipt-preference.json
LIMIT=$(python3 -c "import json,sys; r=json.load(open(sys.argv[1])); p=max(r['peak_reserved_gib'], r['peak_allocated_gib'], int(r['cgroup_memory']['memory_peak'])/2**30); print(f'{max(1.5*p, p+1):.1f}g')" "$SMOKE")

SELVEDGE_MEM=$LIMIT SELVEDGE_CPUS=8 SELVEDGE_PIDS=1024 SELVEDGE_TMP=2g \
  scripts/court --gpu oracle -- python3 scripts/s64_preference_launch.py \
    --arm preference \
    --pairs results/artifacts/stage2/s69-preference/pairs-with-reference.jsonl \
    --student-dir /hf-cache/models/Qwen/Qwen3.5-0.8B/2fc06364715b967f1860aea9cf38778875588b17 \
    --init-checkpoint results/stage2/s67-soup/interp-0.375-arm.pt \
    --budget-tokens 2000000 --cadence 500000,1000000,2000000 \
    --batch-pairs 1 --lr 1e-4 --beta 0.1 --grad-clip 1.0 \
    --rank 64 --alpha 32 --seed 6900 \
    --pair-order shuffled --weight-normalization corpus \
    --trace-out results/stage2/s69-preference/preference/trace-preference.json \
    --out-dir results/stage2/s69-preference/preference

# ---- arm 2: matched-sft control (identical effective pairs, chosen only) ----
SELVEDGE_MEM=$LIMIT SELVEDGE_CPUS=8 SELVEDGE_PIDS=1024 SELVEDGE_TMP=2g \
  scripts/court --gpu oracle -- python3 scripts/s64_preference_launch.py \
    --arm matched-sft \
    --pairs results/artifacts/stage2/s69-preference/pairs-with-reference.jsonl \
    --student-dir /hf-cache/models/Qwen/Qwen3.5-0.8B/2fc06364715b967f1860aea9cf38778875588b17 \
    --init-checkpoint results/stage2/s67-soup/interp-0.375-arm.pt \
    --budget-tokens 2000000 --cadence 500000,1000000,2000000 \
    --batch-pairs 1 --lr 1e-4 --beta 0.1 --grad-clip 1.0 \
    --rank 64 --alpha 32 --seed 6900 \
    --pair-order shuffled --weight-normalization corpus \
    --trace-out results/stage2/s69-preference/matched-sft/trace-matched-sft.json \
    --out-dir results/stage2/s69-preference/matched-sft

# ---- arm 3: teacher recovery (the p = 0 pairs at the matched mean weight) ----
SELVEDGE_MEM=$LIMIT SELVEDGE_CPUS=8 SELVEDGE_PIDS=1024 SELVEDGE_TMP=2g \
  scripts/court --gpu oracle -- python3 scripts/s64_preference_launch.py \
    --arm recovery \
    --recovery-pairs results/artifacts/stage2/s69-preference/recovery-with-reference.jsonl \
    --student-dir /hf-cache/models/Qwen/Qwen3.5-0.8B/2fc06364715b967f1860aea9cf38778875588b17 \
    --init-checkpoint results/stage2/s67-soup/interp-0.375-arm.pt \
    --budget-tokens 2000000 --cadence 500000,1000000,2000000 \
    --batch-pairs 1 --lr 1e-4 --beta 0.1 --grad-clip 1.0 \
    --rank 64 --alpha 32 --seed 6900 \
    --pair-order shuffled --weight-normalization corpus \
    --trace-out results/stage2/s69-preference/recovery/trace-recovery.json \
    --out-dir results/stage2/s69-preference/recovery
```

Each arm writes, under its `--out-dir`:

| arm | arm receipt | update trace | cadence checkpoints |
|---|---|---|---|
| preference | `results/stage2/s69-preference/preference/receipt-preference.json` | `…/preference/trace-preference.json` | `…/preference/ckpt-<step>.pt` + `adapter-preference-<steps>.pt` |
| matched-sft | `results/stage2/s69-preference/matched-sft/receipt-matched-sft.json` | `…/matched-sft/trace-matched-sft.json` | `…/matched-sft/ckpt-<step>.pt` + `adapter-matched-sft-<steps>.pt` |
| recovery | `results/stage2/s69-preference/recovery/receipt-recovery.json` | `…/recovery/trace-recovery.json` | `…/recovery/ckpt-<step>.pt` + `adapter-recovery-<steps>.pt` |

The per-arm directories exist because the merged checkpoint filenames
(`ckpt-<step>.pt`) are not arm-tagged: a shared directory could silently
overwrite one arm's checkpoint with another's.

Each arm's court receipt lands at `results/courts/oracle-<stamp>-<id>.json`.
The arm receipts do **not** name an S69 score; the calibration and test receipts
named by the registration are the only scoring receipts, and none exists yet.

## 6. Reference identity: `pi_ref` is the start point

The frozen reference `pi_ref` is **the same S67 soup champion the student is
initialised from**: `results/stage2/s67-soup/interp-0.375-arm.pt`, step 4000,
`sha256 072b89463c284ac8346b666527eb678f284165ae93f6abf55685fdaf35a1d6ab`.
The reference-cache receipts record this identity explicitly
(`reference.identity` in
`results/artifacts/stage2/s69-preference/pairs-with-reference-receipt.json` and
`…/recovery-with-reference-receipt.json`).

Consequences for interpreting the DPO margin, recorded now:

* The initial margin `(s+ - s-) - (r+ - r-)` is exactly zero at step zero,
  because the student and the reference are the same weights. The margin is
  therefore a **displacement from the start point**, not a preference between
  two independently trained policies.
* The reference term acts as a pull back toward the champion. It bounds drift,
  which is what the selection and retention rules also want, but it means the
  objective cannot distinguish "the student moved toward the chosen side" from
  "the student is far from the champion". Both appear in the same scalar.
* The reward difference is not identifiable as an external judge's preference.
  A positive margin is evidence that the student has moved toward the chosen
  teacher/verifier continuation relative to its own initialisation; it is not
  evidence that a differently initialised model, or a human, would prefer the
  same side.
* The chosen side is a verified-correct teacher continuation and the rejected
  side is the student's own failure, so part of every margin is
  teacher-versus-student quality, not soup-versus-soup preference.
* If the arm clears the win bar, the gain is attributable to the pair
  construction, the objective and the anchor-to-start, not to the choice of an
  independent reference model. No reverse-KL or "true reward" interpretation is
  claimed; the reference receipts carry `reverse_kl_not_claimed: true`.

## 7. What this file does not establish

Nothing here is a measurement of the preference mechanism. It records a
correction to how a registered weight is realized, the corpus it will be
realized on, the commands, and the interpretation limits. It moves no
threshold.
