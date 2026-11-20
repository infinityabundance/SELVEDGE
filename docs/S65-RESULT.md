# S65 — weight-space interpolation (WiSE-FT) between the post-trained 0.8B and S58 stage-2a

Status: **a capability gain on both math tasks, with MMLU retention passing and the
registered MMLU-Pro retention instrument unresolved.** This is one seed. The
method is weight-space interpolation (WiSE-FT; Wortsman et al., CVPR 2022):
`theta(t) = (1 - t) * theta_post + t * theta_s58`. The claim is the measured
capability, not a new mechanism.

Every number below is quoted from the receipt named beside it, with that
receipt's stamp and the revision recorded inside it. No test split was an input
to coefficient selection.

| artifact | receipt | stamp | revision |
|---|---|---|---|
| interpolation grid and calibration | `results/stage2/s60-interpolation/receipt.json` | `20261119T054403Z` | `a3ac4d1` |
| selection with calibration MMLU | `results/stage2/s60-interpolation/selected-receipt.json` | `20261119T092521Z` | `c2eb200` |
| alpha 0.375 GSM8K / MATH-500 | `results/stage1/s65-interp-0.375-nonthinking-greedy.json` | `20261119T200530Z` | `bc54b71` |
| alpha 0.375 paired math | `results/stage1/s65-paired-interp-0.375-vs-post.json` | `20261119T201515Z` | `bc54b71` |
| alpha 0.25 GSM8K / MATH-500 | `results/stage1/s65-interp-0.250-nonthinking-greedy.json` | `20261119T183014Z` | `bc54b71` |
| alpha 0.25 paired math | `results/stage1/s65-paired-interp-0.250-vs-post.json` | `20261119T185456Z` | `bc54b71` |
| full MMLU and MMLU-Pro retention | `results/stage2/s65-eval/raw/20261119T202606Z-mmlu.json`, `.../20261119T204054Z-mmlu_pro.json` | `20261119T205258Z` | `5000414` |
| selection/unused retention audit | `results/stage2/s65-eval/retention-audit.json` | `20261119T215807Z` | `41875e1` |
| selected checkpoint identity | `results/stage2/s65-interp-selected/identity.json` | `20261119T212853Z` | selection receipts above |

## 1. What was interpolated

* Base endpoint (`t = 0`): post-trained `Qwen/Qwen3.5-0.8B` at revision
  `2fc06364715b967f1860aea9cf38778875588b17`; first-shard weight sha256
  `04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696`.
* Candidate endpoint (`t = 1`): S58 stage-2a full-arm checkpoint
  `results/stage2/runs/s58-stage2a/ckpt-004000.pt`, sha256
  `33ee31ae5586799b2544ce4b888ae09c4f2654ab11b03301d05720ddb0bb7fc6`,
  step 4,000, 10,000,000 processed tokens.
* Selected coefficient: `t = 0.375`.
* Formula: `theta(t) = (1 - t) * theta_post + t * theta_s58`, mixed in float32
  and rounded back to each tensor's dtype.
* Scope: the 498,113,344 trainable text-tower elements exported by the full arm.
  The tied embedding/output projection (254,279,680 elements) is frozen and
  identical at both endpoints, so it is inherited from the post-trained
  snapshot and is not mixed.
* Unique active path: **752,393,024** text-tower elements. Headroom under the
  1,000,000,000 ceiling: **247,606,976**.

The accounting certificate for the declared text-only path is
`results/accounting/20261116T183609Z-qwen35-08b-text-only.json` (`ok=true`,
`752,393,024` unique active). **The capability claim rests on that accounting
certificate for the declared path, not on the per-token executed-set audit.** The
latter does not exist yet: it needs the P19.0 kernels (Court E), and no result
here is an executed-set proof.

## 2. Selection rule and calibration curve

The rule was fixed before any test score:

1. Score every grid coefficient on **GSM8K-cal and MATH-cal**, drawn from the
   train split only.
2. Exclude every S58 training and holdout id.
3. Compute the mean gain over the `t = 0` endpoint on the two calibration
   tasks.
4. A coefficient whose calibration MMLU drop exceeds **1.0 point** is
   ineligible. Calibration MMLU was a 2,000-item total-cap subset.
5. Ties choose the smaller coefficient.

No test item was read to choose `t`.

Calibration curve (`results/stage2/s60-interpolation/selected-receipt.json`,
stamp `20261119T092521Z`, revision `c2eb200`; 300 GSM8K-cal + 300 MATH-cal
items, 2,000 MMLU-cal items):

| t | GSM8K-cal | MATH-cal | mean gain vs t=0 | MMLU-cal | eligible |
|---|---|---|---|---|---|
| 0.000 | 52.67% | 34.00% | — | 50.60% | — |
| 0.125 | 55.00% | 34.00% | +1.17 | 51.45% | yes |
| 0.250 | 60.00% | 35.00% | +4.17 | 52.35% | yes |
| **0.375** | **61.33%** | **34.00%** | **+4.33** | **51.95%** | **yes, selected** |
| 0.500 | 60.00% | 34.67% | +3.33 | 51.60% | yes |
| 0.625 | 61.33% | 33.00% | +3.83 | 50.80% | yes |
| 0.750 | 56.00% | 29.33% | +0.33 | 50.50% | yes |
| 0.875 | 56.00% | 25.33% | −1.33 | 49.45% | yes |
| 1.000 | 54.33% | 23.67% | −2.67 | 48.65% | yes |

0.375 and 0.625 tie on GSM8K-cal (61.33%), but the registered objective is the
mean gain over both tasks; MATH-cal breaks the tie in favour of 0.375
(34.00% vs 33.00%). The selection receipt records `calibration_gain = 0.043333`
and `coefficient = 0.375`.

**Alpha 0.25 disclosure.** Alpha 0.25 has a lower calibration mean gain than the
selected 0.375 (+4.17 vs +4.33) and is not the registered selection. It was
scored on the test splits **after** selection to check sensitivity. It was never
adopted, never used to re-select, and appears below only as a sensitivity point.

## 3. Test scores

Instrument: `scripts/s54_post_trained_eval.py`, full GSM8K (1,319) and MATH-500
(500), greedy, non-thinking, zero-shot, user prompt ending in
`Put the final answer in \boxed{}`, batch 32, max_new_tokens 4,096, EOS-only
stop, bf16, transformers 5.23.0 / torch 2.15.0+cu132, SDPA attention. vLLM was
not installed; the transformers path is the engine of record. The primary score
is the receipt's `flexible` basis; `strict` and `exact` are reported beside it
and never substituted.

Receipts: `results/stage1/s65-interp-0.375-nonthinking-greedy.json` (stamp
`20261119T200530Z`, revision `bc54b71`) and its alpha 0.25 counterpart (stamp
`20261119T183014Z`, revision `bc54b71`).

| task | basis | post-trained 0.8B | S58 stage-2a (t=1) | t=0.375 | t=0.25 (sensitivity) |
|---|---|---|---|---|---|
| GSM8K | flexible | 54.21% (715/1319) | 57.09% (753/1319) | **64.90% (856/1319)** | 63.61% (839/1319) |
| GSM8K | strict | 54.06% (713/1319) | 57.01% (752/1319) | 64.90% (856/1319) | 63.31% (835/1319) |
| GSM8K | exact | 54.89% (724/1319) | 57.92% (764/1319) | 65.43% (863/1319) | 64.37% (849/1319) |
| MATH-500 | flexible | 38.20% (191/500) | 27.20% (136/500) | **42.80% (214/500)** | 44.40% (222/500) |
| MATH-500 | strict | 38.20% (191/500) | 27.20% (136/500) | 42.80% (214/500) | 44.40% (222/500) |
| MATH-500 | exact | 40.60% (203/500) | 29.00% (145/500) | 45.20% (226/500) | 48.20% (241/500) |

Prompt-level paired bootstrap, 4,000 resamples, seed 53, unit = the distinct
problem (`results/stage1/s65-paired-interp-0.375-vs-post.json`, stamp
`20261119T201515Z`, revision `bc54b71`; alpha 0.25 pair stamp
`20261119T185456Z`):

| coefficient | task | delta vs post-trained | 95% interval | interval above zero |
|---|---|---|---|---|
| **0.375** | GSM8K | **+10.69** | **[+8.04, +13.42]** | **yes** |
| **0.375** | MATH-500 | **+4.60** | **[+0.60, +8.60]** | **yes** |
| 0.25 | GSM8K | +9.40 | [+6.75, +11.90] | yes |
| 0.25 | MATH-500 | +6.20 | [+2.20, +10.20] | yes |

Both math clauses of the registered win are met at `t = 0.375`: the point
estimate is above the post-trained model and the paired interval is entirely
above zero. MATH-500's lower bound (`+0.60`) is the narrow clause.

## 4. Retention

Full MMLU (14,042 items) and MMLU-Pro were scored with
`scripts/s43_score_total_cap.py` through the lm-evaluation-harness, 5-shot,
identical for candidate and baseline. The S65 raw dumps are
`results/stage2/s65-eval/raw/20261119T202606Z-mmlu.json` and
`.../20261119T204054Z-mmlu_pro.json`; the baseline dumps are the S58 post-trained
baseline receipts. The registered bound is a **drop** bound: point drop ≤ 1.0
point and paired interval lower bound ≥ −1.0 point. An improvement is inside the
bound.

### 4.1 The 2,000 selection items are separated from the 12,042 never-used items

The coefficient selection read a 2,000-item MMLU total-cap subset. The full
retention score read 14,042 items. `scripts/s66_retention_audit.py` (receipt
`results/stage2/s65-eval/retention-audit.json`, stamp `20261119T215807Z`,
revision `41875e1`) recomputes all three partitions with the same bootstrap:

| partition | items | t=0.375 delta | 95% interval | t=0.25 delta | 95% interval |
|---|---|---|---|---|---|
| full MMLU | 14,042 | **+1.2605** | **[+0.6338, +1.8658]** | +1.3175 | [+0.7834, +1.8516] |
| 2,000 items used for selection | 2,000 | +1.3000 | [−0.4000, +2.9500] | +1.7000 | [+0.1988, +3.1500] |
| 12,042 previously unused items | 12,042 | **+1.2539** | **[+0.5979, +1.9681]** | +1.2539 | [+0.6477, +1.8602] |

The full-MMLU result is not driven by the selection items: on the 12,042 items
that selection never read, MMLU improves by +1.25 points with an interval above
zero. The audit reproduces the reported full-MMLU delta and interval exactly.

A note on the receipt's `passed` field. `scripts/s57_paired.py` sets `passed`
with `abs(delta) <= 0.01`, which marks an improvement larger than one point as
a failure. That is not the registered rule, which bounds the **drop**. Under the
registered rule MMLU passes; the audit records both readings rather than
rewriting the earlier receipt.

### 4.2 MMLU-Pro: the registered sample is absent

| sample | items | post-trained | t=0.375 | delta | 95% interval | verdict |
|---|---|---|---|---|---|---|
| retired MMLU-Pro sample | 504 | 22.42% | 25.20% | +2.78 | [−0.60, +6.55] | unresolved |
| registered total-cap sample | 4,000 | — | — | — | — | **not present** |

The S60 registration retires the 504-item MMLU-Pro sample because its paired
interval cannot resolve a 1.0-point bound, and registers a 4,000-item total-cap
subset instead. No such 4,000-item receipt exists on disk. The 504-item point
estimate and interval are inside the bound, but the registered instrument is
absent, so the clause is **unresolved**, not a pass.

### 4.3 Retention verdict and the complete win

* MMLU (14,042): **pass** under the registered drop rule; the improvement is
  reproduced on the 12,042 unused items.
* MMLU-Pro: **unresolved** because the registered 4,000-item sample does not
  exist.
* Math clauses: **both pass** at `t = 0.375` (GSM8K and MATH-500, paired
  intervals above zero).
* **Complete registered win: not met.** The blocking clause is the missing
  MMLU-Pro instrument, not a measured MMLU-Pro regression. If the 504-item
  sample were admitted it would pass; the registration does not admit it.

## 5. Termination and truncation

From the eval receipts' own detector (zlib tail compression + word n-gram
repetition + exact period; calibrated in
`results/stage1/s56-termination-calibration.json`). Truncated = the harness hit
4,096 new tokens without EOS. A truncated answer is never scored correct by the
receipt's `flexible` verdict.

| model | task | truncated | loop-classified | mean new tokens |
|---|---|---|---|---|
| post-trained 0.8B | GSM8K | 6.97% | — | 722 |
| post-trained 0.8B | MATH-500 | 32.00% | — | 1,897 |
| S58 stage-2a (t=1) | GSM8K | 2.27% | 0.53% | 573 |
| S58 stage-2a (t=1) | MATH-500 | 21.00% | 8.40% | 1,695 |
| **t=0.375** | **GSM8K** | **3.26% (43/1319)** | **0.076% (1)** | **595** |
| **t=0.375** | **MATH-500** | **27.80% (139/500)** | **6.20% (31)** | **1,838** |
| t=0.25 | GSM8K | 4.40% (58/1319) | 0.38% (5) | 619 |
| t=0.25 | MATH-500 | 29.80% (149/500) | 8.20% (41) | 1,887 |

At `t = 0.375` no truncated GSM8K answer was correct; on MATH-500 the 139
truncated answers were all incorrect. The loop-classified MATH-500 fraction
(6.2%) is lower than S58's 8.4% but higher than the post-trained baseline; the
post-trained baseline receipt predates the detector, so its loop fraction is
not quoted.

## 6. What this result is not

* It is **not a new mechanism**. Weight-space interpolation of two fine-tuned
  endpoints is WiSE-FT. The contribution here is the measured capability and
  the selection discipline, not the interpolation operation.
* It is **not a multi-seed result**. This page reports seed 0. The seed
  replication is a separate block added to this document only when its receipts
  exist.
* It is **not a completed registered win**, because the MMLU-Pro 4,000-item
  instrument is missing.
* It is **not a per-token executed-set audit**. The 752,393,024 unique-active
  count is the text-path accounting certificate; P19.0 kernels are required for
  the executed-set claim.
* It is **not selected on test**. The coefficient was selected on GSM8K-cal /
  MATH-cal / MMLU-cal. Alpha 0.25 was scored after selection and is disclosed
  as a sensitivity point, never adopted.

## 7. Interpretation

The S58 update at full strength improved GSM8K but destroyed MATH-500. At
`t = 0.375` the same update improves both math tasks and does not reduce MMLU.
The segment therefore contains a point where the capability gain survives while
the MATH damage is gone. That is the measured result; it does not establish that
`t = 0.375` is the best coefficient for a different seed, a different corpus or a
different model.

## 8. Seed replication at the fixed coefficient

Seeds 1 and 2 retrain the **exact S58 stage-2a recipe** and interpolate each
selected checkpoint at the fixed `t = 0.375`. No coefficient was re-selected on
a test split. The recipe holds S58's two known defects constant on purpose:

* example-normalised loss (current `train.py` defaults `auto` to the token
  unit, so the launcher passes `--loss-unit example` explicitly);
* `seq_len` 1,250, micro-batch 1, grad-accum 2, no length-aware batching,
  no gradient checkpointing, no SFT logits gather.

The corrected recipe (token-normalised loss and the 2,500 window) is a separate
future experiment and was not mixed in. Only the seed, run directory and one
safety guard differ.

**Guard-only deviation.** The S58 update guard (`max|update|/lr <= 10`) fires
deterministically for seed 1 at step 73 with `11.759` (two identical launches,
receipts `results/stage2/runs/s66-stage2a-seed1/abort.json` and
`...-seed1-r2/abort.json`). Seed 2 reaches `14.357`. The guard is a fail-safe,
not a data or optimiser knob; it was relaxed to `20.0` under
`docs/s66-replication-guard-amendment-v1.json`. The CE tripwire remains at 1.5.
This is a guard deviation, not a recipe change, and is labelled as such.

Receipts: `results/stage2/s66-seeds/three-seed.json` (stamp
`20261120T043635Z`, revision `2d6756e`), per-seed `result.json`, and the
training receipts under `results/stage2/runs/s66-stage2a-seed{1,2}-guard20/`.

| seed | GSM8K flexible | delta GSM8K | 95% interval | MATH-500 flexible | delta MATH-500 | 95% interval | MMLU | delta MMLU | 95% interval |
|---|---|---|---|---|---|---|---|---|---|
| 0 (S65) | 64.90% | **+10.69** | **[+8.04, +13.42]** | 42.80% | **+4.60** | **[+0.60, +8.60]** | 51.74% | +1.26 | [+0.63, +1.87] |
| 1 | 61.87% | +7.66 | [+5.00, +10.46] | 40.00% | +1.80 | [−2.20, +5.60] | 51.63% | +1.15 | [+0.55, +1.72] |
| 2 | 63.46% | +9.25 | [+6.60, +12.05] | 41.60% | +3.40 | [−0.60, +7.20] | 52.04% | +1.55 | [+0.94, +2.14] |
| **seed mean** | 63.41% | **+9.20** | SD 1.52 | 41.47% | **+3.27** | SD 1.40 | 51.80% | +1.32 | SD 0.21 |

Per-seed paired bootstrap: 4,000 resamples, seed 53, unit = distinct problem.
The seed-level mean and sample SD are descriptive; no coefficient is selected
from them.

**Verdict: the seeds do not hold.** All three point estimates are positive on
both math tasks, and GSM8K's interval is above zero for every seed. But
MATH-500's paired interval is above zero only for seed 0. Seed 1's lower bound
is `−2.20` and seed 2's is `−0.60`; both include zero. MATH-500's lower bound is
the clause that fails. MMLU retention passes for every seed, and its improvement
is stable (+1.15 to +1.55 points, all intervals above zero).

Because both math clauses do not remain above zero across seeds, the registered
seed-level win is **not met** and the conditional soup and depth-ramp
experiments in the task were not started.

### 8.1 Training-run facts

| seed | selected step | held-out CE | base CE | tokens | wall | guard max |
|---|---|---|---|---|---|---|
| 0 | 4000 | 0.30868 | 0.24945 | 10,940,000 | 5,257 s | 8.93 |
| 1 | 4000 | 0.30978 | 0.24945 | 10,940,000 | 5,634 s | 14.27 |
| 2 | 4000 | 0.31164 | 0.24945 | 10,940,000 | 5,799 s | 14.36 |

Every selected checkpoint is the step-4000 checkpoint, the same early-stop rule
as S58. All three runs are worse than the post-trained model on the held-out
teacher-solution CE, exactly as S58 was; this is the recipe's known negative and
is not repaired by interpolation.

## 9. Enlarged calibration split

The calibration split was enlarged from 300 to 1,000 items per task from unused
train problems, disjoint from S58 training/holdout and from every test split
(`docs/s66-calibration-splits-v1.json`, stamp `20261119T220027Z`, revision
`41875e1`). The enlarged split is a strict superset of the 300-item split. No
test split was read to build it.

The alpha grid re-run on the enlarged split is written to
`results/stage2/s66-calibration/large-grid.json` when the run exits. The grid was reduced to the
five coefficients that could change the maximum — `0, 0.25, 0.375, 0.5,
0.625` — because the full nine-point grid at 1,000 items per task exceeds the
available GPU budget; the four omitted points were dominated by more than half a
calibration point already at 300 items. This is a stability check, not a new
selection: no test split is read and no coefficient is adopted from it.
