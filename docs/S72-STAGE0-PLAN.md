# S72 Stage 0 — teacher qualification (plan)

Status: **registered, not run.** The registration is
`docs/s72-teacher-qualification-preregistration-v1.json`. It registers
**only Stage 0**: no trace generation, no training arm and no test scoring are
authorised by it. Execution is blocked on the GPU driver (see §8).

This document is the human-readable companion to that JSON. Where a number is
measured it names the receipt, stamp and revision that produced it; where a
number is arithmetic on measured inputs the inputs and the formula are stated;
where nothing was measured the cell says **unknown**.

## 1. What the recon already tells us

All rows below are our S54 protocol unless marked otherwise: greedy,
non-thinking, zero-shot CoT, the S54 user template, `max_new_tokens = 4096`,
flexible basis with strict and exact beside it. The machine-readable table is
`results/stage2/s72-recon/teacher-recon.json` (stamp `20261122T130947Z`, built
from the receipts named in each row).

| model | revision | instrument | flexible | strict | exact | cap | receipt | stamp | revision in receipt |
|---|---|---|---|---|---|---|---|---|---|
| Qwen/Qwen3.5-2B (teacher) | `15852e8c…` | MATH-500 | **65.80%** | 65.80% | 70.20% | 4096 | `results/stage1/s54-stage1-post-2b-nonthinking-greedy.json` | `20261118T045605Z` | `863b3309…` |
| Qwen/Qwen3.5-2B (teacher) | `15852e8c…` | GSM8K test | **78.39%** | 78.17% | 79.38% | 4096 | same | same | same |
| Qwen/Qwen3.5-2B (teacher) | `15852e8c…` | full MATH (5,000) | **unknown** | unknown | unknown | — | never scored | — | — |
| Qwen/Qwen3.5-2B (teacher) | `15852e8c…` | MMLU | **unknown** | unknown | unknown | — | never scored in our protocol | — | — |
| Qwen/Qwen3.5-2B (teacher) | `15852e8c…` | MATH-500, maj16 on 50 items | 74.00% | 74.00% | 82.00% | 4096 | `results/stage1/s54-stage1-post-2b-nonthinking-maj16-subset50.json` | in-receipt | `863b3309…` |
| Qwen/Qwen3-0.6B | `c1899de2…` | MATH-500 | **48.80%** | 48.80% | 52.00% | 4096 | `results/stage2/s68-frontier/qwen3-0.6b.json` | `20261121T192906Z` | `4550ae79…` |
| Qwen/Qwen3-0.6B | `c1899de2…` | GSM8K test | 61.26% | 61.18% | 62.77% | 4096 | same | same | same |
| Qwen/Qwen3-0.6B | `c1899de2…` | full MATH | **unknown** | unknown | unknown | — | never scored | — | — |
| Qwen/Qwen3-0.6B | `c1899de2…` | MMLU | **unknown** | unknown | unknown | — | never scored | — | — |
| Qwen/Qwen3-1.7B | `70d244cc…` | MATH-500 / GSM8K / full MATH / MMLU | **unknown** | unknown | unknown | — | cached, never scored | — | — |
| Qwen/Qwen3.5-4B | `851bf6e8…` | any | **unknown** | unknown | unknown | — | no weights on disk (config + tokenizer only) | — | — |
| Qwen/Qwen3.5-9B | `c2022362…` | any | **unknown** | unknown | unknown | — | no weights on disk | — | — |
| Qwen/Qwen3-4B | `1cfa9a72…` | any | **unknown** | unknown | unknown | — | never downloaded | — | — |
| Qwen/Qwen2.5-Math-7B-Instruct | unknown | any | **unknown** | unknown | unknown | — | absent; no pinned revision recorded | — | — |
| Qwen/Qwen2.5-Math-1.5B-Instruct | unknown | any | **unknown** | unknown | unknown | — | absent; no pinned revision recorded | — | — |
| amd/ReasonLite-0.6B | `3aad2248…` | MATH-500 | 36.00% | 36.00% | 38.00% | 4096 | `results/stage2/s68-frontier/reasonlite-0.6b.json` | in-receipt | `782ec209…` |
| amd/ReasonLite-0.6B | `3aad2248…` | GSM8K test | 30.93% | 30.02% | 33.43% | 4096 | same | same | same |

The Base checkpoint is measured on a **different instrument** and is listed separately
rather than forced into the flexible/strict columns. From
`results/frontier/anchor-2b-20261116T212613Z.json` (stamp `20261116T212622Z`,
harness commit `d6de8164…`), lm-evaluation-harness 0.4.14.dev0, completion-style
prompting with the harness's own few-shot demonstrations, bf16 on the same card:

| task | metric | value |
|---|---|---|
| minerva_math500 | `math_verify` | 35.20% |
| minerva_math500 | `exact_match` | 33.40% |
| gsm8k | `exact_match,flexible-extract` | 59.74% |
| gsm8k | `exact_match,strict-match` | 59.67% |
| mmlu | `acc` | 53.95% |

Those rows are **not** comparable to the S54 table: different prompt contract,
different shots and a harness extraction. They are recorded so the recon table
has no silently missing cell, and they are excluded from every Stage 0 decision.

Baselines, to confirm `docs/S67-RESULT.md`:

| model | MATH-500 | full MATH | GSM8K test | MMLU (full) | receipt | stamp |
|---|---|---|---|---|---|---|
| post-trained Qwen3.5-0.8B | 38.20% | 39.60% | 54.21% | 50.48% | `results/stage1/s54-stage1-post-0.8b-nonthinking-greedy.json`, `results/stage2/s67-final/baseline-math-full.json`, `results/stage2/s67-final/retention-audit.json` | `20261118T033841Z`, `20261121T025644Z`, `20261121T025710Z` |
| S67 soup (alpha = 0.375) | 40.60% | 42.12% | 63.38% | 52.07% | `results/stage2/s67-final/soup-stage1.json`, `results/stage2/s67-final/retention-audit.json` | `20261120T225912Z`, `20261121T025710Z` |

The S67 table is reproduced exactly. The post-trained full-MATH and MMLU rows
are quoted from the S67 receipts, which score the same checkpoint the S54
receipt scores; the S54 receipt itself has only GSM8K and MATH-500.

**What this means for Stage 0.** The teacher we already used has a measured
MATH-500 of 65.80% at our own cap — below the 70% gate — and 115 of its 500
MATH-500 completions (23.0%, from the receipt's `truncated_samples`) hit the
4,096 cap. Both facts were unknown to the S71 recommendation, which treated the
2B as a control with an unspecified number. A fresh MATH-train sample is the
right test because MATH-train is easier than MATH-500 and a train-sample pass
with a test-sample fail is a meaningful, honest outcome (see §5).

## 2. The 48.80 claim, audited

The claim is `Qwen/Qwen3-0.6B` = 48.80% on MATH-500 (`n = 500`), from
`results/stage2/s68-frontier/summary.json` (stamp `20261121T234201Z`, revision
`782ec209…`), whose per-problem row is
`results/stage2/s68-frontier/qwen3-0.6b-raw/math500-non-thinking-greedy.jsonl`.

Re-extraction of all 500 stored completions with the scorer's own functions
(`scripts/s54_post_trained_eval.py` lines 276–310, replay only — no model was
reloaded):

| question | answer |
|---|---|
| cap | `max_new_tokens = 4096`, EOS-only stop |
| basis of 48.80 | flexible — and flexible equals strict exactly |
| strict | 244/500 = 48.80% |
| flexible | 244/500 = 48.80% |
| exact (receipt, `math_verify 0.9.0`) | 260/500 = 52.00% |
| extraction methods (flexible) | `last \boxed{}` 484, `none` 16 |
| answers parsed by the non-boxed fallback | **0**. No MATH answer used the "final answer is" fallback; no strict-empty row was rescued |
| truncated completions | **18/500 = 3.60%** |
| unparseable (no boxed answer, no fallback) | 16 |
| ambiguous (≥2 distinct boxed answers, or a boxed "or") | 15 |
| new-token p50 / p90 / p99 / max | 449 / 1,485 / 4,096 / 4,096 |

So the 48.80 is a **pure `\boxed{}` number**, not an inflated flexible number:
the flexible fallback contributed nothing on MATH-500, and only 3.6% of its
outputs were truncated. It is comparable to our soup's flexible number on the
same problems, which is also 40.60% with zero fallback rescues and 158/500
truncated (31.6%) — the soup's own truncation rate is the larger of the two.

Paired, same 500 problems:

| pair | both correct | 0.6B only | soup only | neither |
|---|---|---|---|---|
| flexible | 151 | 93 | 52 | 204 |

0.6B 244/500 vs soup 203/500, a net +41 problems for 0.6B. Anything that
claims to beat 48.80 on MATH-500 must beat that on this same instrument; it is
a real ceiling to beat, not an extraction artefact.

## 3. Corpus facts a generation pass must respect

From `results/stage2/s72-recon/corpus-audit.json` (stamp `20261122T130952Z`),
computed by `scripts/s72_corpus_audit.py` through `scripts/court oracle` with
`SELVEDGE_MEM=4g SELVEDGE_CPUS=2 SELVEDGE_PIDS=512 SELVEDGE_TMP=1g`:

* The combined corpus `results/artifacts/stage2/s58-combined/stage2a-accepted-combined.jsonl`
  (sha256 `066c1480…`, 4,084 rows) is 2,639 GSM8K and 1,445 MATH.
* MATH by level: 0 → 2, 1 → 164, 2 → 365, 3 → 370, 4 → 319, 5 → 225.
* MATH by subject: algebra 485, counting/probability 138, geometry 110,
  intermediate algebra 150, number theory 158, prealgebra 290, precalculus 114.
* `solution_ids` (the 0.8B tokenizer's own ids stored at generation time, so no
  tokenizer was loaded): p50 456, p90 915, p99 1,736, max 2,046 over all 4,084
  rows; MATH-only p50 700, p90 1,324, p99 1,951, max 2,046. The identity is
  checked, not assumed: `Qwen/Qwen3.5-0.8B`, `Qwen/Qwen3.5-2B` and the soup
  receipt all record the same `tokenizer.json` (12,807,982 B, sha256
  `5f9e4d49…`), so the stored ids are byte-identical to what the 0.8B would
  produce.
* `prompt_ids`: overall p50 78, p90 113, p99 204, max 831; MATH p50 68, p90 115,
  p99 324, max 831; GSM8K p50 82, p90 113, p99 152, max 200. These are the
  measured prompt lengths (the generator's own tokenisation), not an assumption.
* Truncation if the prompt is included: at 2,048 **11 rows** (all MATH) exceed
  the window; at 2,500, 3,072 and 4,096 **0 rows** do. Completion-only excess is
  0 at every cap because no generation ran above its own 1,024/2,048 cap.
* MATH-train coverage: the fetched train split has 7,472 parseable items (7,500
  rows, 28 with an unparseable reference answer). The union of every accepted
  corpus in the tree (teacher S55/S58 plus the S59/S64/S69 student rollouts)
  covers **1,796 distinct MATH-train problems (24.0%)** and leaves **5,676
  uncovered (76.0%)**. The S58 teacher corpus alone covers 1,445 (19.3%).
* Difficulty of the covered vs uncovered sets: covered level 1/2/3/4/5 =
  204/455/458/402/275; uncovered = 359/889/1,125/1,282/2,021. The uncovered
  mass is concentrated at level 5 (2,021 of 5,676), exactly the stratum S58
  discarded most often.
* The S60 ledger (`results/stage2/s60-long-example-share.json`, stamp
  `20261119T032353Z`, revision `9c179e3`) already measured the row-width
  question on the trained examples: at width 1,250 it drops 191 examples and
  292,068 supervised tokens (13.33% of teacher tokens), every one of them MATH,
  skewing to level 5 (67 of the 191); at 2,048 it drops 11 examples / 21,688
  tokens (5 × level 5, 4 × level 4, 2 × level 3); at 2,500 it drops **zero**.
  Our independent count above agrees: 11 rows exceed 2,048 with the prompt,
  none exceed 2,500.

**What a 2,500-token window now recovers, precisely.** It recovers 100% of the
4,084 already-accepted traces (the 11 rows over 2,048 are the same 11 the S60
ledger names) and 292,068 tokens of long MATH supervision relative to a 1,250
window. It does **not** recover the 840 MATH prompts that were truncated at the
generation cap of 2,048 in the S58 run (receipt
`results/artifacts/stage2/s58-stage2a-math-2048/stage2a-math-2048-receipt.json`,
`truncated` = 840 of 2,400, i.e. 35.0%), nor the 1,267 truncated at 1,024. How
many of those would become verified solutions at 2,500 is **unknown** and can
only be closed by a new generation pass, not by widening the training row.

## 4. The instrument, frozen

`results/artifacts/stage2/s72-stage0/sample-250x2.jsonl`,
sha256 `1c4aa39fc1a25ce9b45671a4966d0ffc8636100d72dae470fa0ffaa40433b838`,
500 rows, 250 MATH-train + 250 GSM8K-train, drawn by
`scripts/s72_stage0_sample.py`:

* pool = parseable train items minus every id in the S58 holdout and the two
  1,000-item S66 calibration splits, minus any train question whose exact
  normalised text matches a test question. MATH: 7,472 → 6,442 eligible.
  GSM8K: 7,473 → 6,420 eligible. The test-question filter removed exactly one
  MATH-train row and zero GSM8K rows.
* selection = sort pool by id, `random.Random(seed).shuffle`, take N, sort by
  id; seeds 7200 (MATH) and 7201 (GSM8K).
* MATH sample levels: 1 → 24, 2 → 40, 3 → 56, 4 → 59, 5 → 71.
* diagnostic, not an exclusion: 58/250 MATH and 105/250 GSM8K sampled problems
  already have an S58 teacher trace, so roughly a quarter of the MATH sample
  overlaps the student's training data.

**Deviation from S71 H.1, stated plainly.** H.1 asks for "the first 250
MATH-train and 250 GSM8K-train problems after the same exclusions". A prefix
cannot be hashed before scoring in a way a reader can verify, so this
registration uses a seeded shuffle instead and records the profile of the draw.
The exclusion set is the one H.1 names plus the test-question filter; every other
part of the instrument is unchanged.

**Disclosure.** The sample builder opened the MATH test corpus, MATH-500 and the
GSM8K test split *only* to compute the exact-question disjointness filter. No
test score, receipt or per-problem verdict was read, and no test item is scored
by Stage 0.

## 5. Gate

A teacher qualifies iff, on that fixed sample:

* flexible accuracy on the 250 MATH-train problems **≥ 70%** (≥ 175 correct), and
* flexible accuracy on the 250 GSM8K-train problems **≥ 90%** (≥ 225 correct).

The point estimate is the gate. The Wilson 95% interval is reported beside it
and does not move, soften or widen the bar. 0.696 is a fail; 0.70 is a pass. The
sample is read once per candidate; an OOM kill is a receipted failure and the
run is repeated on the identical sample, never redrawn.

## 6. Decision rule (fixed before any score)

* **T1 passes both** → T1 (`Qwen/Qwen3.5-2B`) is the qualified teacher and a
  separate Stage 1 registration may proceed. No capability claim is made here.
* **T1 fails MATH but passes GSM8K, and 65.80 lies inside the new sample's
  Wilson interval** → record *the 2B teacher is already near its own ceiling on
  MATH under this protocol*. Consequence: a 2B-teacher distillation cannot reach
  MATH-500 50 through trace quality alone — the teacher's own MATH accuracy
  bounds the traces, and our soup at 40.60 would have to close 37.3% of the
  teacher's 25.20-point margin to reach 50.0 ((50.0 − 40.60) / (65.80 − 40.60)
  = 0.373). Do not fund another 2B trace run; the named next
  route is a stronger teacher (Stage 0b, which needs a weights download and a
  separate operator-approved budget) or G2/G4. Stage 0 is recorded as failed.
* **T1 fails MATH and the interval excludes 65.80** → the instrument or the
  execution is the constraint, not the teacher's level. The next experiment is
  the S71 H.1 cap/format study; the 23.0% MATH-500 truncation is the evidence
  that the cap may be binding.
* **T1 fails GSM8K** → protocol failure (template, chat mode, truncation), not a
  capability result: a checkpoint measured at 78.39% on full GSM8K cannot fall
  below 90% of an easier 250-item train sample for a capability reason alone.
  Diagnose the protocol on the same sample; do not change the teacher.
* **Only T2 (`Qwen/Qwen3-1.7B`) passes** → the stage still fails its primary
  gate. T2 is recorded as a control and cannot be substituted, because it is
  smaller than the teacher G1 tests.
* **Nobody passes** → the stage fails; the named fallback is the cap/format
  study, and if the sample's MATH truncation fraction exceeds 0.20 at 4,096 it
  is the required next experiment rather than an option.
* **Strict ≪ flexible on MATH for a candidate** → report both and refuse to
  quote the flexible number as a pure `\boxed{}` number. The 48.80 row has
  flexible == strict with zero fallback rescues, so fallback-dependent teacher
  rows are not comparable to it.

## 7. Cost and limits

Measured inputs, from the T1 receipt above (stamp `20261118T045605Z`,
revision `863b3309…`): 528.79 tokens/s with mean 1,692.46 new tokens on
MATH-500, and 260.30 tokens/s with mean 576.46 new tokens on GSM8K.

Expected Stage 0 tokens: 250 × 1,692.46 = 423,115 (MATH) and
250 × 576.46 = 144,114 (GSM8K), total 567,229. Time at those rates:
423,115/528.79 + 144,114/260.30 = 1,353.8 s = **0.376 GPU-hours** for T1.
With the 1.5× slack that is 0.564 h; T2 is smaller, so both candidates plus two
3-item smokes are registered at a **2.0 GPU-hour ceiling**. Stage 0 generates no
training data. This replaces S71 H.1's 4–6 h estimate with a derived bound.

Disk: **0 additional bytes**. Both candidates are already under
`/hf-cache/models` (T1 4,548,221,488 B; T2 4,063,515,592 B). Qwen3.5-4B and
Qwen3.5-9B have config and tokenizer but no `*.safetensors`, so they are not
downloadable-without-approval candidates for this stage.

Memory rule: smoke first on the identical shape, then
`max(1.5 × measured peak, measured peak + 1 GiB)`. The prior shape reference is
the same T1 receipt: peak VRAM 7.78 GiB, reserved 8.31 GiB, cgroup peak
1,532,653,568 B, which implies 11.67 GiB of GPU budget and confirms that the 2g
CPU-court default must never be inherited for this work. If the smoke's reserved
peak + 1 GiB exceeds 15.5 GiB, reduce batch size and re-smoke; batch size is an
execution parameter, decoding and extraction are not touched.

## 8. Execution state

**Blocked.** The GPU driver is broken: kernel module 620.70.06 against
userspace 620.73, `nvidia-smi` fails with "Driver/library version mismatch",
and there is no root to reload it. No `scripts/court --gpu`, no CUDA call and no
model load may run until the operator reloads the driver.

Everything in §1–§4 was computed on CPU through `scripts/court` with explicit
limits; no model was loaded.

## 9. What Stage 0 does not do

* It changes **no win bar**. The registered test bars stay MATH-500 ≥ 50.0 and
  full MATH ≥ 50.0 with paired intervals above zero, GSM8K ≥ 62.38 and
  MMLU ≥ 51.07 (`docs/S71-LITERATURE.md` H.4).
* It scores **no test item** and reads no test score (§4 disclosure is the only
  test-file access, and it is a disjointness filter).
* It makes **no capability claim**. A qualified teacher is a permission to
  register Stage 1, not a result.
* MATH-500 and full MATH are **two scales of one test set, not independent
  replication**: MATH-500 is a 500-problem subset of the 5,000-problem
  Hendrycks MATH test split, and the full-corpus gold agrees with the MATH-500
  answer field on all 500 overlapping problems (`docs/S67-RESULT.md` §3).

## 10. What would make this plan wrong

1. **The 70/90 gate may be mis-specified for MATH.** MATH-train is easier than
   MATH-500, so a 70% train gate can pass a teacher that sits at ~66% on the
   test. If T1 passes at exactly 70–72%, the pass is real but weak and should be
   reported as such; the gate must not then be raised post hoc.
2. **The sample is one draw.** 250 items give a Wilson half-width near 5.7
   points at p = 0.70. A 0.696 fail and a 0.704 pass are not distinguishable.
   That is disclosed, not corrected: the gate is fixed, and a reader should read
   any near-threshold outcome as unresolved rather than decisive.
3. **The 2B may be the wrong control for the ceiling argument.** A 2B at 65.80
   and an 0.8B at 40.60 on the same instrument leaves a 25-point gap that a
   stronger teacher might or might not shrink. Stage 0 can only close the
   teacher-quality half of that question; the student-capacity half is G6/G8.
