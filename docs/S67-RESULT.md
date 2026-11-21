# S67 — three-seed delta soup, the registered MATH resolution, and the verdict

Status: **the registered capability win is met for the seed-delta soup.** GSM8K
replicates above zero, the pre-registered 5,000-problem MATH instrument resolves
the effect that n=500 left open (+2.52 points, interval [+1.20, +3.78]), and
retention passes on full MMLU, on the 12,042 never-used MMLU items, and on the
registered 4,000-item MMLU-Pro instrument. This is a capability win for the
interpolation-plus-soup intervention, **not** a frontier advantage, and it is
**not** a direct improvement over the S65 seed-0 checkpoint.

Every number below is quoted from the receipt named beside it, with that
receipt's stamp and the revision recorded inside it. No test split was an input
to any coefficient selection.

| artifact | receipt | stamp | revision |
|---|---|---|---|
| delta diagnostics | `results/stage2/s67-soup/delta-diagnostics.json` | `20261120T063347Z` | `ddcd187` |
| soup checkpoints | `results/stage2/s67-soup/soup-receipt.json` | `20261120T064304Z` | `44b0e13` |
| coefficient selection | `results/stage2/s67-soup/calibration.json` | `20261120T124459Z` | `0c71d9c` |
| resolution instrument (pre-registered) | `docs/s67-resolution-instrument-preregistration-v1.json` | `2026-11-20T06:39:40Z` | `fa4f51d` |
| soup GSM8K / MATH-500 / full MATH | `results/stage2/s67-final/soup-stage1.json` | in-receipt | `d532b79` |
| baseline full MATH | `results/stage2/s67-final/baseline-math-full.json` | in-receipt | `d532b79` |
| soup MMLU and MMLU-Pro | `results/stage2/s67-final/soup-retention.json` | in-receipt | `d532b79` |
| baseline MMLU-Pro 4,000 | `results/stage2/s67-final/baseline-retention.json` | in-receipt | `d532b79` |
| retention audit | `results/stage2/s67-final/retention-audit.json` | `20261121T025710Z` | `bacb13e` |
| final verdict | `results/stage2/s67-final/verdict.json` | `20261121T025730Z` | `bacb13e` |

## 1. The three deltas are compatible

Before averaging, each seed's update was measured:

    d_s = theta_s - theta_0

with `theta_0` the post-trained Qwen3.5-0.8B endpoint and `theta_s` the three
S66 full-arm step-4,000 checkpoints (seed 0 is the S58 run reused as seed 0).
Over all 498,113,344 trainable text-tower elements:

| pair | cosine |
|---|---|
| seed 0 – seed 1 | 0.3205 |
| seed 0 – seed 2 | 0.3195 |
| seed 1 – seed 2 | 0.3210 |

The delta norm is 13.42 (base norm 361.99, so a 3.7% relative update) and is
almost identical across seeds (13.4170 / 13.4162 / 13.4170). Per layer the
cosines sit near the global value (e.g. layer 00: 0.322/0.321/0.324; layer 20:
0.316/0.315/0.316). Two non-layer entries (norms/biases) are slightly negative
(−0.0496 and −0.0012) and the smallest single tensor is
`layers.22.linear_attn.dt_bias` at −0.64, but that tensor is tiny and its layer
aggregate is positive. **No layer falls below the pre-declared −0.10
anti-correlation threshold**, so the registered soup route was taken rather
than the seed-0 fallback.

## 2. The soup and its coefficient

    delta_bar      = mean_s (theta_s - theta_0)
    theta_soup(alpha) = theta_0 + alpha * delta_bar

mixed in float32 and rounded back to bf16, over the same 498,113,344 elements.
The tied embedding/output projection is frozen and inherited, so the certified
unique active text path stays **752,393,024** elements.

The averaged delta is smaller than any single seed's (`||delta_bar||` is about
0.74 of `||d_s||` at the observed cosines), so the coefficient grid was shifted
up: `0, 0.375, 0.5, 0.625, 0.75`.

Selection used the registered 1,000-item GSM8K-cal and MATH-cal train splits
(`docs/s66-calibration-splits-v1.json`) and measured 2,000-item calibration
MMLU as the retention filter:

| alpha | GSM8K-cal | MATH-cal | mean gain vs 0 | MMLU-cal | eligible |
|---|---|---|---|---|---|
| 0.000 | 52.20% | 32.20% | — | 50.60% | — |
| **0.375** | **60.90%** | **32.60%** | **+4.55** | **52.15%** | **yes, selected** |
| 0.500 | 60.70% | 30.70% | +2.50 | 52.10% | yes |
| 0.625 | 60.30% | 29.80% | +1.85 | 51.05% | yes |
| 0.750 | 57.60% | 27.30% | −1.35 | 50.95% | yes |

Every alpha keeps calibration MMLU within the 1.0-point drop bound, so the
selection is the calibration gain alone: **alpha = 0.375**. No test split was
read. The `alpha = 0` row is the S66 large-grid row (same model, same protocol,
same registered splits), reused only after a tensor-for-tensor equality check
against the soup's own `alpha = 0` checkpoint.

For context: re-running the *single-seed* seed-0 grid at 1,000 items selected
0.375 as well, but the 300-item subset alone would have selected 0.25, so the
enlarged split changed that stability check (`stable=false` in
`results/stage2/s66-calibration/large-grid.json`). That is not a re-selection of
the soup coefficient.

## 3. The resolution instrument was registered before scoring

`docs/s67-resolution-instrument-preregistration-v1.json` registers the full
Hendrycks MATH test split — 5,000 problems, revision
`21a5633873b6a120296cce3e2df9d5550074f4a3` — as the single-use resolution
instrument, with the same S54 protocol as MATH-500 (greedy, non-thinking,
`max_new_tokens=4096`, batch 32, paired problem-level bootstrap, 4,000
resamples, seed 53). Its gold rule (the last balanced `\boxed{}` in the
reference solution) reproduces the MATH-500 answer field on all 500 overlapping
problems with zero disagreements (`results/datasets/math-test-fetch.json`).

The power arithmetic was fixed in the registration: the observed MATH-500
half-width is about 3.9 points at n=500, so the standard error falls by
sqrt(10) = 3.162 to about 0.63 at n=5,000 and the expected half-width is about
1.23. Any effect at or above about +1.8 points is resolved at 80% power. The
instrument was scored once, after the coefficient was selected, and is not
enlarged again if it had failed.

## 4. Test scores

Paired problem-level bootstrap, 4,000 resamples, seed 53, primary `flexible`
basis (the receipt's registered extraction). Four decimals are quoted from the
verdict receipt; the percentages in the table are the same numbers.

| task | n | post-trained | soup | delta | 95% interval | above zero |
|---|---|---|---|---|---|---|
| GSM8K | 1,319 | 54.21% | **63.38%** | **+9.17** | **[+6.60, +11.98]** | **yes** |
| MATH-500 | 500 | 38.20% | 40.60% | +2.40 | [−1.60, +6.40] | no |
| **full MATH** | **5,000** | 39.60% | **42.12%** | **+2.52** | **[+1.20, +3.78]** | **yes** |

The MATH-500 point estimate is positive but unresolved at n=500, exactly the
clause S66 could not close. The pre-registered 5,000-problem instrument resolves
it: the same direction, with an interval entirely above zero. MATH-500 is a
500-problem subset of the 5,000 and is reported separately, not pooled.

### 4.1 Direct comparison with the S65 seed-0 checkpoint

| task | soup | S65 seed-0 | soup − seed-0 | 95% interval |
|---|---|---|---|---|
| GSM8K | 63.38% | 64.90% | −1.52 | [−3.64, +0.53] |
| MATH-500 | 40.60% | 42.80% | −2.20 | [−6.00, +1.41] |

**The soup does not beat the seed-0 checkpoint directly.** Both point estimates
are slightly lower and both intervals include zero. The seed-0 checkpoint was
never scored on the 5,000-problem instrument, so no full-MATH direct comparison
is claimed.

## 5. Retention

The registered bound is a drop bound: point drop ≤ 1.0 point and paired interval
lower bound ≥ −1.0 point; an improvement is inside the bound.

| instrument | n | post-trained | soup | delta | 95% interval | verdict |
|---|---|---|---|---|---|---|
| full MMLU | 14,042 | 50.48% | 52.07% | **+1.59** | **[+0.98, +2.16]** | pass |
| MMLU, 2,000 selection items | 2,000 | 50.65% | 52.20% | +1.55 | [−0.10, +3.15] | pass |
| MMLU, 12,042 unused items | 12,042 | 50.46% | 52.05% | **+1.59** | **[+0.97, +2.26]** | pass |
| **MMLU-Pro 4,000** | 4,000 | 23.43% | **26.75%** | **+3.33** | **[+2.00, +4.62]** | **pass** |

The full-MMLU improvement is reproduced on the items selection never read. The
4,000-item MMLU-Pro instrument — the registered one, not the retired 504-item
sample — is for the first time actually present for both models, and it passes
with an interval above zero. The 2,000-item selection partition's interval
includes zero, which is expected at that sample size; the claim rests on the
full set and on the 12,042 unused items.

## 6. Wall-clock and throughput

Quoted from the verdict receipt (`20261121T025730Z`, revision `bacb13e`), same
engine for both models (transformers 5.23.0 / torch 2.15.0+cu132, bf16, SDPA,
batch 32, one RTX 4080 SUPER).

| model | task | items | wall (s) | generated tokens | tokens/s |
|---|---|---|---|---|---|
| soup | GSM8K | 1,319 | 3,149.4 | 825,698 | 262.6 |
| soup | MATH-500 | 500 | 1,324.2 | 959,895 | 726.0 |
| soup | full MATH | 5,000 | 13,495.5 | 9,006,620 | 668.4 |
| post-trained | full MATH | 5,000 | 13,879.2 | 9,651,726 | 696.4 |

## 7. Verdict

**(a) Does the soup beat the S65 seed-0 checkpoint directly?** No. It is 1.52
points below on GSM8K and 2.20 below on MATH-500, and both intervals include
zero. The soup is a variance-reduction route with a different coefficient, not
an improvement over the best single seed.

**(b) Does it beat post-trained 0.8B on GSM8K and MATH?** Yes on GSM8K
(+9.17, interval above zero) and yes on the full 5,000-problem MATH test (+2.52,
interval above zero). MATH-500 alone remains unresolved at n=500 (+2.40,
interval includes zero) and is reported as such, not as a win.

**(c) Is retention satisfied on the 4,000-item MMLU-Pro?** Yes: +3.33 points
with interval [+2.00, +4.62], alongside a +1.59-point MMLU improvement that
survives on the 12,042 never-used items.

The three distinctions the registration asked for:

* **replicated GSM8K win** — yes, established by the three S66 seeds and
  unchanged here; the soup's own GSM8K interval is also above zero.
* **MATH effect resolved at larger n** — yes, on the pre-registered 5,000-item
  instrument, scored once.
* **complete registered win** — **met** under the registered definition, for
  this soup, on this single scored run.

## 8. What this is not

* It is **not a frontier advantage**. No external system was compared, and the
  model remains a 0.8B text model.
* It is **not a multi-seed soup**. Three seeds were averaged; the soup itself
  was trained once and scored once. The GSM8K replication is seed-level (the
  S66 runs), not a soup-seed replication.
* It is **not an improvement over the single best seed**. See §4.1.
* It is **not a per-token executed-set audit**. The 752,393,024 count is the
  text-path accounting certificate for the declared path.
* It is **not tuned on test**. Alpha was selected on 1,000-item train-split
  calibration with MMLU-cal as a filter; the 5,000-problem instrument was
  scored once, afterwards.
