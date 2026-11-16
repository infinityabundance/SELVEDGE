# T2 diagnostics — what `H*`'s target measures, and what it should

Status: **the artifact this document was waiting for now exists, and the tool has
been run on it.** The measurement below is the P12.2 one, not a new diagnostic.
The claim in §2 that "there is no utility label artifact in `results/` at this
revision" was true when written and is now superseded: `H*`'s Phase U produced
`results/artifacts/train/t2/Hstar/utility-labels.pt`, whose sha256 sidecar is
`9e41d604…`, and `oracle/eval/t2_diagnostics.py` ran on it, writing
`results/t2/diagnostics-20261116T070700Z.json` (stamp `20261116T070703Z`,
revision `41e4a7e1bbeaf59c7b75bdcf28b19acbdfcf461e`). That receipt is the
registered output; the numbers live in it, not here.

Two readings of that receipt, quoted with the stamp and revision that produced
them, because they decide the successor:

- `margin_over_noise` is `0.041129` (summary mean over the 28 layers). It is far
  below one, so **decision rule 1 of §5 fires**: the fourth/fifth boundary is
  inside the label noise, and changing the head cannot fix the target. This is
  the measured reason P12.17 part 2 (absolute utility, noise-calibrated
  temperature) is not optional.
- `router_identity_bias_share` is `0.003632`, so **decision rule 3 fires the
  other way**: routing is token-feature driven and the identity table is not the
  confound. P12.17's offset term is therefore kept for the magnitude it
  restores, not because routing is a constant.

The deployed-budget comparison (part (a) of §2, and P12.17 part 1) remains
unmeasured: the receipt still reports `deployed_budget_utility_available: false`
and names the five-per-layer pass recipe it needs. The interaction term of
P12.17 part 3 is also unmeasured. Nothing in this note is evidence that the
successor target works.

Original status line follows, unchanged for the record.

Status: **tooling implemented and CPU-tested; the measurement itself is blocked
on `H*`'s label artifact, which does not exist yet.** Nothing in this document is
a measured gap. It defines the three diagnostics precisely, says which of them
the label artifact can answer, names the ones that need new passes, and
registers the corrected target for the successor experiment (`P12.17` in
`docs/PROGRAM-PLAN.md`). The tool is `oracle/eval/t2_diagnostics.py`; its tests
are `oracle/tests/test_t2_diagnostics.py`, run on fixtures whose answers are
known by construction.

## 1. The target as registered

For a calibration position `t`, layer `l` and specialist `e` (T2.5):

```
u_e(t) = L_{-e}(t) - L_{+all}(t)
```

`L_{+all}` runs the layer's **full bank** (`full_bank=True`, every specialist
present, unit weight); `L_{-e}` is the same loss with `e` masked out and every
other specialist still present. The router is trained by a listwise KL to
`softmax((u - mean_e(u)) / std_e(u))` at temperature one, on cached pre-expert
hidden states, and then frozen. Selection at inference is the hard top-`4` of
the router logits.

Three properties follow, and each is a question, not a defect:

1. the label is measured with **eight** specialists present, but the deployed
   read uses **four**;
2. the target **standardises per expert**, so the absolute utility magnitude —
   nats of loss — is deleted from what the router is fitted to;
3. `u_e` is the marginal value of *omitting* `e` from a full bank, while the
   deployed decision is about which experts to *add* to a top-4 set.

## 2. What each gap needs, and what is available

| gap | what it needs | available in this session | computed by |
| --- | --- | --- | --- |
| (a) full-bank LOO versus deployed top-4 | new label passes under the router's own top-`4` mask | **no** — needs H\*'s bank and the GPU | not the label artifact; recipe in §4 |
| (b) per-expert standardisation | the label artifact only | no artifact on disk yet | `t2_diagnostics.py` `kl_absolute_vs_relative_*`, `label_mean`, `label_std` |
| (c) identity (bias) versus token features | the label artifact plus the router weights | no artifact yet; the router weights can be read from the step checkpoint | `t2_diagnostics.py` `router_identity_bias_share`, `router_top1_is_identity_winner`, overlaps |

There is **no** utility label artifact in `results/` at this revision: `H*` has
not started, so `Phase U` has not run. Reporting a gap number now would be
invention, so this document reports none and the tool exits non-zero with the
reason when the artifact is missing.

## 3. The three diagnostics, defined

Every statistic below is deterministic and per layer, with a percentile
bootstrap over calibration positions (resamples and seed are in the receipt).

**(b) What standardisation removes.**

* `kl_absolute_vs_relative_mean` — `KL(softmax(u) || softmax(u_std))`, averaged
  over positions. This is exactly the target change caused by per-expert
  standardisation: zero means the normalisation is an affine map that the
  softmax ignores, large means the router is being fitted to a different
  question than the utility answers.
* `label_mean`, `label_std` — the per-expert mean and spread in nats. The
  standardised target deletes the mean *and* rescales each expert.
* `first_component_energy_share` — the share of the per-token utility matrix's
  variance explained by its first principal component. Near one, the utilities
  are dominated by a single **token-level** factor, and the listwise softmax is
  mostly ranking tokens rather than choosing among experts.
* `mean_pairwise_utility_correlation`, `mean_absolute_pairwise_utility_correlation`
  — co-movement of experts' labels across positions. Location-invariant, so it
  separates "the experts differ in level" (correlation near zero, means apart)
  from "the experts share a factor" (absolute correlation near one).

**(c) Identity versus token features.** The router has no bias, and its logit
splits exactly:

```
z_e(t) = w_e . h_t = w_e . mean_t(h)  +  w_e . (h_t - mean_t(h))
         \_____identity_____/           \______token-varying______/
```

* `router_identity_bias_share` — the variance share of the identity term,
  `var_e(constant) / (var_e(constant) + mean_e var_t(varying))`. One means
  routing is a constant table; zero means the token features drive it.
* `router_top1_is_identity_winner`, `router_top4_overlap_with_identity` — how
  often the router's actual choice is the identity winner's.
* `router_top4_overlap_with_utility` — how often the router's top-4 equals the
  utility's own top-4, i.e. the utility-oracle agreement on the calibration set.
* `router_additive_gap_mean` — `sum_{e in top4(u)} u_e - sum_{e in top4(z)} u_e`,
  the loss in additive utility from routing on `z` instead of on `u`.

**(a) The deployed budget, as far as the labels allow.**

* `routing_margin_median` — `u_(4)(t) - u_(5)(t)` after ranking each token's
  utilities: the width of the decision boundary the router has to resolve.
* `label_noise_median` — the median per-expert label standard deviation.
* `margin_over_noise` — the ratio. **Below one, the four/five boundary is inside
  the label noise**, so no router can separate those experts on this slice and
  the target's own noise floor is the binding constraint, not the head.
* `router_least_harmful_share` — fraction of tokens whose whole chosen top-4 has
  non-negative marginal utility. A negative `u_e` means adding `e` makes the
  loss *worse*, so a set containing one is a "least harmful omission" choice.
* `router_chosen_negative_marginal_pair_share` — the same property counted per
  chosen pair.
* `deployed_budget_utility_available` — `false`, in every receipt written before
  the passes in §4 exist. It is a field so a reader can tell "measured" from
  "not measured".

## 4. The gap that needs new passes, and its exact recipe

The rank correlation P12.2 asks for — full-bank `u_e` versus deployed-budget
`u^dep_e` — cannot be derived from the full-bank labels: `L(S \ {e})` for the
router's own set `S` was never computed. The pass count is the same shape as the
registered labelling, so it is cheap on the same slice (`32` windows, `32,768`
tokens):

* `1` pass with `S(t)` (the frozen router's top-4 per position);
* `k = 4` passes with `S(t) \ {e}`, one per selected expert;
* five passes per layer (`1 + k`), so five times the `28` routed layers —
  against the `1 + m` passes per layer the full-bank set costs. This is
  arithmetic from the registered constants, not a measurement.

The measurement is run as part of `P12.1a`/`P12.17`'s label step, not here,
because it needs the trained bank and the GPU the pilot owns. When the artifact
exists:

```bash
scripts/court oracle -- python3 oracle/eval/t2_diagnostics.py \
    --labels results/artifacts/train/t2/Hstar/utility-labels.pt \
    --checkpoint results/artifacts/train/t2/Hstar/step-000064.pt \
    --out results/t2/diagnostics-$(date -u +%Y%m%dT%H%M%SZ).json
```

## 5. Decision rules, fixed before the measurement

These are written now so the diagnostic cannot be read backwards later.

1. **Separability.** If `margin_over_noise < 1` at the median layer, the
   listwise target is asking for a distinction the labels cannot support, and
   the successor must change the *reference set* (deployed budget, fewer
   alternatives) or the *temperature* (measured noise) rather than the head.
   If it is comfortably above one, the target is separable and a routing failure
   is a fitting failure.
2. **Least harmful omission versus most useful addition.** If
   `router_least_harmful_share` is well below one while
   `router_top4_overlap_with_utility` is high, the router is doing what the
   target asks and the target is what is wrong (it rewards the least harmful
   omission). If the overlap is low while the share is high, the head is the
   problem.
3. **Identity.** If `router_identity_bias_share` is near one, "routing" is a
   constant per-layer table and any token-level interpretation of the arm's
   result is unsupported. If it is near zero, the router uses the features and
   the identity term is not a confound.
4. **Common factor.** If `first_component_energy_share` is near one, the
   utilities are one token-level factor plus noise; an additive top-4 target can
   then only rank tokens, and the interaction term of `P12.17` is required, not
   optional.
5. **Standardisation.** If `kl_absolute_vs_relative_mean` is small, dropping the
   per-expert standardisation changes little and the simpler target is
   preferred. If it is large, the normalisation is deleting the quantity the
   deployed decision reads.

Thresholds: the qualitative comparisons above are the registration; the numeric
values that will be compared are reported by the receipt, and no threshold on
`margin_over_noise` other than one is invented here.

## 6. The corrected target for the successor (registered as `P12.17`)

`P12.17` in `docs/PROGRAM-PLAN.md` carries the definition. In one line: labels
measured **within the deployed top-`k` set**, the router given a **learned
per-expert offset initialised from the measured mean utility** so the absolute
magnitude is retained, the temperature set to the **measured label noise**
instead of one standardised unit, and the selection scored by an
**interaction-aware set value** `U(S) = sum_{e in S} a_e + half sum_{e != e'} I_{e,e'}`
whose interaction term `I` is measured and reported. Each part is conditional on
the diagnostic that motivates it, so the successor can drop a part the labels
show is unnecessary.

## 7. What this document does not establish

* No gap is measured. The numbers arrive with `H*`'s artifact and its receipt.
* The diagnostics are estimators, not proofs: they are tested against fixtures
  whose answers are known, which shows the code computes what is written here,
  not that the registered target is wrong in this run.
* The interaction term in `P12.17` is a proposal with no measurement yet; it is
  registered before the successor runs, not after.
