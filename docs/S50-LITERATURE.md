# S50 literature: second-moment precision and linear absmax quantisation

Recorded 2026-11-17. The structured receipt with the quoted figures, URLs and
verification notes is `results/stage2/s50-literature-20261117T210432Z.json`;
this page is the readable index and deliberately defers performance figures to
that receipt.

Question: was the pre-S50 construction — a linear absmax int8 code for the Adam
second moment, one scale per 256-element block — a published construction, and
is there published evidence that it fails?

## Sources

| Source | Date | Claim (summary) | Status |
|---|---|---|---|
| Dettmers, Lewis, Shleifer, Zettlemoyer, *8-bit Optimizers via Block-wise Quantization*, ICLR 2022, <https://arxiv.org/abs/2110.02861> | submitted 2021-10-06; v2 2022-06-20 | 8-bit optimizer states need block-wise **dynamic** (non-linear) quantisation because the second Adam state spans several orders of magnitude. The linear-quantisation ablation is dominated by unstable runs; the dynamic variant is not. Block-wise isolation of outliers is separately required at scale. | **VERIFIED** (primary text, Table 3, Appendix F) |
| Tang, Gan, Awan, Rajbhandari, Li, Lian, Liu, Zhang, He, *1-bit Adam*, <https://arxiv.org/abs/2102.02888> | submitted 2021-02-04; v2 2021-06-29 | Adam's variance is the non-linear term. Error-compensated compression cannot correct its history error. The published recipe freezes the variance in high precision after warmup and compresses only the momentum; compressing the variance itself fails to converge. | **VERIFIED** (primary text §4.2, §7.2, supplementary Fig. 12) |
| Dettmers, Lewis, Belkada, Zettlemoyer, *LLM.int8()*, NeurIPS 2022, <https://arxiv.org/abs/2208.07339> | submitted 2022-08-15; v2 2022-11-10 | Per-tensor linear absmax fails when systematic outlier features are present; the published fix is vector-wise quantisation plus a mixed-precision decomposition. Published evidence that linear absmax collapses the small-magnitude majority in the presence of an outlier. | **PARTIAL** (demonstrated for activations, not optimizer moments) |
| Ramesh et al., *Zero-Shot Text-to-Image Generation*, <https://arxiv.org/abs/2102.12092> | submitted 2021-02-24 | 16-bit Adam optimizer states were reported unstable for a large model; stable embedding, tensor-wise scaling constants for both Adam states and loss scaling were required. | **PARTIAL** (quoted from Dettmers et al. §5; primary appendix not read here) |
| Shazeer, Stern, *Adafactor*, ICML 2018, <https://arxiv.org/abs/1804.04235> | submitted 2018-04-11 | The standard memory saving for the second moment factorises it as a row-column outer product; it changes rank, not the numeric precision of the statistic. | **PARTIAL** (described by Dettmers et al. §5; primary paper not read here) |
| Exact per-256 linear absmax int8 code on `v` | — | No primary source evaluates this exact construction; the published linear ablations use different block size and 32-bit embedding states. The S50 reproduction is therefore our own measurement, not a citation. | **UNSUPPORTED** as a published result; reproduced locally |

## Reading

- The published 8-bit Adam does **not** use a linear absmax code. It uses a
  non-linear dynamic code, whose stated purpose is precision at both large and
  small magnitudes, and its linear ablation is unstable. The pre-S50 module's
  docstring claimed to follow that paper while implementing the configuration
  the paper rejects.
- The second moment is treated separately from the first in the published
  work, both by Dettmers et al. (different code for a strictly positive,
  wide-range statistic) and by 1-bit Adam (variance kept high-precision and
  frozen). The S50 fix follows the same separation: the first moment stays
  int8, the second moment is bf16.
- The remaining S50 parity error is attributed to the int8 first moment, not
  the second: see `results/stage2/s50-parity-components-20261117T203840Z.json`.
- None of these sources measures the full-backbone distillation arm. They
  establish the mechanism and the published direction; the reproduction and
  the 200-step revalidation establish what this optimiser did.

## What is not claimed

- That bf16 is universally sufficient for a second moment. It is sufficient
  for this arm's measured peak and it has a full exponent range; the receipted
  200-step run is the evidence for this recipe, not a general theorem.
- That the S50 defect was the only instability. The corrected optimiser at
  `lr = 1e-4` did not trip the update tripwire but still raised fixed-batch
  teacher KL; that rate was rejected on the KL gate. The defect explains the
  catastrophic updates, not every hyperparameter sensitivity.
