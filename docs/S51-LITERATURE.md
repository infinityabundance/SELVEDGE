# S51 literature — first-moment precision, real gradients, sparse reads, eligibility

Recorded 2026-11-17 alongside the S50 index in `docs/S50-LITERATURE.md`. The
machine-readable receipt for this page is
`results/stage2/s50-literature-s51-20261117T224516Z.json`; the S50 receipt is
`results/stage2/s50-literature-20261117T210432Z.json`. Every row below names the
URL, the date, the specific claim used here and the verification status.
`VERIFIED` means the primary page/abstract was read in this session or the S50
pass; `PARTIAL` means the source is named but the claim was not quoted from the
page in this session; `UNSUPPORTED` means no primary source was found.

## First-moment int8 and dynamic quantisation

| source | URL | date | claim used | status |
| --- | --- | --- | --- | --- |
| Dettmers et al., *8-bit Optimizers via Block-wise Quantization*, arXiv:2110.02861 | https://arxiv.org/abs/2110.02861 | submitted 2021-10-06, v2 2022-06-20 | block-wise dynamic quantisation is precise at both magnitudes; linear quantisation is ~10x less accurate for the first state and unstable for the second (Table 3, Appendix F Table 6) | VERIFIED (S50 pass) |
| Tang et al., *1-bit Adam*, arXiv:2102.02888 | https://arxiv.org/abs/2102.02888 | 2021-02-04, v2 2021-06-29 | low-bit variance quantisation does not converge at n <= 8 bits; the variance needs high precision (Section 4.2, supp. Fig. 12) | VERIFIED (S50 pass) |
| Dettmers et al., *LLM.int8()*, arXiv:2208.07339 | https://arxiv.org/abs/2208.07339 | 2022-08-15, v2 2022-11-10 | per-tensor linear absmax fails with outlier features; the fix is vector-wise plus a 16-bit outlier decomposition | PARTIAL (carried from S50) |
| S51 search | — | searched 2026-11-17 | no primary source measures a per-256 linear absmax int8 **first** moment on real Qwen3.5-0.8B gradients versus fp32 AdamW | UNSUPPORTED (absence of evidence) |

Interpretation: the shipped hybrid (linear int8 first moment + bf16 second
moment) is a byte-cost compromise, not a published recipe. The 0.88% synthetic
parameter error (stamp `20261117T203858Z`, revision `547953f`,
`results/stage2/s50-parity-components-20261117T203840Z.json`) is not evidence
about real gradients, which is why the S51 harness captures real ones.

## Real-gradient versus synthetic optimiser validation

| source | URL | date | claim used | status |
| --- | --- | --- | --- | --- |
| Schaul, Antonoglou, Silver, *Unit Tests for Stochastic Optimization*, arXiv:1312.6055 | https://arxiv.org/abs/1312.6055 | 2013-12-20, v3 2014-02-25 | isolated synthetic unit tests are necessary but **not sufficient** for a generality/robustness claim | VERIFIED |
| Choi et al., *On Empirical Comparisons of Optimizers for Deep Learning*, arXiv:1910.05446 | https://arxiv.org/abs/1910.05446 | 2019-10-11, v3 2020-06-16 | optimizer comparisons are highly sensitive to the tuning protocol and search space; benchmark on the real training regime | VERIFIED |

Interpretation: the lognormal fixture is a necessary unit test; the real-gradient
fixture at the bar's own lr/betas/block is the sufficiency check. The harness
replays captured gradients cyclically, which tests the moment storage on the
captured spectrum but not data order; that limit is stated in the receipt.

## Sparse-gather memory execution and byte accounting

| source | URL | date | claim used | status |
| --- | --- | --- | --- | --- |
| Kwon et al., *PagedAttention* (vLLM), arXiv:2309.06180 | https://arxiv.org/abs/2309.06180 | 2023-09-12 | cache memory is allocated in fixed pages and shared within/across requests, so physical rows/pages are fewer than logical token requests | VERIFIED |
| Rajbhandari et al., *ZeRO-Infinity*, arXiv:2104.07857 | https://arxiv.org/abs/2104.07857 | 2021-04-16 | training state is partitioned across GPU/CPU/NVMe tiers; byte accounting is per tier | VERIFIED |
| PyTorch, `torch.nn.EmbeddingBag` | https://pytorch.org/docs/2.15/generated/torch.nn.EmbeddingBag.html | accessed 2026-11-17 | bags avoid materialising intermediate rows; `sparse=True` yields a sparse weight gradient; empty bags return zeros | VERIFIED |

Interpretation: `ReadAccounting` counts **unique rows** (via optional
`row_ids`), bytes per named tier, and store-reported cache hits/misses. A token
request is not a physical row request. `measured_cost(read)` is arithmetic on the
dense read tensor and remains a surrogate until the store path reports real bytes
and cache state; the docstring and `basis` string say so.

## Eligibility and provenance of invalidated results

| source | URL | date | claim used | status |
| --- | --- | --- | --- | --- |
| OpenLineage object model, v1.55.0 | https://openlineage.io/docs/spec/object-model | accessed 2026-11-17 | datasets carry `lifecycleStateChange`/`version`; runtime and design lineage are separate; revisions are metadata, not deletions | VERIFIED |
| Mitchell et al., *Model Cards for Model Reporting*, arXiv:1810.03993 | https://arxiv.org/abs/1810.03993 | 2018-10-05, v2 2019-01-14 | reported numbers must carry their evaluation conditions and intended-use context | VERIFIED |
| Pineau et al., *Improving Reproducibility in ML Research*, arXiv:2003.12206 | https://arxiv.org/abs/2003.12206 | 2020-03-27, v4 2020-12-30 | the reproducibility checklist requires code, data and workflow, not only the final metric | VERIFIED |
| MLflow Model Registry docs | https://mlflow.org/docs/latest/model-registry.html | accessed 2026-11-17 | versions are immutable; stages/aliases/tags label eligibility and stages are deprecated in favour of aliases/tags | PARTIAL (page returned no machine-readable text this session) |

Interpretation: invalidation is a **label over an immutable receipt**, never a
deletion or an edit. `scripts/status_consistency.py` derives
`currently_eligible` / `historically_scored` / `invalidated` from the
invalidation master record and the per-directory markers, and the court refuses
prose that cites an invalidated artefact as the current bar or baseline unless
the line labels it historical.
