# S42 focused literature — bf16 update numerics, error compensation, corrective memory, base preservation

Status: **focused verification, 2026-11-17 (S42).** Each entry names the URL, the
fetch date, the claim the source itself states, and a verdict for the question
this session had to decide. `VERIFIED` means the claim is in the fetched abstract
or record; `PARTIAL` means the source supports part of the claim or a different
setting; `UNSUPPORTED` means no source was located that states the claim. An
abstract does not settle a full paper, and the verdicts are scoped to what was
read.

The session had to decide four things:

1. whether a bf16 weight updated in place is an implementation detail or a
   numerical defect, and what the published remedy is;
2. whether carrying the discarded fraction of an update (an error-compensation
   buffer) is a recognised technique or a local invention;
3. whether a token-addressed memory can be trained by something other than
   reconstruction;
4. whether "do not move where the base is already right" is a published
   regulariser.

## 1. Stochastic rounding for low-precision training

| source | fetched | claim as stated | verdict |
| --- | --- | --- | --- |
| Gupta, Agrawal, Gopalakrishnan, Narayanan, *Deep Learning with Limited Numerical Precision*, `https://arxiv.org/abs/1502.02551` (ICML 2015) | 2026-11-17 | "we observe the rounding scheme to play a crucial role in determining the network's behavior during training"; "deep networks can be trained using only 16-bit wide fixed-point number representation when using stochastic rounding, and incur little to no degradation in the classification accuracy" | **VERIFIED**. The mechanism is stated for fixed point; it is evidence that the rounding scheme, not only the word width, decides whether a low-precision update trains. |
| Wang, Choi, Brand, Chen, Gopalakrishnan, *Training Deep Neural Networks with 8-bit Floating Point Numbers*, `https://arxiv.org/abs/1812.08011` (NeurIPS 2018) | 2026-11-17 | successful 8-bit floating-point training "through the introduction of a number of key ideas including chunk-based accumulation and floating point stochastic rounding"; arithmetic precision for "weight updates" reduced "from 32 bits to 16 bits" | **VERIFIED**. This is the closest published statement to our setting: the *weight update* is where stochastic rounding buys the precision back, and the accumulation is chunked. |
| Dettmers, Lewis, Shleifer, Zettlemoyer, *8-bit Optimizers via Block-wise Quantization*, `https://arxiv.org/abs/2110.02861` (ICLR 2022) | 2026-11-17 (S40) | "the first optimizers that use 8-bit statistics while maintaining the performance levels of using 32-bit optimizer states"; block-wise dynamic quantization | **PARTIAL** for this question. The abstract verifies 8-bit *moments* at 32-bit performance; it does not state stochastic rounding of the weight update, so our SR claim rests on the two rows above and the paper body, which this session did not read. |

**Decision.** A bf16 weight with no fp32 master copy and a nearest-even update is
below half an ulp at the registered learning rate, so the published finding
applies directly: the rounding scheme decides whether the update trains.
`Adam8bit` now rounds stochastically by default (`oracle/frontier/optim8bit.py`),
and the run receipt measures the bitwise unchanged fraction rather than assuming
the fix worked. **Verdict for the exact setting (bf16 weights, block-wise int8
moments, 1e-4 LR): PARTIAL** — no located source runs this exact combination, so
the local measured evidence (the S42 receipts) is the evidence.

## 2. Error compensation

| source | fetched | claim as stated | verdict |
| --- | --- | --- | --- |
| Karimireddy, Rebjock, Stich, Jaggi, *Error Feedback Fixes SignSGD and other Gradient Compression Schemes*, `https://arxiv.org/abs/1901.09847` (ICML 2019) | 2026-11-17 | "using error-feedback, i.e. incorporating the error made by the compression operator into the next step, overcomes these issues"; "EF-SGD with arbitrary compression operator achieves the same rate of convergence as SGD without any additional assumptions" | **VERIFIED**. This is the canonical error-feedback statement: carry the discarded part forward instead of dropping it. Our compensation buffer is the same idea in the weight-update domain. |
| De Sa, Leszczynski, Zhang, Marzoev, Aberger, Olukotun, Ré, *High-Accuracy Low-Precision Training*, `https://arxiv.org/abs/1803.03383` (2018) | 2026-11-17 | HALP "converges at the same theoretical rate as full-precision algorithms despite the noise introduced by using low precision"; the key ideas are SVRG and "a novel technique called bit centering to reduce quantization error" | **PARTIAL**. Bit centering is an error-reduction technique, not error feedback; it supports the premise (carry/centre the error rather than drop it) but is a different construction. |
| Wang et al., *8-bit Floating Point Numbers* (above) | 2026-11-17 | chunk-based accumulation and floating-point stochastic rounding for weight updates | **PARTIAL** for error compensation. It bounds the update by accumulating in chunks; it does not claim an explicit residual buffer. |

**Decision.** The registered *default* is stochastic rounding, not the
compensation buffer: SR is unbiased, needs no extra state, and the measured
sub-half-ulp regime is exactly what it fixes. The compensation buffer is
registered as an optional variant with its bytes measured (`1 bf16 per trainable
element`), because error feedback is the published technique but costs memory the
headroom may not have. **UNSUPPORTED** for the claim that an error-compensation
buffer is better than SR at this budget: that is a local experiment, not a
literature result.

## 3. Corrective and residual memory objectives

| source | fetched | claim as stated | verdict |
| --- | --- | --- | --- |
| Khandelwal, Levy, Jurafsky, Zettlemoyer, Lewis, *Generalization through Memorization: Nearest Neighbor Language Models*, `https://arxiv.org/abs/1911.00172` (ICLR 2020) | 2026-11-17 | kNN-LMs "extend a pre-trained neural language model (LM) by linearly interpolating it with a k-nearest neighbors model"; Wikitext-103 perplexity 15.79, "a 2.9 point improvement with no additional training"; "particularly helpful in predicting rare patterns" | **VERIFIED** for the gated residual memory and its long-tail benefit. **PARTIAL** for our question: the interpolation is with a scalar weight, there is no identity guarantee at the untrained point, and no advantage weighting. |
| NGM, *A Plug-and-Play Training-Free Memory Module for LLMs*, `https://arxiv.org/abs/2605.16893v2` | 2026-11-17 (S40) | causal n-gram encoder over the pretrained embedding space plus a cosine-gated memory injector, training-free | **PARTIAL**. A residual memory with a gate, but the gate is fixed by cosine similarity and nothing is trained. |
| Memory Grafting (as recorded in `docs/PRIOR-ART-AND-NOVELTY.md` §18 and `docs/S40-LITERATURE.md`) | 2026-11-17 (S35/S40) | a table built from a larger grafting model's frozen hidden states improves a 2.8B-scale average from 51.95/52.43 to 53.86 | **PARTIAL** and directly relevant to the mechanism we are replacing: the published graft is also a *representation* graft. Our measured +0.575 nats CE harm at 2M tokens is a local result that the published number does not contradict, because the budgets and sites differ. |
| search: `ti:"corrective memory" AND abs:transformer`, `abs:"residual memory" AND abs:"distillation"` (`https://export.arxiv.org/api/query`) | 2026-11-17 | no located record trains a token-addressed memory table by the teacher's advantage over a frozen base, with a zero-initialised correction gate | **UNSUPPORTED**. No source states the construction in `oracle/frontier/corrective.py`. |

**Decision.** Residual, gated memories are published; *training the residual on
the teacher's advantage instead of reconstructing the teacher's state*, with an
exact-identity initialisation, is not located in the checked sources. P19.2b is
therefore registered as a mechanism change with a local measured target, not as
a reproduction.

## 4. Base-preservation and distillation regularisers

| source | fetched | claim as stated | verdict |
| --- | --- | --- | --- |
| Li, Grandvalet, Davoine, *Explicit Inductive Bias for Transfer Learning with Convolutional Networks*, `https://arxiv.org/abs/1802.01483` (ICML 2018) | 2026-11-17 | fine-tuning has "no mechanism ... for retaining the features learned on the source task"; they "investigate several regularization schemes that explicitly promote the similarity of the final solution with the initial model" and recommend "a simple L² penalty with the pre-trained model being a reference" | **VERIFIED** for base preservation in *weight space*. **PARTIAL** for our term, which is in *output space* per token: penalise a KL increase where the base already matches the teacher. |
| Zhang, Jiang, Shen, Zhang, Ram, Yang, Tu, Xia, Soatto, *Reinforcement-aware Knowledge Distillation for LLM Reasoning*, `https://arxiv.org/abs/2602.22495v3` | 2026-11-17 | "teacher supervision may not align with the student's evolving rollout distribution, and the KL regularizer can compete with reward maximization"; RLAD performs "selective imitation ... guiding the student toward the teacher only when it improves the current policy update" | **VERIFIED** for the stated claim that unconditional KL can compete with what the student is already doing. **PARTIAL** for ours: RL post-training and advantage-ratio conditioning, not a frozen-base preservation penalty. |
| Lee, Kim, Lee, Chun, *ASKD-Whisper*, `https://arxiv.org/abs/2601.19919v2` | 2026-11-17 | static teacher dependence makes the student "inherit the teacher's domain-specific blind spots"; they decay teacher dependence during training | **PARTIAL**. Supports the premise that teacher signal can hurt where it is wrong; no per-token base-preservation term. |
| search: `abs:"base preservation" AND abs:"distillation"`, `abs:"KL increase" AND abs:"frozen model"` (`https://export.arxiv.org/api/query`) | 2026-11-17 | no located record penalises a per-token *increase over a frozen base's own* teacher-KL, weighted by where the base is already close | **UNSUPPORTED**. This is the specific form registered in `docs/ws19-corrective-graft-preregistration-v1.json`. |

**Decision.** Preserving the base is published in weight space (L²-SP) and
selective imitation is published in policy space (RLAD). Neither states the
per-token, output-space preservation weight registered for P19.2b, so that term
is a local construction with a registered decision rule, not a reproduction.

## What this document does not establish

- It does not establish that stochastic rounding is sufficient at every
  learning rate: the S42 measurement covers the registered schedule only.
- It does not establish that an error-compensation buffer is unnecessary: it is
  registered as an optional variant with its bytes measured, and the comparison
  is a local experiment.
- It does not establish that the corrective objective beats CE+KL, or that the
  preservation weight is well chosen: both are registered before scoring and the
  decision rule says what each outcome means.
- It does not establish that any published memory model is *incorrect* about
  reconstruction; only that the checked sources do not train a token-addressed
  table by the teacher's advantage with a base-preservation term.
