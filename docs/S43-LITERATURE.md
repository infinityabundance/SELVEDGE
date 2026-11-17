# S43 focused literature — math mixtures, Nemotron-CC-Math, on-policy solutions, token gates, advantage weighting

Status: **focused verification, 2026-11-17 (S43).** Each entry names the URL, the
fetch date, the claim the source itself states, and a verdict for the question
this session had to decide. `VERIFIED` means the claim is in the fetched abstract
or record; `PARTIAL` means the source supports part of the claim or a different
setting; `UNSUPPORTED` means no source was located that states the claim. An
abstract does not settle a full paper, and the verdicts are scoped to what was
read.

The session had to decide five things:

1. whether a math-heavy pretraining/distillation mixture is a published lever;
2. what Nemotron-CC-Math is, and whether it was usable here;
3. whether on-policy work on GSM8K-format solutions is established;
4. whether a token-conditioned gate differs from a global scalar in the
   literature;
5. whether advantage-weighted distillation beyond KL is established.

## 1. Math-heavy pretraining and distillation mixtures

| source | fetched | claim as stated | verdict |
| --- | --- | --- | --- |
| Azerbayev et al., *Llemma: An Open Language Model For Mathematics*, `https://arxiv.org/abs/2310.10631` (2023; v3 2024) | 2026-11-17 | continues pretraining "on the Proof-Pile-2, a mixture of scientific papers, web data containing mathematics, and mathematical code"; the resulting Llemma "outperforms all known open base models" on MATH | **VERIFIED** for the mixture as a lever at continued-pretraining scale. **PARTIAL** for this session: the mixture is billions of tokens and the objective is next-token pretraining, not 20M-token online distillation into a frozen-backbone arm. |
| Wang, Li, Xia, Liu, *MathPile: A Billion-Token-Scale Pretraining Corpus for Math*, `https://arxiv.org/abs/2312.17120` (NeurIPS 2024) | 2026-11-17 | about 9.5B math-centric tokens; "data contamination detection on downstream benchmark test sets to eliminate duplicates"; continual-pretraining experiments "booting the performance on common mathematical reasoning benchmarks" | **VERIFIED** for a curated math corpus plus contamination detection improving math benchmarks. **PARTIAL** for us: different scale and no distillation objective; it does support the contamination discipline we applied. |
| Shao et al., *DeepSeekMath*, `https://arxiv.org/abs/2402.03300` (2024) | 2026-11-17 | continues pretraining with "120B math-related tokens sourced from Common Crawl, together with natural language and code data" and reaches 51.7% on MATH | **VERIFIED** that a large math-heavy mixture with general data is a published lever. **PARTIAL** for this session: 120B tokens against 20M, and a different base. |
| Paster et al., *OpenWebMath* dataset card, `https://huggingface.co/datasets/open-web-math/open-web-math` (paper `https://arxiv.org/abs/2310.06786`) | 2026-11-17 | 6.3M documents, 14.7B tokens of filtered mathematical web text, ODC-By 1.0 | **VERIFIED** as a dataset record. It was a candidate substitute for Nemotron-CC-Math; NuminaMath-CoT was chosen instead because it carries worked solutions in the scored formats, not only web prose. |

**Decision.** A math-heavy mixture is a published lever, at pretraining scale
(billions of tokens). No located source tests **57% math in a 20M-token online
distillation** into a frozen-backbone student, so the stream is a local
registration with a measured target, not a reproduction. **Verdict for the exact
setting: PARTIAL.**

## 2. Nemotron-CC-Math

| source | fetched | claim as stated | verdict |
| --- | --- | --- | --- |
| Karimi Mahabadi et al., *Nemotron-CC-Math: A 133 Billion-Token-Scale High Quality Math Pretraining Dataset*, `https://arxiv.org/abs/2508.15096` (2025) | 2026-11-17 | a Common Crawl math corpus with layout-aware extraction preserving MathJax/KaTeX/MathML structure; "Nemotron-CC-Math-3+ (133B tokens) and Nemotron-CC-Math-4+ (52B tokens)"; pretraining a Nemotron-T 8B model yields "+4.8 to +12.6 gains on MATH and +4.6 to +14.3 gains on MBPP+" and improves MMLU/MMLU-Stem | **VERIFIED** for the corpus and its reported gains. **PARTIAL** for this session: the gains are from full pretraining of an 8B model on 52–133B tokens, not from 20M distillation tokens. |
| dataset access, `https://huggingface.co/datasets/nvidia/Nemotron-CC-Math-v1` | 2026-11-17 | the dataset card is gated (`gated: auto`, NVIDIA data agreement); an unauthenticated request for `3/part_000000.parquet` returns HTTP 401 `GatedRepo` | **VERIFIED** as an access fact. No byte was used. NuminaMath-CoT (Apache-2.0) is the declared substitute in `docs/ws19-full-distillation-amendment-math-stream-preregistration-v1.json`. |

**Decision.** Nemotron-CC-Math is the right *kind* of source and the wrong
*access* for this session. The substitution is recorded, not silent; the
manifest names the failed revision and the substitute's own revision, licence
and SHA-256.

## 3. On-policy distillation on GSM8K-format solutions

| source | fetched | claim as stated | verdict |
| --- | --- | --- | --- |
| Agarwal et al., *On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes* (GKD), `https://arxiv.org/abs/2306.13649` (ICLR 2024) | 2026-11-17 | static KD "suffer[s] from distribution mismatch"; GKD "trains the student on its self-generated output sequences by leveraging feedback from the teacher", with alternative losses and integration with RL fine-tuning; evaluated on "arithmetic reasoning" among other tasks | **VERIFIED** for on-policy student sequences with teacher feedback on arithmetic reasoning. **PARTIAL** for this session: the registered stream is teacher-forced worked solutions, not student-generated rollouts, so the arm is off-policy relative to GKD. |
| Gu, Dong, Wei, Huang, *MiniLLM: On-Policy Distillation of Large Language Models*, `https://arxiv.org/abs/2306.08543` (ICLR 2024) | 2026-11-17 | replaces forward KL with reverse KL "to prevent the student from overestimating the low-probability regions of the teacher distribution" and derives an on-policy optimisation for it | **VERIFIED** for reverse-KL/on-policy distillation. **PARTIAL** for us: our objective is forward KL(teacher‖student) on a fixed stream, with a corrective residual rather than a full student. |

**Decision.** On-policy feedback on self-generated solutions is published and
is the natural next route if the off-policy GSM8K/MATH solutions fail to move
the family. It is **not** what this session trains: the stream supplies teacher-
forced solutions and the teacher is queried online on those tokens.

## 4. Token-conditioned gates versus global gates

| source | fetched | claim as stated | verdict |
| --- | --- | --- | --- |
| Srivastava, Greff, Schmidhuber, *Training Very Deep Networks* (Highway Networks), `https://arxiv.org/abs/1507.06228` (NeurIPS 2015) | 2026-11-17 | "adaptive gating units to regulate the information flow"; even hundreds of layers train with gradient descent | **VERIFIED** that adaptive gates on a residual path are standard. **PARTIAL** for us: the gate is a learned transform of the layer input, not a correction gate trained by a teacher. |
| Chen et al., *Dynamic ReLU*, `https://arxiv.org/abs/2003.10027` (ECCV 2020) | 2026-11-17 | "parameters [of the rectifier] are generated by a hyper function over all input elements"; "adapts the piecewise linear activation function accordingly" with negligible extra cost | **VERIFIED** that an input-dependent (per-element) gate/activation differs from a static one and is cheap. **PARTIAL** for our site-level residual correction. |
| Fedus, Zoph, Shazeer, *Switch Transformers*, `https://arxiv.org/abs/2101.03961` (JMLR 2022) | 2026-11-17 | per-token routing selects different parameters "for each incoming example"; trained in bfloat16 | **VERIFIED** that per-token gating is a mainstream construction. **PARTIAL** for our scalar-to-token change: this is expert routing, not a residual magnitude. |
| search for a token-conditioned *corrective memory* gate trained on a teacher advantage | 2026-11-17 | no located record states `alpha_t = tanh(w·[h_t;r_t] + b)` as a correction gate with zero initialisation and teacher-advantage supervision | **UNSUPPORTED** for the exact construction. |

**Decision.** Token conditioning is not novel in general; it is the natural
generalisation of the scalar gate and costs `2*hidden+1` elements per site. The
exact corrective-memory form is registered as a local construction with a
decision rule (P19.2b/P19.8 amendment at S43).

## 5. Advantage-weighted distillation beyond KL

| source | fetched | claim as stated | verdict |
| --- | --- | --- | --- |
| Nair, Gupta, Dalal, Levine, *AWAC: Accelerating Online Reinforcement Learning with Offline Datasets*, `https://arxiv.org/abs/2006.09359` (2020; v6 2021) | 2026-11-17 | combines "sample efficient dynamic programming with maximum likelihood policy updates"; the actor is advantage-weighted (the paper's advantage-weighted actor-critic) and uses prior data without behaviour cloning | **VERIFIED** that weighting a maximum-likelihood update by an advantage is a published objective. **PARTIAL** for us: the advantage is a critic's return advantage, not `log p_T(y) - log p_B(y)` on a supervised token. |
| Agarwal et al., GKD (above) | 2026-11-17 | "flexibility to employ alternative loss functions between the student and teacher, which can be useful when the student lacks the expressivity to mimic the teacher's distribution" | **VERIFIED** that KL is not the only registered distillation loss in the on-policy setting. **PARTIAL** for a signed log-probability advantage on a fixed token. |
| Zhang et al., *Reinforcement-aware Knowledge Distillation for LLM Reasoning* (RLAD), `https://arxiv.org/abs/2602.22495v3` (quoted from `docs/S42-LITERATURE.md` §4) | 2026-11-17 (S42) | "teacher supervision may not align with the student's evolving rollout distribution"; selective imitation "guiding the student toward the teacher only when it improves the current policy update" | **VERIFIED** for selective/advantage-conditioned imitation in RL post-training. **PARTIAL** for a frozen-base token-level weight. |
| search: `abs:"advantage-weighted" AND abs:"distillation"`, `abs:"signed advantage" AND abs:"teacher"` (`https://export.arxiv.org/api/query`) | 2026-11-17 | no located record uses `A_t = log p_T(y_t) - log p_B(y_t)` as the position weight of a corrective residual's KL, beside a KL-disagreement weight | **UNSUPPORTED** for the exact combination. |

**Decision.** Advantage-weighted maximum-likelihood is published (AWAC) and
selective imitation is published (RLAD/GKD); the specific signed log-probability
advantage on the supervised token, compared against the KL-disagreement weight
at equal tokens and parameters, is a local construction. It is registered with a
four-branch decision rule in
`docs/ws19-corrective-gate-advantage-preregistration-v1.json`.

## What this document does not establish

- It does not establish that a 57% math mixture at 20M tokens transfers the
  billion-token pretraining results; the scale differs by 3–4 orders of
  magnitude.
- It does not establish that NuminaMath-CoT is equivalent to Nemotron-CC-Math;
  it is a substitute for an unavailable gated dataset with a different
  distribution.
- It does not establish that the token gate or the signed advantage is better;
  both are registered before training with decision rules.
- It does not establish that on-policy self-generated solutions would help here;
  that is the next route if the registered off-policy stream misses.
- The arXiv abstract is not the full paper; the verdicts are scoped to what was
  fetched.
