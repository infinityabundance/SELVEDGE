# S44 focused literature — signed-advantage distillation, token gates, preservation penalties, math mixtures at small budgets

Status: **focused verification, 2026-11-17 (S44).** Each entry names the URL, the
fetch date, the claim the source itself states, and a verdict for the question
this session had to decide. `VERIFIED` means the claim is in the fetched record;
`PARTIAL` means the source supports part of the claim or a different setting;
`UNSUPPORTED` means no source located states the claim. An abstract does not
settle a paper, and the verdicts are scoped to what was read. Where S43 already
read a source it is named rather than re-read, and says so.

The session had four questions:

1. is advantage-weighted — and specifically **signed**-advantage — distillation
   established against KL-only distillation;
2. is a **token-conditioned** gate established, as against a global scalar;
3. is a **preservation penalty** established for protecting a frozen base's
   existing behaviour;
4. what is known about **math-data mixtures at small token budgets**.

## 1. Signed-advantage and advantage-weighted distillation versus KL-only

| source | fetched | claim as stated | verdict |
| --- | --- | --- | --- |
| Sabbaghi, Pappas, Hassani, Javanmard, *When to Follow the Teacher: Advantage-Based On-Policy Distillation*, `https://arxiv.org/abs/2610.31563` (2026-10-22) | 2026-11-17 | "standard OPD trains the student to follow the teacher at every state… assuming that teacher guidance is always beneficial. We instead distinguish between what the teacher prefers and what actually helps the current student succeed… weights teacher actions by their counterfactual advantage under the student. Teacher actions that improve the student's expected outcome are reinforced, while those that do not are downweighted." | **VERIFIED** that advantage-weighted (rather than uniformly weighted) teacher supervision is a published objective with reported gains on math, coding and MedQA. **PARTIAL** for us: on-policy student rollouts with a counterfactual outcome advantage, not an off-policy token-level `log p_T(y_t) - log p_B(y_t)` on a supervised target inside a frozen-base residual. |
| Jia et al., *Asymmetric On-Policy Distillation: Bridging Exploitation and Imitation at the Token Level*, `https://arxiv.org/abs/2605.06387` (2026-08-30) | 2026-11-17 | "its standard advantage weighted policy gradient suffers from three structural weaknesses, including high variance updates, vanishing gradients in zero-advantage regions, and exploration bottlenecks when corrective signals are insufficient"; the method "replaces ineffective negative reinforcement with localized divergence minimization in non-positive advantage regions while preserving positive reinforcement learning", with "+4.09 / +8.34" average gains on mathematical reasoning under strong / weak initialisation | **VERIFIED** that advantage weighting alone has a named failure mode in the non-positive-advantage region and that the published fix is divergence minimisation *there* — which is structurally the same two-part design we register (signed weight **plus** a preservation penalty on the positions where the base already matches). **PARTIAL** for the exact mechanism: they minimise a divergence, we penalise any increase over the base's own KL with a per-token weight. |
| Zhang et al., *Reinforcement-aware Knowledge Distillation for LLM Reasoning* (RLAD), `https://arxiv.org/abs/2602.22495` (v3, 2026-06-17) | 2026-11-17 (also read at S42) | "performs selective imitation during RL — guiding the student toward the teacher only when it improves the current policy update"; the KL regularizer "can compete with reward maximization and require careful loss balancing" | **VERIFIED** for selective, advantage-conditioned imitation in post-training. **PARTIAL** for a frozen-base token weight: their signal is a trust-region likelihood ratio on student rollouts. |
| Li et al., *RL-Native Distillation: Exploiting Scored Trajectories for Few-Step Image Generation* (REST), `https://arxiv.org/abs/2608.09226` (2026-09-28) | 2026-11-17 | "Advantage-Modulated Distillation (AMD) transforms rollout advantages into signed weights, strengthening imitation of preferred trajectories and aligning distillation priorities with task value" | **VERIFIED** that *signed* advantage weights are used in a distillation objective (diffusion, RL-scored trajectories). **PARTIAL** for language-model distillation on a fixed supervised stream. |
| Ko, Kim, Chen, Yun, *DistiLLM: Towards Streamlined Distillation for Large Language Models*, `https://arxiv.org/abs/2402.03898` (ICML 2024) | 2026-11-17 | "a novel skew Kullback-Leibler divergence loss" plus "an adaptive off-policy approach"; "up to 4.3x speedup compared to recent KD methods" | **VERIFIED** that KL-form loss choice (not only forward KL) matters for LLM distillation. **PARTIAL** for advantage weighting: the weight is a divergence, not a signed log-probability advantage. |
| S43 read (not re-read): GKD `https://arxiv.org/abs/2306.13649`, MiniLLM `https://arxiv.org/abs/2306.08543`, AWAC `https://arxiv.org/abs/2006.09359` | 2026-11-17 (S43) | alternative distillation losses, reverse KL / on-policy, and advantage-weighted maximum likelihood | **VERIFIED** in `docs/S43-LITERATURE.md` §3 and §5; the verdicts there stand. |
| search for the exact statistic `A_t = log p_T(y_t) - log p_B(y_t)` as the weight of a *corrective residual's* KL, beside a KL-disagreement weight, off-policy, at a frozen base | 2026-11-17 | no located record states this construction | **UNSUPPORTED** for the exact combination. |

**Decision.** Weighting teacher supervision by an advantage is established, and
the 2026 on-policy line has converged on the same two-part lesson: weight by the
advantage **and** stop applying pressure where the advantage is non-positive.
Our registration is the off-policy, frozen-base, token-level instance of that
lesson, and the AOPD result is used as a caution, not a guarantee: the signed
weight here is bounded (`exp(clamp(A, -4, 4))`, never negative), so the failure
mode they name — negative reinforcement in the non-positive region — is not
available by construction; the preservation penalty is what handles that region.

## 2. Token-conditioned gates versus a global scalar

| source | fetched | claim as stated | verdict |
| --- | --- | --- | --- |
| Perez, Strub, de Vries, Dumoulin, Courville, *FiLM: Visual Reasoning with a General Conditioning Layer*, `https://arxiv.org/abs/1709.07871` (AAAI 2018) | 2026-11-17 | "a simple, feature-wise affine transformation based on conditioning information"; "FiLM layers … modulate features in a coherent manner" and are "robust to ablations" | **VERIFIED** that input-dependent, feature-wise modulation of a residual path is a standard, cheap construction. **PARTIAL** for us: the conditioning signal there is the task/question embedding, not the site's own hidden state and memory read. |
| Jiang, Xie, Jiang, Zhang, *Distill Skills into Weights, Not Prompts: Abstract Skills as Privileged Signals for On-Policy Self-Distillation* (SKALD), `https://arxiv.org/abs/2608.09826` (2026-08-10) | 2026-11-17 | "an empirical gate activates distillation only when verified rollouts estimate a positive teacher advantage"; "an annealed exponentially tilted objective that downweights teacher-preferred tokens with very low student likelihood" | **VERIFIED** that gating distillation on a measured teacher advantage per token is a published construction, and that *downweighting* rather than negating is the published safe direction. **PARTIAL** for the gate input: their gate is an estimate from verified rollouts, ours is `tanh(w·[h_t ; r_t] + b)` on the forward-pass features. |
| S43 read (not re-read): Highway Networks `https://arxiv.org/abs/1507.06228`, Dynamic ReLU `https://arxiv.org/abs/2003.10027`, Switch Transformers `https://arxiv.org/abs/2101.03961` | 2026-11-17 (S43) | adaptive gates, input-dependent activation parameters, per-token routing | **VERIFIED** in `docs/S43-LITERATURE.md` §4. |
| search for a zero-initialised per-token correction gate on a frozen backbone's residual, trained by a teacher advantage | 2026-11-17 | no located record states `alpha_t = tanh(w·[h_t;r_t] + b)` with `w = b = 0` as a correction gate | **UNSUPPORTED** for the exact construction; the zero-init bit-identity property is a local design choice, not a published one. |

**Decision.** Token conditioning is mainstream and cheap; zero-initialising the
gate so the untrained arm *is* the base is our own identity guarantee, tested
rather than cited.

## 3. Preservation penalties

| source | fetched | claim as stated | verdict |
| --- | --- | --- | --- |
| Kirkpatrick et al., *Overcoming catastrophic forgetting in neural networks* (EWC), `https://arxiv.org/abs/1612.00796` (PNAS 2017) | 2026-11-17 | "remembers old tasks by selectively slowing down learning on the weights important for those tasks" | **VERIFIED** that a penalty weighted by importance protects previously learned behaviour. **PARTIAL** for us: importance is per weight from a Fisher estimate, ours is per token from the base's own KL. |
| Li, Grandvalet, Davoine, *Explicit Inductive Bias for Transfer Learning with Convolutional Networks* (L2-SP), `https://arxiv.org/abs/1802.01483` (ICML 2018) | 2026-11-17 | "several regularization schemes that explicitly promote the similarity of the final solution with the initial model"; recommends "a simple L² penalty with the pre-trained model being a reference" | **VERIFIED** that anchoring to the pre-trained reference is the simple, effective baseline for transfer. **PARTIAL**: a weight-space penalty, not a function-space KL-increase penalty. |
| Biderman et al., *LoRA Learns Less and Forgets Less*, `https://arxiv.org/abs/2405.09673` (TMLR 2024) | 2026-11-17 | on "programming and mathematics", LoRA "substantially underperforms full finetuning" in standard low-rank settings, yet "LoRA better maintains the base model's performance on tasks outside the target domain" and "mitigates forgetting more than common regularization techniques such as weight decay and dropout" | **VERIFIED** that on a math target domain a low-capacity update can lose in-domain while protecting out-of-domain. **PARTIAL** as an explanation of our measurements: their comparison is LoRA vs full fine-tuning at 100K instruction pairs or 20B continued-pretraining tokens, not a 537,600-element residual at 2M–20M distillation tokens. |
| Jia et al., Asymmetric OPD (above) | 2026-11-17 | "localized divergence minimization in non-positive advantage regions" | **VERIFIED** that region-conditioned protection against harmful supervision is a published design. |
| search for a per-token preservation weight `p_t = sigmoid(-(a_t - mean a)/mean a)` multiplying a penalty on any *increase* over the base's own KL | 2026-11-17 | no located record states this weight or this penalty | **UNSUPPORTED** for the exact form. |

**Decision.** Protecting the base is established in three separate literatures
(weight importance, weight-space anchoring, parameter-efficient capacity), and
the 2026 distillation line protects the non-positive-advantage region
functionally. Our per-token KL-increase penalty is the functional-space instance
of the same idea; its harm masks (lowest advantage quartile, signed `A_t >= 0`)
are how the registration refuses to accept a mean-only improvement.

## 4. Math-data mixtures at small token budgets

| source | fetched | claim as stated | verdict |
| --- | --- | --- | --- |
| Xie et al., *DoReMi: Optimizing Data Mixtures Speeds Up Language Model Pretraining*, `https://arxiv.org/abs/2305.10429` (NeurIPS 2023) | 2026-11-17 | a 280M proxy sets domain weights for an 8B model (30x larger); "improves average few-shot downstream accuracy by 6.5% points" and "reaches the baseline accuracy with 2.6x fewer training steps" | **VERIFIED** that mixture proportions are a first-order lever and that a small proxy is predictive. **PARTIAL** for us: domain-reweighted pretraining at 8B-scale compute, not a 20M-token distillation mixture. |
| Liu et al., *RegMix: Data Mixture as Regression for Language Model Pre-training*, `https://arxiv.org/abs/2407.01492` (ICLR 2025) | 2026-11-17 | 512 one-million-parameter models trained for 1B tokens fit a regression that picks the mixture for a 1B/25B run; "data mixture effects transcend scaling laws"; "web corpora rather than data perceived as high-quality like Wikipedia have the strongest positive correlation with downstream performance" | **VERIFIED** that mixture effects are measurable at very small proxy scale and do not reduce to a single "quality" axis. **PARTIAL**: pretraining from scratch, not distillation into a frozen base. |
| Ye et al., *Data Mixing Laws*, `https://arxiv.org/abs/2403.16952` (ICLR 2025) | 2026-11-17 | mixture performance is quantitatively predictable from small-scale training; extending to continual training "accurately predicts the critical mixture proportion that avoids catastrophic forgetting" | **VERIFIED** that mixture interacts with forgetting in continued training. **PARTIAL** for a 57%-math, 43%-general 20M-token distillation stream. |
| Hu et al., *MiniCPM*, `https://arxiv.org/abs/2404.06395` (2024) | 2026-11-17 | warmup-stable-decay scheduler for continuous training and domain adaptation; "much higher compute optimal data-model ratio than Chinchilla Optimal" for small models | **VERIFIED** that small models are data-hungry relative to the classic ratio and that domain adaptation is a distinct data regime. **PARTIAL** for 20M tokens. |
| S43 read (not re-read): Llemma `https://arxiv.org/abs/2310.10631`, MathPile `https://arxiv.org/abs/2312.17120`, DeepSeekMath `https://arxiv.org/abs/2402.03300`, Nemotron-CC-Math `https://arxiv.org/abs/2508.15096` | 2026-11-17 (S43) | math-heavy mixtures as a lever, at billions of tokens | **VERIFIED** in `docs/S43-LITERATURE.md` §1–§2; the verdict for a 20M-token distillation stream stays **PARTIAL**. |
| search for a measured mixture ablation of **57% math in 20M online-distillation tokens into a frozen 0.8B student** | 2026-11-17 | no located record | **UNSUPPORTED** for the exact setting: the stream is a local registration with a measured target, not a reproduction. |

**Decision.** Mixture is a real lever and proxy scale predicts it, but no source
covers our budget. The S44 measurement sharpens the local question: the trained
arms lose the *generative* math tasks while the adapter gains on MMLU, which is
the same direction as LoRA-versus-full-fine-tuning on a math target domain
(`2405.09673`) — a capacity/target-domain trade, not evidence that the mixture is
wrong. The registered corrective arms keep the stream fixed so that a win or a
loss is attributable to the mechanism rather than to a second change.

## What this file does not establish

- That any cited gain transfers to a frozen 0.8B backbone with a 539,649-element
  residual. Every source above changes more parameters, or changes the student's
  rollouts, or both.
- That the signed advantage is better than the KL-disagreement weight here. The
  literature says advantage weighting is established *somewhere*; the comparison
  at our budget is the registered experiment.
- That the preservation penalty cannot itself cost in-domain gain. AOPD's
  reported weakness of advantage weighting is the reason the penalty exists, not
  evidence that the penalty is free.
