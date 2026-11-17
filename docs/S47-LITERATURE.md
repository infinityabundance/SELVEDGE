# S47 focused literature — signed weighting, harm/tie diagnostics, token penalties, matched controls, utility routing

Status: **focused verification, 2026-11-17 (S47).** Each entry names the URL,
the fetch date, the claim the source itself states, and the verdict for the
question this session had to decide. `VERIFIED` means the claim is in the
fetched record; `PARTIAL` means the source supports part of the claim or a
different setting; `UNSUPPORTED` means no located source states the claim. An
abstract does not settle a paper, and the verdicts are scoped to what was read.
Sources already read at S43/S44 are named rather than re-read, and say so.

The session had six questions:

1. is **signed / advantage weighting** established, and what distinguishes a
   bounded sign-split weight from an unbounded positive exponential;
2. is a **harm/tie diagnostic** established for the region where the base is
   already better than the teacher;
3. is **token-specific regularisation** established (an aligned per-token
   penalty rather than a product of means);
4. are **matched-data controls** established for distillation comparisons;
5. is **counterfactual utility routing** established;
6. is **outcome-supervised gating** established.

## 1. Signed and advantage weighting

| source | fetched | claim as stated | verdict |
| --- | --- | --- | --- |
| Sabbaghi, Pappas, Hassani, Javanmard, *When to Follow the Teacher: Advantage-Based On-Policy Distillation*, `https://arxiv.org/abs/2610.31563` (2026-10-22) | 2026-11-17 | "standard OPD trains the student to follow the teacher at every state… assuming that teacher guidance is always beneficial. We instead distinguish between what the teacher prefers and what actually helps the current student succeed… weights teacher actions by their counterfactual advantage under the student. Teacher actions that improve the student's expected outcome are reinforced, while those that do not are downweighted." | **VERIFIED** that counterfactual-advantage weighting is a published objective with reported math/coding/MedQA gains. **PARTIAL** for us: their advantage is an outcome advantage on student rollouts; ours is an off-policy token statistic `A_t = log p_T(y_t) - log p_B(y_t)` inside a frozen-base residual. |
| Jia et al., *Asymmetric On-Policy Distillation: Bridging Exploitation and Imitation at the Token Level*, `https://arxiv.org/abs/2605.06387` (2026-08-30) | 2026-11-17 (also read at S44) | "its standard advantage weighted policy gradient suffers from three structural weaknesses, including high variance updates, vanishing gradients in zero-advantage regions… replaces ineffective negative reinforcement with localized divergence minimization in non-positive advantage regions while preserving positive reinforcement learning", with "+4.09 / +8.34" average gains on mathematical reasoning | **VERIFIED** that the non-positive-advantage region needs a *different* objective, not just a smaller weight. **PARTIAL** for the exact form: they minimise a divergence there; we add a token-loss protection term on `A_t < 0` beside a sigmoid weight. |
| Li et al., *RL-Native Distillation: Exploiting Scored Trajectories for Few-Step Image Generation* (REST), `https://arxiv.org/abs/2608.09226` (2026-09-28) | 2026-11-17 (also read at S44) | "Advantage-Modulated Distillation (AMD) transforms rollout advantages into signed weights, strengthening imitation of preferred trajectories and aligning distillation priorities with task value" | **VERIFIED** that signed advantage weights are used in a distillation objective. **PARTIAL** for language-model off-policy token distillation: the setting is diffusion with RL-scored trajectories. |
| Ko, Kim, Chen, Yun, *DistiLLM: Towards Streamlined Distillation for Large Language Models*, `https://arxiv.org/abs/2402.03898` (ICML 2024) | 2026-11-17 (also read at S44) | "a novel skew Kullback-Leibler divergence loss" plus "an adaptive off-policy approach"; "up to 4.3x speedup compared to recent KD methods" | **VERIFIED** that the *form* of the KL term matters for LLM distillation. **PARTIAL** for advantage weighting: the weight is a divergence, not a signed log-probability advantage. |
| S43 read (not re-read): GKD `https://arxiv.org/abs/2306.13649`, MiniLLM `https://arxiv.org/abs/2306.08543`, AWAC `https://arxiv.org/abs/2006.09359` | 2026-11-17 (S43) | alternative distillation losses, reverse KL / on-policy, and advantage-weighted maximum likelihood | **VERIFIED** in `docs/S43-LITERATURE.md` §3 and §5; the verdicts there stand. |
| search for a **bounded, tie-centred sigmoid weight** `u_t = sigmoid(A_t / tau)` on teacher-directed KL beside a separate `A_t < 0` token-loss term, off-policy, at a frozen base's residual | 2026-11-17 | no located record states this exact pair | **UNSUPPORTED** for the exact construction. The nearest published lesson is the two-part design above: weight by advantage *and* change the objective in the non-positive region. |

**Decision.** Advantage weighting is established. Our `signed-exp` and
`signed-true` are two instances of the same lesson and are registered as a
comparison, not as a claimed improvement. `signed-exp` is monotone and positive,
so it only suppresses imitation. `signed-true` adds the explicit `A_t < 0`
protection term, which is the part the 2026 line says is missing from weighting
alone. The exact pair is ours and is tested, not cited.

## 2. Harm and tie diagnostics

| source | fetched | claim as stated | verdict |
| --- | --- | --- | --- |
| Sabbaghi et al., AOPD (above) | 2026-11-17 | distinguishes "what the teacher prefers" from "what actually helps the current student succeed" | **VERIFIED** that a diagnostic separation between teacher preference and student benefit is a published idea. **PARTIAL** for a token mask: their separation is an expected-outcome advantage, not a reported `A_t` sign mask. |
| Jia et al., Asymmetric OPD (above) | 2026-11-17 | "non-positive advantage regions" are the region where standard weighting is ineffective | **VERIFIED** that the non-positive region is named and handled separately. **PARTIAL** for a three-way `A>0` / `A<0` / `A==0` split: no source reports the tie bucket. |
| S45 internal measurement (not literature): `results/REPORT.md` §S45, `results/stage2/s45-forward-analysis-20261117T082511Z.json` | 2026-11-17 | on the correct path the `memory-2m` residual adds +0.0373 nats/token over 80.0% of positions; 203 of 208 newly-wrong GSM8K items change the extracted final answer | **VERIFIED** as a *local measurement*, not a citation: next-token likelihood and exact-answer correctness can move in different directions. |
| search for a published `tie` bucket (`A_t == 0`) in a distillation harm report | 2026-11-17 | no located record states one | **UNSUPPORTED** for the exact three-way split. The S47 correction is a local diagnostic fix: a `>=` mask cannot separate a tie from a teacher-better token, and its label was reversed. |

**Decision.** The corrected diagnostic reports `teacher_better`, `base_better`
and `tie` separately. The old `signed_base_better` key is removed rather than
relabelled in place, so a reader of a new receipt cannot mistake the meaning.
No threshold moves.

## 3. Token-specific regularisation

| source | fetched | claim as stated | verdict |
| --- | --- | --- | --- |
| Kirkpatrick et al., *Overcoming catastrophic forgetting in neural networks* (EWC), `https://arxiv.org/abs/1612.00796` (PNAS 2017) | 2026-11-17 (also read at S44) | "remembers old tasks by selectively slowing down learning on the weights important for those tasks" | **VERIFIED** that a penalty weighted by importance protects prior behaviour. **PARTIAL** for us: importance is per weight from a Fisher estimate; ours is per token from the base's own KL and the sign of `A_t`. |
| Li, Grandvalet, Davoine, *Explicit Inductive Bias for Transfer Learning with Convolutional Networks* (L2-SP), `https://arxiv.org/abs/1802.01483` (ICML 2018) | 2026-11-17 (also read at S44) | recommends "a simple L² penalty with the pre-trained model being a reference" | **VERIFIED** that anchoring to the pre-trained reference is a strong simple baseline. **PARTIAL**: weight space, not per-token function space. |
| Jia et al., Asymmetric OPD (above) | 2026-11-17 | "localized divergence minimization in non-positive advantage regions" | **VERIFIED** that a *region-conditioned* penalty is a published design. **PARTIAL** for the form: no per-token alignment between a preservation weight and a gate magnitude is reported. |
| search for `mean(p_t * alpha_t^2)` as a token-aligned interference penalty, versus the factorised `mean(p_t) * mean(alpha_t^2)` | 2026-11-17 | no located record states either form | **UNSUPPORTED** for the exact penalty. The factorised product is a local construction whose defect is arithmetic: it is invariant to whether the gate and the preservation weight occur on the same token. |

**Decision.** The aligned mean is the correct arithmetic for "charge the gate
where protection matters", and a CPU test constructs two configurations with
identical means and different alignment. The old product is retained only as a
labelled diagnostic, because it is useful for showing that a mean-only reading
cannot see the difference.

## 4. Matched-data controls

| source | fetched | claim as stated | verdict |
| --- | --- | --- | --- |
| Biderman et al., *LoRA Learns Less and Forgets Less*, `https://arxiv.org/abs/2405.09673` (TMLR 2024) | 2026-11-17 (also read at S44) | on "programming and mathematics", LoRA "substantially underperforms full finetuning" in standard low-rank settings, yet "better maintains the base model's performance on tasks outside the target domain" | **VERIFIED** that a lower-capacity update can lose in-domain while protecting out-of-domain. **PARTIAL** as an attribution control: their comparison changes capacity and data together, not only the mechanism. |
| Xie et al., *DoReMi*, `https://arxiv.org/abs/2305.10429` (NeurIPS 2023) | 2026-11-17 (also read at S44) | a 280M proxy sets domain weights for an 8B model; "improves average few-shot downstream accuracy by 6.5% points" and "reaches the baseline accuracy with 2.6x fewer training steps" | **VERIFIED** that mixture proportions are a first-order lever. **PARTIAL** for a matched-data control: it shows the data matters, not how to hold it fixed while changing the mechanism. |
| Liu et al., *RegMix*, `https://arxiv.org/abs/2407.01492` (ICLR 2025) | 2026-11-17 (also read at S44) | 512 one-million-parameter models fit a regression that picks a mixture for a 1B/25B run; "data mixture effects transcend scaling laws" | **VERIFIED** that mixture effects are measurable at small proxy scale. **PARTIAL**: pretraining from scratch, not distillation into a frozen base. |
| search for a dense adapter trained on the **exact same** stream, teacher and optimizer budget at the **same active union** as a corrective residual, as the decisive control | 2026-11-17 | no located record states this exact matched-control design | **UNSUPPORTED** for the exact control. It is a local experimental-design requirement: without it a corrective gain cannot be separated from a data gain. |

**Decision.** The stage-2b registration makes `adapter-20m-math-matched` and
`gated-dense-20m-math` mandatory comparison rows. The first holds the data,
teacher and optimizer fixed and removes the memory and gate; the second holds
the gate fixed and removes the external memory. A gain that either control also
has is attributed to data or gate, not to the memory mechanism.

## 5. Counterfactual utility routing

| source | fetched | claim as stated | verdict |
| --- | --- | --- | --- |
| Sabbaghi et al., AOPD (above) | 2026-11-17 | "weights teacher actions by their counterfactual advantage under the student… can be estimated using only standard student rollouts and their terminal rewards" | **VERIFIED** that counterfactual advantage is a published routing signal and can be estimated without trying every alternative action. **PARTIAL** for P19.9: we predict a per-token loss improvement from `(h_t, r_t)` and gate by it minus a measured cost; their estimator is an importance-weighted terminal-reward advantage. |
| Zhang et al., *Reinforcement-aware Knowledge Distillation for LLM Reasoning* (RLAD), `https://arxiv.org/abs/2602.22495` (v3, 2026-06-17) | 2026-11-17 (also read at S42) | "performs selective imitation during RL — guiding the student toward the teacher only when it improves the current policy update"; the KL regularizer "can compete with reward maximization and require careful loss balancing" | **VERIFIED** that selective, improvement-conditioned imitation is published. **PARTIAL** for a token utility head: their signal is a trust-region likelihood ratio on rollouts. |
| search for a per-token regressor of `Delta_t = l_base(t) - l_{base+mem}(t)` from `(h_t, r_t)`, gated by `sigmoid((Delta_hat - cost)/tau)` | 2026-11-17 | no located record states this exact construction | **UNSUPPORTED** for the exact router. The nearest published idea is the counterfactual advantage above. |

**Decision.** P19.9 is registered as a skeleton with an explicit supervision
source, leakage rules and a cost term. It is not run. The label interface
accepts either a measured counterfactual token-loss difference or a verified
sequence outcome; it rejects a target token or a label key in the feature dict,
and it splits by prompt id.

## 6. Outcome-supervised gating

| source | fetched | claim as stated | verdict |
| --- | --- | --- | --- |
| Jiang, Xie, Jiang, Zhang, *Distill Skills into Weights, Not Prompts* (SKALD), `https://arxiv.org/abs/2608.09826` (2026-08-10) | 2026-11-17 | "an empirical gate activates distillation only when verified rollouts estimate a positive teacher advantage"; "an annealed exponentially tilted objective that downweights teacher-preferred tokens with very low student likelihood"; gains of +2.46 / +4.85 / +12.01 avg@8 over GRPO at 0.6B / 1.7B / 4B | **VERIFIED** that gating distillation on a **verified** signal is published and that downweighting, not negating, is the published safe direction. **PARTIAL** for P19.9: their gate is estimated from verified rollout groups; our `verified-outcome` source uses the S45 exact final-answer verdict as a sequence label and broadcasts it to tokens. |
| S45 read (not re-read): the S45 verifier in `oracle/frontier/rft.py` | 2026-11-17 | exact numeric or canonical-expression comparison with a documented extractor; ambiguity rejected | **VERIFIED** in `docs/S45-LITERATURE.md` and the S45 tests. It is **not** a symbolic verifier, and a false accept is treated as worse than a false reject. |
| search for broadcasting a **sequence-level verified answer** to every token as the supervision of a token router | 2026-11-17 | no located record states this exact broadcast | **UNSUPPORTED** for the exact construction. It is a registered experiment: sequence outcomes are sparse and a token that is not responsible for the final answer can receive credit or blame. |

**Decision.** Outcome supervision is registered as its own arm, not as a
replacement for the counterfactual-token-loss arm. The leakage rule that a
teacher-forced likelihood may not stand in for a verified outcome is enforced in
`labels_from_records`.

## What this file does not establish

- That `signed-true` beats `signed-exp` at this budget. The literature says the
  non-positive region needs its own objective; the comparison here is the
  registered experiment.
- That the token-aligned interference penalty improves the arm. It repairs an
  arithmetic defect in a diagnostic and objective term; the effect is unmeasured.
- That the matched controls will run before the 20M bar exits. Their classes are
  implemented and CPU-tested, but `train.py` wiring is deferred by the operator's
  no-edit rule while that process is running.
- That P19.9 improves routing. It is a skeleton with CPU tests and no run.
- That CE/KL and exact-answer accuracy agree. They are both reported because the
  S45 measurement shows they can diverge.
