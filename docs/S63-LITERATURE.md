# S63 — literature: preference objectives, DPO/IPO/CPO, length normalisation, LoRA merge, recovery

Every entry names the URL, the date it was read, the exact claim being checked,
and one of **VERIFIED** / **PARTIAL** / **UNSUPPORTED**. `VERIFIED` means the
claim is visible in the named primary source; it does not mean the paper is
right, or that it transfers to this student and this corpus. `PARTIAL` means the
source supports part of the claim, or supports it in a different setting.
`UNSUPPORTED` means no source for the claim was found in this session; it is not
a claim that no source exists.

Sources were read on **2026-11-19** at metadata/abstract level through the arXiv
abstract pages, unless a specific section is named below.

## 1. DPO: a preference objective that needs only action log-probabilities

| source | claim checked | status |
| --- | --- | --- |
| [Direct Preference Optimization](https://arxiv.org/abs/2305.18290), Rafailov et al., submitted 2023-05-29, NeurIPS 2023 | The RLHF objective can be reparameterised as a classification loss on preference pairs with a frozen reference policy: `-log sigmoid(beta [log(pi(y_w|x)/pi_ref(y_w|x)) - log(pi(y_l|x)/pi_ref(y_l|x))])`. | **VERIFIED** |
| same | Both responses are conditioned on the same prompt `x`, so the comparison is prefix-local at the sequence level. | **VERIFIED** |
| same | The sequence log-probability is the sum over response tokens; the paper does not length-normalise the sum. | **VERIFIED** |
| Inference | "A cached scalar teacher log-prob per token is a reverse KL" | **FALSE by construction, not a literature claim.** Reverse KL needs the teacher distribution over the vocabulary at each prefix; a preference objective needs only the two actions' log-probabilities. This is why S63 replaces the cached-scalar reverse-KL clause, not why DPO is preferred on empirical grounds. |

**What this changes for S63.** The corrected stage 2 uses the DPO form, not
because DPO is reported to win, but because it is the objective the cached
frozen-reference data can actually compute. The registered length
normalisation is the per-token mean (section 3), which is *not* the DPO paper's
default sum.

## 2. IPO and CPO: alternatives to the unbounded log-sigmoid margin

| source | claim checked | status |
| --- | --- | --- |
| [A General Theoretical Paradigm to Understand Learning from Human Preferences (IPO)](https://arxiv.org/abs/2310.12036), Azar et al., submitted 2023-10-18, AISTATS 2024 | A general preference framework yields Identity Preference Optimisation: a squared loss on the preference margin instead of the log-sigmoid, motivated by DPO's tendency to overfit a deterministic preference. | **VERIFIED** |
| [Contrastive Preference Optimization (CPO)](https://arxiv.org/abs/2401.08417), Xu et al., submitted 2024-01-16, ICML 2024 | A contrastive preference objective with a reference-model regulariser can replace a separate reward model, and uses average log-likelihood (length-normalised) in its reward. | **VERIFIED** |
| Inference | "IPO or CPO is the right objective for this student" | **UNSUPPORTED** — no source measures either on this corpus, this student or this reference. S63 registers DPO-form as the first objective and keeps IPO/CPO as later ablations, not as evidence. |

**What this changes for S63.** The loss stays the log-sigmoid form with a stated
`beta=0.1`. IPO's squared loss and CPO's reference regulariser are recorded as
the natural next ablations if the preference margin saturates, but neither is
claimed to help here.

## 3. Sequence-level versus token-level, and length normalisation

| source | claim checked | status |
| --- | --- | --- |
| [Token-level Direct Preference Optimization (TDPO)](https://arxiv.org/abs/2404.11999), Zeng et al., submitted 2024-04-18 | Sequence-level DPO aggregates all token log-probabilities; a token-level f-divergence objective controls the per-token trade-off and is reported to improve alignment. | **PARTIAL** — the paper supports the distinction and its own token-level objective; it does not show that token-level is better for a 0.8B math SFT route. |
| [SimPO](https://arxiv.org/abs/2405.14734), Meng, Xia, Chen, submitted 2024-05-23 | Uses the **average** log-likelihood (length-normalised) as the implicit reward, plus a target reward margin, and reports that this removes the need for a reference model. | **VERIFIED** |
| [CPO](https://arxiv.org/abs/2401.08417) (above) | Also length-normalises by response length in its reward. | **VERIFIED** |
| Inference | "Length normalisation removes length bias" | **PARTIAL** — SimPO and CPO report it as part of a larger objective. S63 uses the per-token mean so a long failed continuation cannot win a pair by having more tokens, and reports both token counts; it does not claim to have removed length bias from the policy. |

**What this changes for S63.** The objective is sequence-level with a per-token
mean normalisation, stated in the registration and written into every run
receipt. Token-level preference and SimPO's reference-free reward are later
ablations.

## 4. Prefix-local preference pairs

| source | claim checked | status |
| --- | --- | --- |
| [DPO](https://arxiv.org/abs/2305.18290) | The preference pair is conditioned on a shared prompt, so the likelihood ratio is local to that prefix. | **VERIFIED** |
| [FPRT: Failure Profiling and Reachable Trajectory Selection](https://arxiv.org/abs/2611.09042), submitted 2026-11-06 | Repeated sampling profiles failure modes, and among correct teacher candidates the one whose first corrective steps are reproducible by the student is preferred, because the first steps after divergence are the bottleneck. | **VERIFIED** |
| Inference | "A shared student prefix with a teacher continuation and the student's failed continuation is a published objective" | **PARTIAL** — DPO supplies the shared-prefix comparison; FPRT supplies the reachable-continuation selection. The exact pair construction (teacher continuation from the student's own cut prefix vs the student's failed continuation) is this registration's instantiation. |

**What this changes for S63.** A pair is `(h, a+, a-)` with one shared prefix
`h`; chosen is a verified teacher continuation from `h`, rejected is the
student's failed continuation from the same `h`. No teacher prefix is inserted
into the student's context, and the frozen reference is scored on both actions.

## 5. LoRA merge parity

| source | claim checked | status |
| --- | --- | --- |
| [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685), Hu et al., submitted 2021-06-17, ICLR 2022 | A LoRA update is `W = W_0 + BA` with `B` zero-initialised; merging the update into `W_0` introduces no inference latency and can be subtracted to recover the base task. | **VERIFIED** |
| Inference | "The merged model computes exactly the adapter's function" | **VERIFIED in exact arithmetic.** In floating point, `xW_0^T + xA^T B^T` and `x(W_0 + BA)^T` round differently. S63 states a test tolerance of `atol=1e-5, rtol=1e-4` on float32 and does not claim bit identity. |
| Inference | "The adapter stays in the active union after merge" | **FALSE by construction.** The adapter's elements are written into weights the base already reads; the deployed unique active union is the base count, 752,393,024. |

**What this changes for S63.** All-linear rank-64 LoRA is merged at the end; the
merge parity and active-union arithmetic are CPU tests, and the capability
comparison is the run.

## 6. Teacher-guided recovery for unsolved problems

| source | claim checked | status |
| --- | --- | --- |
| [STaR: Bootstrapping Reasoning With Reasoning](https://arxiv.org/abs/2203.14465), Zelikman et al., submitted 2022-03-28, NeurIPS 2022 | When the model fails to produce a correct rationale, a hint or teacher rationale can be used to obtain one, and the model is then trained on the successful rationales. | **VERIFIED** |
| [Distilling Step-by-Step](https://arxiv.org/abs/2305.02301), Hsieh et al., submitted 2023-05-03 | A larger teacher's rationales can train a smaller model, including on problems the smaller model does not solve on its own. | **PARTIAL** — supports teacher rationales for capability gaps, not the matched-weight rule used here. |
| [ReST](https://arxiv.org/abs/2308.08998), Gulcehre et al., submitted 2023-08-17 | Filtered self-generated data can improve a model over iterations; the corpus is the model's own successful samples. | **PARTIAL** — the source supports self-generated filtering, not a teacher-guided branch for p=0. |
| Inference | "For a p=0 problem, adding a verified teacher solution with the mean positive p(1-p) is a published rule" | **UNSUPPORTED** — the teacher-guided recovery idea is supported by STaR; the **matched weight** is this registration's design, chosen because p(1-p)=0 would otherwise drop the problem entirely. |

**What this changes for S63.** The recovery arm is registered with its own
control and its own paired interval. It is not called a PACED result, and no
source is cited for the matched weight.

## 7. Why a cached scalar cannot be reverse KL

| source | claim checked | status |
| --- | --- | --- |
| [On-Policy Distillation of Language Models](https://arxiv.org/abs/2306.13649), Agarwal et al., submitted 2023-06-23, ICLR 2024 | On-policy distillation uses the teacher's full next-token distribution on the student's own generated sequences; the reverse-KL objective is over the vocabulary at each prefix. | **VERIFIED** |
| Definition of KL divergence | `KL(p || q) = sum_y p(y) [log p(y) - log q(y)]`; a single `log q(a)` for one sampled action is not the sum over `y`. | **VERIFIED** (standard definition; no empirical claim) |
| Inference | "Summing cached scalar teacher log-probs over a trajectory implements reverse KL" | **UNSUPPORTED and mathematically false.** It is a teacher likelihood, not a KL. S63 does not use that name for any quantity. |

## 8. Effect on the S63 design (summary)

| S63 decision | literature position |
| --- | --- |
| reference-anchored log-sigmoid preference on shared-prefix pairs | DPO form is verified; the shared *student* prefix and teacher/failed continuation pairing are this registration's instantiation |
| per-token mean normalisation | supported by SimPO and CPO as a length control; not the DPO default and not claimed to remove all length bias |
| `beta=0.1`, no IPO/CPO ablation in the first run | IPO and CPO are verified alternatives, but neither is evidenced for this student |
| rank-64 all-linear LoRA, merged | LoRA's merge identity is verified; the rank is a registered choice (see `docs/S60-LITERATURE.md` for the all-linear regime) |
| active union 752,393,024 after merge | follows from the merge identity; the adapter adds no deployed identity |
| teacher-guided recovery with matched weight for p=0 | STaR supports teacher rationales for failures; the matched weight is **this file's design, not a published result** |
| no quantity called reverse KL | a cached scalar cannot implement KL; the preference margin is not renamed |
| CURE intervention screen | **UNSUPPORTED / not yet defined** — operator-named later item; no protocol, source or receipt exists in this session |
