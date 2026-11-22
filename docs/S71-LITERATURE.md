# S71 — literature review and ranked mechanism plan

Status: **reading, reasoning and writing only.** No GPU was used, no court was
run, no checkpoint was scored, and no new measurement exists in this file.
Program numbers are quotations from the receipts named beside them. Literature
numbers are quotations from the cited paper, model card or report; where the
source does not state a protocol detail, this file says so instead of filling
it in. Numbers labelled **estimate** show their arithmetic and are not claims.
This document is not a performance doc and registers no `perf_numbers` block;
the throughput and memory numbers it quotes are receipt quotations, named at the
point of use.

This is not a survey. It answers one question: *what should the next GPU-days
be spent on, given that the target is MATH-500 ≥ 50 and full MATH ≥ 50 at
752,393,024 unique active parameters under a greedy, non-thinking, 4096-cap
protocol?*

---

## 0. The situation, and the two facts that decide the plan

The program's own receipts:

| quantity | value | receipt |
|---|---|---|
| deployed unique active text parameters | 752,393,024 | `results/accounting/20261116T183609Z-qwen35-08b-text-only.json`; `docs/S65-RESULT.md` §1 |
| post-trained baseline: GSM8K / MATH-500 / MMLU | 54.21 / 38.20 / 50.48 | `docs/S67-RESULT.md` §4; `results/stage2/s67-final/verdict.json` |
| full-backbone SFT on 4,084 verified 2B-teacher solutions | 4,376 steps, 10,940,000 processed tokens, 2,081 tok/s, 5,257 s | `docs/S58-STAGE2A.md` §3 |
| that SFT at 100%: GSM8K / MATH / MMLU deltas | +2.9 / −11.0 / −1.9 | program statement; consistent with `docs/S65-RESULT.md` §2 (α=1.0 MATH-cal 23.67 vs 34.00 at α=0) |
| WiSE-FT at α=0.375 into the base (champion) | GSM8K 63.38, MATH-500 40.60, full MATH 42.12, MMLU 52.07, MMLU-Pro 26.75 | `docs/S67-RESULT.md` §4–5 |
| S67 seed deltas: norm / base norm / cosine | 13.42 / 361.99 (3.7%) / ≈0.32 | `docs/S67-RESULT.md` §1 |
| S69 preference arm: 214 pairs, 6.05 epochs, β=0.1 | loss 0.6285 → 5.85e-05, margin ≈97 nats, 20.9% clipping, pre-clip max 95.5 | `docs/S69-RESULT.md` §2 |
| S69 preference generation behaviour | 100% of 2,000 calibration sequences hit the 2048 cap; 0% EOS, at every cadence | `docs/S69-RESULT.md` §4 |
| S69 CE controls on the identical chosen side | GSM8K-cal within 3.6 points, MATH-cal within 6.1 points of the soup at 2M; no collapse | `docs/S69-RESULT.md` §3, §7 |
| target | MATH-500 ≥ 50 and full MATH ≥ 50 at ≤1B unique active | program statement |

**Fact 1.** Every published sub-1B model at or above 50 MATH uses either
200–1,500× more post-training data than our 4,084 traces, or long chain-of-thought
at inference, or both. The closest published protocol match to ours — greedy,
zero-shot CoT — is Qwen2.5-Math-1.5B-Instruct at **75.8 MATH** (arXiv:2409.12122,
Table 3), and it is a 1.5B model trained on **2.0M English + 0.5M Chinese CoT
samples, 470K tool-integrated samples, and 66K RL queries**, after **1T**
math-specific pre-training tokens. Our corpus is 4,084 problems / 10.94M tokens.
The ratio is roughly 600× in SFT samples.

**Fact 2.** Our existing delta has a measured ceiling. The best calibration
point in the S65 single-seed grid is **35.0 MATH-cal at α=0.25**
(`docs/S65-RESULT.md` §2); the soup's best is **32.60 at α=0.375**
(`docs/S67-RESULT.md` §2). The best test score obtained from this delta is
**42.80 MATH-500** (S65 seed-0 at α=0.375, `docs/S67-RESULT.md` §4.1) and
**42.12 full MATH** (the soup). Even the most favourable point is about eight
points short of 50. There is no mechanism in the delta-manipulation literature
by which re-scaling, rank-truncation, sign-resolution or re-merging of *that*
delta adds +10 points, because the delta's content is a 4,084-problem
2B-teacher signal. Manipulating it is a safety exercise (how much of it is
safe), not a capability exercise. Reaching 50 requires a different delta, which
means different data or a different teacher.

The rest of this file defends those two facts and converts them into a ranked
experiment plan.

---

## A. How published sub-1B and small models actually reach ~50 on MATH

### A.1 Sources

The table separates **training-time** from **inference-time** compute. "TTC"
means test-time compute: search, sampling plus voting, a verifier, or a long
thinking trace at inference. "Protocol" is what the source says about decoding;
where the source does not say, that is recorded.

| model (params) | training data | objective | TTC at inference | MATH number | protocol | source |
|---|---|---|---|---|---|---|
| **MobileLLM-R1-950M** (949M) | 4.2T pre-train tokens (2T unique, resampled) + 6.2M reasoning SFT samples × 4 epochs, 32k seq; plus 866K general SFT | full SFT | none stated for MATH; AIME reported avg@64 | **74.0 MATH-500** | 0-shot pass@1; temperature/cap not stated on the card | arXiv:2509.24945; HF card `facebook/MobileLLM-R1-950M` |
| **Qwen3-0.6B, thinking** (0.6B) | 36T pre-train + cold-start SFT + RL (report §4) | SFT + RL | long CoT (thinking block) | **77.6 MATH-500** | report benchmark settings (not our 4096 greedy) | arXiv:2505.09388, Table 19 |
| **Qwen3-0.6B, non-thinking** (0.6B) | same weights, thinking disabled | same | none | **55.2 MATH-500** | report benchmark settings | arXiv:2505.09388, Table 20 |
| **Qwen2.5-Math-1.5B-Instruct** (1.5B, over ceiling) | 1T math pre-train + 2.0M EN + 0.5M ZH CoT + 470K TIR (190K annotated, 205K synthesised, 75K translated); 3 epochs at 4,096; then GRPO on 66K queries | SFT + GRPO | none (greedy CoT); RM@8/maj@8 reported separately | **75.8 MATH CoT (greedy); 79.9 TIR (greedy)** | zero-shot pass@1; MMLU-STEM is 5-shot | arXiv:2409.12122, Table 3, §3.1–3.3 |
| **DeepSeek-R1-Distill-Qwen-1.5B** (1.54B, over ceiling) | 800K samples curated with DeepSeek-R1 | SFT | long CoT, **32,768 max generation** | **83.9 MATH-500** | pass@1, temperature 0.6, top-p 0.95; 64 responses are used for AIME, not for MATH | HF card `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`; arXiv:2501.12948 |
| **rStar-Math** (3.8B / 7B) | 747K math problems, millions of MCTS-synthesised solutions, 4 self-evolution rounds | SFT + process preference model | **MCTS at inference guided by a process reward model** | **86.4 (3.8B); 90.0 (7B)** | search, not greedy; not comparable to pass@1 greedy | arXiv:2501.04519 |
| **Qwen2.5-1.5B + SimpleRL-Zoo** (1.5B, over ceiling) | ~8K GSM8K/MATH problems, 8 rollouts, 8,192 max rollout; zero RL | GRPO | none | **29.6 → 59.0 MATH-500** | pass@1, temperature 1.0, 16K generation | arXiv:2503.18892, §2.1, App. B |
| **Qwen2.5-0.5B + SimpleRL-Zoo** (0.5B) | same | GRPO | none | **15.8 → 34.4 MATH-500** | same | arXiv:2503.18892 |
| **R1-Distill-Qwen-1.5B + RL** (1.5B) | 7K samples, 500 global steps, 6 generations, 4,096 max completion; 4×A40, 24 h | GRPO | none | **83 → 84 MATH-500** (AMC23 63 → 80) | 4,096 completion; accuracy and format rewards | arXiv:2503.16219 |
| **TinyLlama** (1.1B) | ~1T pre-train tokens | none | none | **no MATH number reported** | — | arXiv:2401.02385 |
| **"Small Language Models are the Future of Agentic AI"** | position paper | — | — | **no math benchmark** | — | arXiv:2506.02153 |

Two entries deserve a loud warning because they are widely repeated as if they
were greedy pass@1 numbers:

* **rStar-Math's 90.0 is a search number.** The abstract says the policy SLM
  performs "test-time search" guided by a process reward model, with MCTS. It is
  not a greedy number and it is not a training-only number. The headline is a
  *training plus inference* result.
* **DeepSeek-R1-Distill-Qwen-1.5B's 83.9 is a long-CoT number.** The card sets
  the maximum generation length to 32,768 and recommends temperature 0.6. Under
  our 4,096-token greedy cap the same weights are a different measurement, and
  the card itself warns that these models sometimes bypass the thinking pattern
  unless forced. This is exactly the test-time-compute separation the program
  needs to keep.

The same-model harness variance is large enough to matter at our target: the
MobileLLM card reports Qwen3-0.6B at **73.0** MATH-500 and R1-Distill-Qwen-1.5B
at **83.2**, while the Qwen3 report reports the same Qwen3-0.6B at **77.6** and
the DeepSeek card reports the same distilled model at **83.9**. A four-point
spread on one model is normal. A claim of "50" therefore needs the instrument,
not just the model name.

### A.2 What separates training from test-time compute

| comparison | number | what it isolates | source |
|---|---|---|---|
| Qwen3-0.6B thinking vs non-thinking, same weights | 77.6 vs 55.2 = **+22.4 MATH-500** | the effect of long CoT / thinking at inference | arXiv:2505.09388, Tables 19–20 |
| rStar-Math 3.8B vs its base Phi-3-mini | 41.4 → 86.4, with MCTS + PPM | training *and* search together; the paper's ablations attribute much of the gain to the search and the process model | arXiv:2501.04519 |
| DeepSeekMath-7B greedy vs self-consistency@64 | 51.7 vs 60.9 MATH | pure inference-time voting, no training change | arXiv:2402.03300 |
| self-consistency over CoT | +17.9 GSM8K (PaLM) | pure inference-time voting | arXiv:2203.11171 |
| compute-optimal TTC | on prompts where a smaller model has non-trivial success, TTC can beat a 14× larger model; >4× more efficient than best-of-N | TTC versus parameters | arXiv:2408.03314 |
| Qwen2.5-Math-1.5B-Instruct greedy vs RM@8 | 75.8 → 83.9 MATH | a reward model at inference, not training | arXiv:2409.12122, Table 3 |
| SimpleRL-Zoo 1.5B | 29.6 → 59.0 with GRPO, no voting | a training-time gain, but evaluated at temperature 1.0 and 16K generation | arXiv:2503.18892 |

The load-bearing comparison is the first row: **the largest single effect at
0.6B is not a training objective at all; it is whether the model is allowed to
think for thousands of tokens.** Our protocol forbids that. Any plan that
reaches 50 by "training better" must reproduce, in the weights and within 4,096
greedy tokens, a gain that Qwen3 obtains by spending inference compute.

### A.3 Data volume versus objective versus test-time compute

The cleanest published ablations:

| study | manipulation | result | source |
|---|---|---|---|
| OpenMathInstruct-2 | 14M question–solution pairs, ~600K unique questions; ablations on solution format, teacher strength, filtering, question diversity | verbose solutions are **detrimental**; a strong teacher beats an equally sized weak student; SFT is robust to imperfect filtering; **question diversity is crucial** for data scaling; Llama-3.1-8B-Base 51.9 → 67.8 MATH | arXiv:2410.01560 |
| Yuan et al. | SFT data amount; rejection-sampling fine-tuning (RFT) | log-linear relation between data amount and performance; better models improve less with more data; RFT helps **less-performant** models more; LLaMA-7B GSM8K 35.9 → 49.3 | arXiv:2308.01825 |
| "Small Models Struggle to Learn from Strong Reasoners" | long vs short CoT distillation at ≤3B | small models do **not consistently** benefit from long CoT or stronger-teacher distillation; they do better on shorter, simpler chains; mixing long and short is better than either alone | arXiv:2502.12143 |
| LIMO | ~1% of prior data (a few hundred to ~1K curated examples), 32B base | 63.3 AIME24, 95.6 MATH-500; argues the threshold is pre-trained knowledge plus "cognitive templates", not task complexity | arXiv:2502.03387 |
| s1 | 1,000 curated questions + budget forcing, 32B base | 50% → 57% AIME24 with test-time budget forcing | arXiv:2501.19393 |
| Qwen2.5-Math | 1T math pre-train + 2.5M CoT + 470K TIR + 66K RL queries | 75.8 greedy at 1.5B | arXiv:2409.12122 |

The pattern is consistent: **question diversity and teacher quality dominate data
volume per se; long chains are not automatically better for a small student;
test-time compute is a separate axis.** LIMO and s1 are widely cited as
"less is more", but both are 30–32B models with knowledge already in the
pre-trained base. They do not establish that 4,084 traces are enough at 0.8B.

### A.4 What this establishes and what it does not

**Establishes.**

1. MATH-500 ≥ 50 at ≤1B is not physically impossible: Qwen3-0.6B non-thinking
   reports **55.2** and MobileLLM-R1-950M reports **74.0**.
2. Every published route to those numbers uses either 200–1,500× our
   post-training data, or long-CoT/test-time compute, or both.
3. The closest protocol match — Qwen2.5-Math-1.5B-Instruct at 75.8 greedy —
   is a **1.5B** model with **1T math pre-training tokens** and **~3M SFT
   samples**. It is not evidence that a 0.8B model with 4,084 traces can reach
   50; it is evidence about what a 600× larger data budget buys.
4. Training-time gains at 1–1.5B are real but modest when measured under a
   fixed inference budget: SimpleRL-Zoo's 59.0 at 1.5B is a **+29.4** training
   effect, but at temperature 1.0 and 16K generation; the same model family at
   0.5B reaches only 34.4.
5. The 4,096 greedy non-thinking protocol is not the protocol under which most
   headline numbers were produced. A number is only comparable if it names the
   cap, the sampling, and whether a verifier or vote was used.

**Does not establish.**

1. That any of these numbers transfers to Qwen3.5-0.8B's tokenizer, template,
   or 4,096 cap. The harness spread on the same model is ±4 points.
2. That Qwen3-0.6B's **55.2** is a 4,096-token number. The report does not state
   the generation cap in the table this file read.
3. That a GSM8K gain implies a MATH gain. Our own SFT run gained GSM8K and lost
   MATH at 100%; the tasks are not interchangeable.
4. That a strong teacher's traces are learnable by a 0.8B student. The "Small
   Model Learnability Gap" is a direct warning against assuming that.

---

## B. Why preference learning on a tiny corpus collapses, and which fixes are real

### B.1 The objective family

| method | mechanism | reference model? | bounded margin? | reported effect | source |
|---|---|---|---|---|---|
| **DPO** | `−log σ(β [log π(y+)/πref(y+) − log π(y−)/πref(y−)])`; Bradley–Terry classification | yes | **no** — the log-sigmoid saturates but the implicit reward is unbounded | the baseline; stable on large preference sets | arXiv:2305.18290 |
| **IPO** | squared loss on the preference margin instead of log-sigmoid; identity transform | yes (in the general ΨPO form) | **yes**, the target margin is finite | motivated by DPO overfitting a deterministic preference; empirical gains on illustrative examples | arXiv:2310.12036 |
| **CPO** | contrastive preference objective with a reference regulariser; average log-likelihood reward | yes | partially | 22K parallel sentences, ALMA-R matches WMT winners | arXiv:2401.08417 |
| **ORPO** | monolithic SFT + odds-ratio preference penalty; no reference | **no** | partially (odds ratio) | 125M–7B; 12.20 AlpacaEval2 at 7B; the SFT term anchors the chosen side | arXiv:2403.07691 |
| **SimPO** | average log-prob as implicit reward + target reward margin; no reference | **no** | target margin yes, reward unbounded | +6.4 AlpacaEval2 over DPO; +7.5 Arena-Hard | arXiv:2405.14734 |
| **KTO** | prospect-theory utility on a binary desirable/undesirable signal; unpaired | **no** | utility is bounded | matches or exceeds preference methods at 1B–30B | arXiv:2402.01306 |
| **RPO (iterative reasoning PO)** | DPO **plus an NLL term on the chosen**; the NLL term is reported "crucial" | yes | no, but the NLL anchor prevents chosen-likelihood collapse | Llama-2-70B-Chat GSM8K 55.6 → 81.6; 88.7 with maj@32 | arXiv:2404.19733 |
| **APO** | anchored objective: `apo_zero` boosts winners and downweights losers; `apo_down` downweights both, stronger on losers | no explicit ref, but an explicit anchor term | controllable | 32K CLAIR pairs, +7.65% MixEval-Hard on Llama-3-8B-Instruct | arXiv:2408.06266 |
| **TR-DPO** | dynamically update the reference policy during training | yes, but moving | no | mitigates over-optimisation; surpasses conventional methods on dialogue/summarisation | arXiv:2404.09656 |
| **Robust DPO (rDPO)** | de-bias label flips; label-smoothing-style loss | yes | no | sub-optimality `O(1/(1−2ε) √(d/n))` for flip rate ε | arXiv:2403.00409 |
| **cDPO** | conservative DPO: label smoothing on the preference label | yes | no | the form is implemented in TRL (`label_smoothing`) and named "the cDPO report"; **no peer-reviewed cDPO paper or effect size was verified in this session** | TRL `DPOConfig` docs, v0.9.6 |
| **Length-regularised DPO** | regularise the length–quality correlation; LD-DPO token weighting | yes | no | "up to 20% improvement in win rates when controlling for length" | arXiv:2403.19159; LD-DPO is implemented as `ld_alpha` in TRL |
| **DPOP / Smaug** | add a term that prevents the chosen likelihood from falling | yes | no | shows DPO can reduce chosen likelihood, especially at low edit distance; DPOP fixes it; Smaug-72B >80% Open LLM | arXiv:2402.13228 |

The mechanism-level theory is not controversial:

* The DPO gradient is `β σ(−βm) ∇m`. Once the margin `m` is large, the
  log-sigmoid is saturated and the gradient is numerically zero. The loss can
  always be driven lower by separating the *remaining* pairs further. On a
  finite pair set with no held-out pairs, the minimum is a property of the
  pair set, not of the task.
* The gradient is asymmetric: Feng et al. show theoretically that DPO decreases
  the probability of dispreferred data faster than it increases the probability
  of preferred data (arXiv:2404.04626). Smaug shows empirically that the chosen
  likelihood can fall (arXiv:2402.13228). The practical consequence is that a
  saturated DPO run can suppress *both* sides and leave the language manifold.
* The implicit reward model generalises worse than an explicit one under
  distribution shift: mean 3%, max 7% accuracy drop across five OOD settings
  (arXiv:2409.03650).
* The SFT stage, not the preference stage, carries most of the alignment
  effect and the data-scaling effect in a controlled benchmark
  (SOS-Bench, arXiv:2409.15268). This is a warning against expecting a
  preference stage to add capability that SFT did not.

### B.2 What happened in S69, in the language of the theory

Our run is not an unexplained anomaly. It is the textbook failure:

| observation (S69) | mechanism it is consistent with |
|---|---|
| loss 0.6285 → 5.85e-05 by step 1,294; implied mean margin ≈97 nats | margin saturation on 214 pairs; the log-sigmoid has no bound on the margin |
| first step below 0.05 at step 61, inside the first third of the first epoch | the easy pairs saturate almost immediately; the remaining steps chase the hard pairs |
| 20.9% clipping, pre-clip norm p90 4.24, max 95.5 | after easy pairs contribute ~0, the gradient is dominated by a few hard pairs; with `batch_pairs = 1` the update direction is one pair's `∇m`, and clipping turns it into a fixed-size step |
| 100% cap / 0% EOS on unseen calibration prompts at every cadence | off-manifold policy: the rejected continuations were suppressed so hard that the model cannot terminate |
| CE controls on the identical chosen side do not collapse | the corpus is not the cause; the objective and the update regime are |
| collapse is worse at 500K/1M than at 2M | the direction is bad from the start; this is not an end-of-training artefact and not a pure magnitude problem |

The last row is important. The CE arms improve as the update gets smaller; the
preference arm is worst at the smallest update. A pure "the update is too big"
story does not cover the arm it was invented to explain. The objective is
producing a direction that leaves the manifold at every magnitude tested.

### B.3 Which remedy would I predict to work, for this exact geometry?

The geometry: 214 positive-weight pairs, 6 epochs at 2M tokens, β=0.1,
mean-normalised sequence log-probs, reference anchored at the initialisation,
`batch_pairs = 1`, lr 1e-4, rank-64 all-linear LoRA, merge after training.

I would predict, in order:

1. **RPO-style NLL anchor + batching + one epoch + β ≥ 0.5** (arXiv:2404.19733;
   TRL `rpo_alpha`; arXiv:2408.06266 for the anchored view). The NLL term on
   the chosen side is the only published fix that directly opposes the
   "suppress both sides" failure, and RPO reports it as *crucial* for reasoning.
   Batching removes the single-pair gradient variance that produced the 95.5
   pre-clip spikes. β=0.5–1.0 penalises a given log-ratio more strongly, so the
   policy is pushed less far from the reference before the gradient saturates.
   One epoch keeps the pair set from being exhausted. Predicted outcome: **no collapse**
   (EOS present, cap fraction in the CE-control range), but **at most +1–2
   MATH-cal**, because 214 pairs are a regularisation signal, not a capability
   signal.
2. **IPO** (arXiv:2310.12036) as the bounded-margin control. It is the
   cleanest theoretical fix for saturation. Predicted outcome: stable, but
   underfits if the target margin is small; likely a null on capability.
3. **ORPO** (arXiv:2403.07691) as the reference-free monolithic alternative.
   Its SFT term anchors the chosen side by construction. Predicted outcome:
   stable; hard to attribute because the SFT term is also extra chosen-side
   compute.
4. **TR-DPO** (arXiv:2404.09656) would make it **worse** for us: moving the
   reference toward a collapsing policy removes the only anchor.
5. **cDPO / Robust DPO** (arXiv:2403.00409) addresses label noise. Our labels
   are verified by a checker, so the noise model is wrong; it adds bias without
   addressing saturation. Not recommended.
6. **SimPO / CPO** address length bias and reference cost, not the unbounded
   margin. They would not fix this collapse; they might change its shape.

**What would falsify the prediction.** Run the NLL-anchored arm at a matched
*realised* adapter norm (the L2 distance from the init adapter, not processed
tokens) and score generation behaviour. If it still hits 100% cap and 0% EOS,
then the collapse is not margin saturation. The remaining explanations are
(i) the rejected continuations are so far off-distribution that any contrastive
update leaves the manifold, or (ii) the rank-64 adapter cannot represent the
correction and the optimiser finds a degenerate direction instead. In that case
the correct conclusion is to abandon preference learning on this corpus, not to
try a sixth objective.

**The cheapest decisive test is not a retrain.** The failed preference
checkpoint already exists. Interpolate its delta into the soup at
α ∈ {0.02, 0.05, 0.1, 0.2}, and score MATH-cal plus the cap/EOS diagnostics. If
EOS returns and MATH-cal is at or above the soup at some α, the direction has
signal and a properly anchored objective is worth one retrain. If no α restores
EOS, the direction is bad and the honest move is to stop. This costs about
4 GPU-hours of scoring and does not touch the GPU for training. It is ranked as
G6 below.

### B.4 What this establishes and what it does not

**Establishes.** The S69 collapse is a known failure mode with a known
mechanism (margin saturation plus single-pair gradients), and the published
fixes are NLL anchoring, bounded margins, reference-free monolithic objectives,
and trust-region/reference updates. The CE controls rule out the corpus as the
cause.

**Does not establish.** That any of these fixes produces a capability gain on
214 pairs. The literature's effect sizes are for preference *alignment* at
7B–70B with tens of thousands of pairs; none of them is a MATH gain at 0.8B
from 214 pairs. The fixes are about not destroying the model, not about making
it smarter.

---

## C. Is LoRA the right capacity for a multi-point reasoning change?

### C.1 Sources

| source | finding | relevance to us |
|---|---|---|
| **LoRA Learns Less and Forgets Less** (arXiv:2405.09673) | on **programming and mathematics**, with ~100K instruction pairs and 20B continued-pre-training tokens, standard low-rank LoRA **substantially underperforms full fine-tuning**; LoRA forgets less; **full FT learns perturbations with rank 10–100× greater than typical LoRA** | the closest published ablation to our domain; rank-64 is at the low end of the rank range it studies |
| **LoRA vs Full Fine-tuning: An Illusion of Equivalence** (arXiv:2410.21228) | LoRA produces new high-ranking singular vectors ("intruder dimensions") that full FT does not; forgetting is localised to the intruder dimensions; scaling them down improves pre-training-distribution modelling with minimal downstream loss | a mechanistic reason LoRA can look fine on the training task and break elsewhere; also a reason a LoRA delta and a full-FT delta are not interchangeable |
| **LoRA+** (arXiv:2402.12354) | using the same learning rate for A and B is suboptimal at large width; separate LRs give 1–2% gains and up to ~2× speed | if LoRA is used, the optimiser is part of the capacity question |
| **LoRA-Pro** (arXiv:2407.18242) | LoRA is mathematically full FT with a low-rank gradient; correcting the low-rank gradient narrows the gap to full FT on mathematical reasoning | the gap is not only rank; it is also gradient approximation |
| **LoRA-SB / LoRA-One** (arXiv:2411.19557; arXiv:2502.01235) | better initialisation approximates full FT within a low-rank subspace; claims to close the gap with 27–90× fewer parameters | rank alone is not the whole story; initialisation matters |
| **Intrinsic dimensionality** (arXiv:2012.13255) | 200 trainable parameters projected into the full space reach 90% of full-FT performance on MRPC; larger pre-trained models have lower intrinsic dimension | low-dimensional updates can be sufficient for NLU; this is **not** evidence for multi-step math at 0.8B |
| **LoRI** (arXiv:2504.07448) | sparse low-rank adapters claim to outperform full FT with up to 95% fewer parameters in multi-task settings | a counterpoint, but not a math-specific single-task ablation |
| **LoRA init under RLVR** (arXiv:2606.31813) | PiSSA/MiLoRA, which help under SFT, can underperform or destabilise under RLVR; orthonormal init is better | if we ever do RL with LoRA, the init is a live variable |

### C.2 Verdict for our case

* **Rank-64 all-linear is probably sufficient for a small correction and not
  demonstrably sufficient for a multi-point reasoning gain.** The published
  math ablation (LoRA Learns Less) uses ~100K pairs; we use 214. It finds
  standard low-rank LoRA substantially behind full FT and full FT's update rank
  10–100× higher. Nothing in that paper says rank 64 is enough for +10 MATH.
* **Our S69 preference run is not evidence about LoRA capacity**, because the
  objective collapsed. But the two CE controls *were* rank-64 all-linear on the
  same corpus, and they did not gain: MATH-cal −0.024 to −0.061, GSM8K-cal
  −0.013 to −0.067. At this data volume and this adapter geometry, the
  intervention did not add capability. That is a data-volume result as much as
  a capacity result; it cannot separate the two.
* **Full FT is the right default for a capability delta.** Our one large
  successful delta (S58) was full FT over 498,113,344 elements. The preference
  arm's LoRA was a second experiment with a different objective; the capacity
  and the objective changed at once, which is a design confound.
* **If capacity is the question, the decisive experiment is G8**: same SFT
  corpus and token budget, arms = full FT, rank-64 all-linear, rank-256
  all-linear, rank-64 MLP-only; each with the α ladder. Prediction: full FT wins
  on fit, interpolation recovers retention, and no LoRA arm closes the gap at
  10.94M tokens. If that prediction holds, stop tuning LoRA ranks and spend the
  budget on data.

### C.3 What this establishes and what it does not

**Establishes.** There is a published, math-specific gap between LoRA and full
FT at standard ranks, with a mechanistic explanation (intruder dimensions,
low-rank gradient approximation). Rank 64 is not a safe default for a
multi-point reasoning delta.

**Does not establish.** That rank 64 caused the S69 collapse. The CE controls
show that rank-64 LoRA can train without collapsing. The collapse is the
objective's.

---

## D. Delta manipulation: soup, arithmetic, interpolation, and what our overshoot implies

### D.1 Sources

| method | what it does | source | implication for a measured 37.5%-helps / 100%-hurts case |
|---|---|---|---|
| **WiSE-FT** | average the zero-shot and fine-tuned weights, `θ(t) = (1−t)θ0 + tθ1` | arXiv:2109.01903 | the canonical "the fine-tuned endpoint overshoots; interpolate back" result; +4–6 pp under distribution shift, +1.6 pp in-distribution on ImageNet |
| **Model Soups** | average the weights of multiple fine-tuned runs | arXiv:2203.05482 | averaging multiple runs improves accuracy without inference cost; the models lie in a single low-error basin |
| **Task arithmetic** | task vector `τ = θ_ft − θ0`; add, negate, scale | arXiv:2212.04089 | the coefficient is a behaviour dial; negation decreases target performance with little change on control tasks |
| **TIES-Merging** | trim small deltas, elect a sign per parameter, merge only sign-agreeing parameters | arXiv:2306.01708 | sign conflicts between runs are a real interference source; resolving them beats naive averaging |
| **DARE** | randomly drop 90–99% of delta parameters and rescale the rest | arXiv:2311.03099 | SFT deltas are extremely redundant; sparsification can merge homologous models without retraining |
| **Linear mode connectivity** | after early training, SGD noise outcomes lie in a linearly connected region | arXiv:1912.05671 | interpolation is expected to work when the endpoints share an init/early trajectory — which our three seeds do |
| **Git Re-Basin** | align models modulo permutation symmetry before averaging | arXiv:2209.04836 | independently trained models need alignment; our seeds share an init, so this is a control, not a requirement |
| **Intrinsic dimensionality** | fine-tuning lives in a low-dimensional subspace | arXiv:2012.13255 | supports the idea that a delta can be projected/truncated; does not say which directions are useful |
| **Task arithmetic in the tangent space** | weight disentanglement arises from pre-training; linearisation amplifies it | arXiv:2305.12827 | direction, not just norm, carries task identity; linearisation is a tool for cleaner arithmetic |
| **Illusion of Equivalence** | LoRA and full FT have different spectral structure; LoRA's intruder dimensions cause forgetting | arXiv:2410.21228 | a LoRA delta and a full-FT delta are not interchangeable; rank-truncating a full-FT delta is a genuinely different operation from training LoRA |

### D.2 The three questions the operator actually asked

**(i) Could a different training regime produce a delta that is safe at 100%?**

Yes — but the published examples change the **data and the objective**, not just
a hyperparameter. Qwen2.5-Math-1.5B-Instruct and DeepSeek-R1-Distill-Qwen-1.5B
deploy 100% SFT endpoints that are strong; they use 800K–3M samples and, in
Qwen's case, 1T math pre-training tokens plus RL. Our 4,084-trace, 2B-teacher
endpoint is harmful at 100%. Nothing in the delta-manipulation literature says
that a learning-rate or schedule change alone makes the same content safe at
100%; WiSE-FT exists precisely because the endpoint overshoots and the *post hoc*
coefficient is the reliable knob. The honest prediction is: **a stronger teacher
or more diverse verified data could produce a delta that is safe at 100%, but
that is an empirical question answered by the α ladder, not an assumption.**

**(ii) Is norm or direction the controlling variable?**

Both, but for **overshoot** the norm is the knob and for **what capability
changes** the direction is. The evidence in our own receipts is unusually clean:

* The three seed deltas have almost identical norms (13.4170 / 13.4162 / 13.4170)
  and pairwise cosines of about 0.32. They are *different directions* of the
  *same length*.
* The seed-0 grid and the soup both improve at α ≈ 0.25–0.375 and degrade at
  α = 1.0. The soup improves over the base at α=0.375 (+2.52 full MATH,
  interval above zero).
* Therefore the harm at α=1.0 is not "the direction is wrong for seed 0 but
  right for seed 1": the three directions are different (cosine 0.32), the
  seed-0 endpoint is harmful at full magnitude, and the soup degrades
  monotonically beyond 0.375 — yet the mean direction is helpful at 0.375.
  That is a magnitude effect shared across directions. Direction changes
  *which* capability moves; norm changes *how far*.

The controls that would settle it: a norm-matched random direction, a
sign-flipped delta, and a rank-truncated delta renormalised to the same L2 norm.
If a random direction of norm 13.42 also improves MATH, the effect is a
regularisation/flatness effect, not a task direction. If it does not, direction
matters. If the rank-truncated delta at the same norm recovers the gain without
the loss, the useful part is low-rank and the harmful part is high-rank.

**(iii) Does merging multiple seeds' deltas dominate interpolating their soups?**

For the **mean**, the two operations are algebraically identical:
`mean_s(θ_s) − θ0 = mean_s(δ_s)`, so the S67 soup *is* the average delta.
There is no separate "merge" to dominate. The non-identical operations are
**sign-resolving** (TIES) and **sparsifying** (DARE). Model Soups shows that
averaging multiple runs helps; our cosines of 0.32 mean sign conflicts are
substantial, so TIES has something to resolve. But the expected effect is small:
the mean is already a variance-reduced estimator, and all three seeds share the
same data and objective. The decisive test is TIES vs mean vs DARE at matched
norm on the 1,000-item calibration; a >1-point MATH-cal gain with a paired
interval above zero would say sign resolution matters.

### D.3 What this establishes and what it does not

**Establishes.** Interpolation, soups, task arithmetic and sign-resolved merging
are established, with the coefficient as the standard safety knob. Our
37.5%-helps / 100%-hurts observation is the expected shape of a fine-tuning
delta, not an anomaly. The three-seed evidence points to norm as the overshoot
variable.

**Does not establish.** That any manipulation of the *existing* delta reaches
50. The existing delta's best calibration point is ~35.0 MATH-cal; the target
requires a different delta. This is the central reason the recommendation below
is a new-data experiment, not a merging experiment.

---

## E. On-policy distillation versus offline distillation at small scale

### E.1 Sources

| method | student / teacher | signal | reported effect | compute | source |
|---|---|---|---|---|---|
| **GKD** | T5 77M–800M students, T5-XL (~3B) teacher; also instruction tuning | on-policy: train on **student-generated** sequences with teacher token probabilities; forward KL, reverse KL, JSD | **1.9× relative gain on arithmetic reasoning** over baseline KD; on-policy GKD on 5% of XSum beats supervised KD on 100% | on-policy sampling, but the paper reports data efficiency gains | arXiv:2306.13649 |
| **MiniLLM** | 120M–13B | **reverse KL**, on-policy policy gradient | better precision, lower exposure bias, better calibration and long-text generation than baselines | on-policy | arXiv:2306.08543 |
| **DistiLLM** | LLM students | skew KL + adaptive off-policy use of student outputs | up to **4.3× speedup** over recent KD methods, better instruction following | lower than on-policy KD | arXiv:2402.03898 |
| **CCE-OPD** (preprint) | Qwen3-8B → 0.6B and 1.7B students | condition-contrastive reweighting of teacher supervision at student prefixes | **+1.60 / +1.95 avg@8 over OPD** at 0.6B / 1.7B; invalid-continuation preference 27.0% → 16.0% | not stated in the abstract | arXiv:2610.21195 |
| **SCOUT / On the Off-Policy Teacher** (preprint) | teacher–student pairs | the teacher is optimised for its own prefixes, not the student's; SCOUT periodically adapts the **teacher** to student-generated prefixes with RLVR | teacher continuation degrades as student prefixes grow; SCOUT improves OPD effectiveness across configurations | additional teacher training | arXiv:2609.38360 |
| **Supervision Fidelity Decay** (preprint) | 7B student | shows teacher confidence/discrimination decays as student prefixes lengthen; adds a lookahead group reward | **+2.57 mean@8 over OPD** for a 7B student; +4.92 on AIME-26 at 39K tokens | entropy-triggered tree attention | arXiv:2605.30833 |
| **TIP** (from the program's S60 review) | — | token importance: high student entropy, or low entropy with high teacher–student divergence | retaining 50% of tokens matches all-token training; up to 47% peak-memory reduction | — | arXiv:2604.14084, quoted in `docs/S60-LITERATURE.md` |

### E.2 Is dense token-level distillation on teacher continuations at student prefixes the right signal?

**In principle, yes.** GKD's framing is the correct one: supervised KD on a
fixed set of teacher sequences is imitation learning with a fixed state
distribution, and the student's inference distribution differs. Training on
student-generated prefixes with teacher token probabilities directly addresses
exposure bias. MiniLLM's reverse KL is the mode-seeking version of the same
idea, and the 2026 student-prefix literature (CCE-OPD, SCOUT, SFD) is a serious
attempt to fix the obvious weakness: the teacher's next-token distribution
degrades as the student's prefix gets longer and weirder.

**But the published evidence that it beats plain SFT on teacher solutions is
modest at our scale.** The arithmetic-reasoning evidence is T5 77M–800M with a
3B teacher (GKD), not a 0.8B decoder-only math model. CCE-OPD's +1.6–2.0 is over
OPD, not over SFT. MiniLLM's gains are on instruction following and long text.
Against that, the largest published effects at 1–1.5B are still **offline SFT
on strong-teacher traces** (R1-Distill-Qwen-1.5B: 83.9 MATH-500 from 800K
traces; Qwen2.5-Math-1.5B: 75.8 greedy from ~3M samples). The ordering is:
teacher/data quality first, offline SFT second, OPD as a refinement third.

**The practical constraint that decides it for us:** token-level OPD needs a
shared tokenizer and teacher logits at every student prefix. A text-only
teacher is enough for SFT and not enough for OPD. If a same-family teacher is
available (Qwen3-4B and Qwen3.5-0.8B plausibly share a tokenizer, but this was
**not verified**), a branch-local OPD arm is feasible; if not, the choice is
made for us. The branch-local corpus in `docs/TODO.md` C1 is the right shape
for OPD because the prefixes are short, which is exactly the regime where the
teacher's continuation is still reliable.

### E.3 What this establishes and what it does not

**Establishes.** OPD is a real, published improvement over offline KD at small
student sizes, with GKD's 1.9× arithmetic result and the 2026 student-prefix
work. Dense token-level supervision at student prefixes is the right *signal
design* if the teacher and tokenizer allow it.

**Does not establish.** That OPD beats SFT on strong-teacher math traces at
0.8B, or that it is worth doing before the teacher/data bottleneck is fixed.
The published math gains at 1.5B come from data and long CoT, not from the
divergence choice.

---

## F. RLVR / GRPO at ~1B on a single 16 GB consumer GPU

### F.1 Sources

| run | model | data / rollout budget | compute | MATH effect | stability | source |
|---|---|---|---|---|---|---|
| **SimpleRL-Zoo** | Qwen2.5-0.5B / 1.5B / 7B / 14B / 32B, zero RL | ~8K GSM8K/MATH problems; 8 rollouts/prompt; 8,192 max rollout; prompt batch 1,024; clip 0.2; KL 1e-4 for ≤14B | GPU count not stated in the HTML this file read | 1.5B: 29.6 → **59.0** MATH-500; 0.5B: 15.8 → **34.4** | response length and clip ratio monitored; "aha moment" not always correlated with length | arXiv:2503.18892 |
| **1.5B RL under a 24 h budget** | DeepSeek-R1-Distill-Qwen-1.5B | 7K samples; 6 generations/prompt; 4,096 max completion; 500 global steps | **4×A40 48 GB, 24 h** (~96 GPU-h); $42 at the cited rate | **83 → 84** MATH-500; AMC23 63 → 80 | unstable after ~200 steps; accuracy drops below 60 on AMC23, to 80 on MATH-500; lengths oscillate | arXiv:2503.16219 |
| **DeepScaleR-1.5B** (as quoted) | 1.5B | 40K problems × 16 samples | **$3,629** (quoted in arXiv:2503.16219) | not verified here | — | arXiv:2503.16219, Table 2 discussion |
| **ProRL** | Nemotron-Research-Reasoning-Qwen-1.5B | prolonged RL, diverse tasks | not stated in the abstract | improves pass@k beyond the base, including tasks the base fails at any k | KL control + reference-policy resetting | arXiv:2505.24864 |
| **Does RL really incentivize…** | multiple families, math/code/visual | pass@k at large k | — | RLVR beats the base at small k but **the base wins at large k**; current RLVR does not elicit fundamentally new patterns | — | arXiv:2504.13837 |
| **Spurious Rewards** | Qwen2.5-Math-7B and others | random / spurious rewards | — | random rewards give **+21.4** MATH-500 vs **+29.1** with ground truth on Qwen2.5-Math-7B; gains are highly model-dependent and fail on Llama3/OLMo2 | clipping bias amplifies high-prior behaviours | arXiv:2506.10947 |
| **DAPO** | Qwen2.5-32B | large-scale RL | — | 50 AIME 2024 | clip-higher, dynamic sampling, token-level loss, overlong reward shaping | arXiv:2503.14476 |
| **Entropy mechanism** | reasoning LMs | — | — | empirical law `R = −a e^H + b`: performance is traded from entropy and bottlenecked by its exhaustion | Clip-Cov / KL-Cov to avoid entropy collapse | arXiv:2505.22617 |

### F.2 The arithmetic on our hardware

Our measured generation throughput on this exact model and instrument is
**668 tokens/s** (S67 full MATH: 9,006,620 tokens / 13,495.5 s;
`docs/S67-RESULT.md` §6). One GPU-hour is 2.405M generated tokens at that rate.

| run | generated tokens per step (estimate) | tokens for 100 steps | GPU-hours on one 4080 at 668 tok/s |
|---|---|---|---|
| 1.5B run with 6 generations × 6 prompts/device × 4 devices, ~3K tokens (arXiv:2503.16219) | ~432K | — | the full 500-step run was ~216M tokens = **~90 GPU-h** (arithmetic from the cited 24 h × 4 A40 and ~3K average completion) |
| SimpleRL-Zoo-style, 8 rollouts × 1,024 prompts, ~2K tokens | ~16.4M | 1.64B | **~680 GPU-h** (28 days) |

The first run is the honest anchor because it is the one that fits a 24-hour
wall clock on four 48 GB cards. On one 4080 it is about **3.7 days**, and its
MATH-500 effect was **+1 point**, with degradation after 200 steps. The second
run is the recipe that produced +29.4 at 1.5B, and at our throughput it is
**hundreds of GPU-hours**, not days.

### F.3 Is a single 16 GB GPU enough for a multi-point MATH gain from RLVR?

**No, not at the published rollout budgets, and not in days.** Three independent
reasons:

1. **Compute.** The one published RL run at 1.5B that used a constrained
   wall clock (4×A40, 24 h) produced +1 MATH-500 on an already-strong distilled
   model. The run that produced +29.4 used a rollout budget that is ~7× larger
   per step and hundreds of steps. At 668 tok/s, that is not a days-scale
   experiment.
2. **Memory.** Full-parameter GRPO at 0.8B with 8 rollouts and 8K context needs
   the policy, the reference, the optimizer state, the activations for the
   update, and the KV cache for sampling. That does not fit in 16 GB without
   offload or quantization. LoRA RL fits, but then the capacity question from
   §C returns, and the 2026 RLVR-LoRA literature says initialization matters and
   can destabilise (arXiv:2606.31813).
3. **Signal.** On Qwen2.5-Math-7B, random rewards give +21.4 of the +29.1 MATH
   gain (arXiv:2506.10947). A positive RL result on a Qwen model is not by
   itself evidence of a reasoning gain; it can be clipping bias amplifying a
   pre-trained behaviour. And the pass@k analysis says the base model's
   coverage bounds what RLVR can reach; distillation is the intervention that
   expands coverage (arXiv:2504.13837).

A cut-down RLVR run is feasible: LoRA, 2K problems, K=4, 2K tokens ≈ 16M
tokens per epoch ≈ 6.6 GPU-hours, 2–3 epochs ≈ 20 GPU-hours. The published
evidence at a comparable budget predicts **+0 to +2**, with a real risk of
length explosion and entropy collapse. It is not the best use of the next
GPU-days.

**Blunt verdict:** RLVR at 1B on one 16 GB card is a real method with real
published gains, but the gains that matter were bought with rollout budgets this
machine cannot pay in days. A multi-point MATH gain from RLVR here is a fantasy
at the published budgets, and a coin-flip at a 20-GPU-hour budget.

---

## G. The uncombined and inverted opportunities

Ranking rule: **expected MATH effect × probability of working / GPU-hours**.
The numbers are judgement calls, not measurements; the cost arithmetic is shown.
The ranking is fragile among the top four (all ≈0.09–0.16), so the tie-break is
**ceiling**: only G1 has a published analogue that could plausibly produce the
+10 points the target needs; G2 is the only other candidate with a plausible
+5–6 ceiling; G3–G8 are safety or variance-reduction experiments on a delta
whose best measured effect is +2.5 full MATH.

Cost basis: generation 668 tok/s and full-FT training 2,081 tok/s, both from
program receipts (`docs/S67-RESULT.md` §6; `docs/S58-STAGE2A.md` §3). One full
calibration point (1,000 GSM8K-cal + 1,000 MATH-cal + 2,000 MMLU-cal) is about
**1.0 GPU-hour** at those rates (the S69 soup re-score generated 1,806,060
tokens over 2,000 GSM8K+MATH items; MMLU is short). A 300-item screen is about
0.3 GPU-hours.

| rank | # | mechanism | components (citations) | why it is not already standard | cheapest decisive experiment | kill criterion | confound that makes a positive uninterpretable | E | P | GPU-h | E·P/C |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | **G3** | **Rank-truncated delta application (delta-SVD)** | LoRA Learns Less (arXiv:2405.09673); Illusion of Equivalence (arXiv:2410.21228); intrinsic dimensionality (arXiv:2012.13255); task arithmetic (arXiv:2212.04089) | LoRA constrains *training* to low rank; truncating a *full-FT* math delta post hoc and measuring the rank–accuracy curve at 1B is not published | SVD each weight delta, keep top-r, renormalise to the S67-selected norm; r∈{32,64,128,256,full}; score | monotone improvement with r (no harmful high-rank component) or no r beats α=0.375 | truncation changes the norm; a gain may be a norm effect, not a rank effect | 2.5 | 0.35 | 5.5 | **0.159** |
| 2 | **G4** | **Short/long CoT mix curriculum SFT** | Small Model Learnability Gap (arXiv:2502.12143); OpenMathInstruct-2's format/diversity findings (arXiv:2410.01560); Yuan et al. data scaling (arXiv:2308.01825) | we have only long 2B-teacher traces; the small-model literature predicts short student-reachable chains will beat long teacher chains, and the mix is not standard | 50/50 mix of short student-correct and long teacher traces at matched 10.94M tokens; α ladder | no α beats the long-only arm at matched tokens | the mix arm gets extra student data; a gain could be self-distillation, not the short/long balance | 3 | 0.40 | 9.5 | **0.126** |
| 3 | **G2** | **Self-consistency / verifier-selected trace distillation (TTC → weights)** | STaR (arXiv:2203.14465); ReST^EM (arXiv:2312.06585); Distilling System 2 (arXiv:2407.06023); rStar-Math's MCTS-to-policy (arXiv:2501.04519); self-consistency (arXiv:2203.11171) | rStar distils MCTS into a 7B policy with millions of solutions; the cheap 0.8B version under a 4,096 cap, using our own verifier, is not published | 1,000 train problems × K=8 from the champion; keep modal-correct traces; full FT; α ladder | greedy gain < 1 point at matched tokens | the majority label is correct on train but the trace is post-hoc; the model may learn answer-only mapping; K=8 coverage may be the ceiling | 4 | 0.45 | 15.5 | **0.116** |
| 4 | **G7** | **TIES/DARE sign-resolved merge of the three seed deltas** | TIES (arXiv:2306.01708); DARE (arXiv:2311.03099); Model Soups (arXiv:2203.05482); LMC (arXiv:1912.05671) | our soup is the plain mean; cosine 0.32 means sign conflicts are real; TIES has not been tried at this scale on math | TIES vs mean vs DARE at matched norm; score | <1 point over mean with a paired interval crossing zero | TIES/DARE change the norm and sparsity; match the norm to the mean delta | 1 | 0.30 | 3.2 | **0.094** |
| 5 | **G5** | **Bounded-margin, NLL-anchored, batched preference learning at small realised norm** | RPO/NLL (arXiv:2404.19733); IPO (arXiv:2310.12036); APO (arXiv:2408.06266); ORPO (arXiv:2403.07691) | our S69 arm used unbounded log-sigmoid, β=0.1, batch=1, 6 epochs; the reasoning-specific NLL-anchored variant has not been run from the champion | 1 epoch, batch_pairs=16, β=0.5, NLL weight 0.5, realised-norm cap; score + cap/EOS diagnostics | still 100% cap / 0% EOS at matched realised norm | the NLL anchor is also extra chosen-side compute; control with the existing matched-SFT arm | 1.5 | 0.25 | 4.7 | **0.080** |
| 6 | **G1** | **Strong-teacher long-CoT distillation + measured delta-fraction ladder** | R1 distillation (arXiv:2501.12948); Qwen2.5-Math data recipe (arXiv:2409.12122); MobileLLM-R1 (arXiv:2509.24945); WiSE-FT (arXiv:2109.01903); Mix Distillation (arXiv:2502.12143) | distillation papers deploy the 100% SFT endpoint; WiSE-FT is used for robustness, not to rescue distillation overshoot; nobody publishes an α/rank ladder on a strong-teacher math delta at 0.8B under a 4,096 greedy protocol | Stage 0 qualify teachers on 250 MATH-train prompts at 4,096; Stage 1 generate 4,000 verified traces (~10.94M tokens); Stage 2 full FT; Stage 3 α∈{0.125…1.0} and rank∈{64,256,full} | no α/rank reaches MATH-cal ≥ soup + 0.02 with MMLU-cal retention; or the best point is no better than the 2B-teacher arm at matched budget | teacher contamination with MATH test items; trace length rather than reasoning; α selected on the same calibration it is scored on | 8 | 0.40 | 48 | **0.067** |
| 7 | **G8** | **Higher-rank / MLP-only LoRA SFT at matched tokens** | LoRA Learns Less (arXiv:2405.09673); LoRA+ (arXiv:2402.12354); LoRA-Pro (arXiv:2407.18242); LoRA-SB (arXiv:2411.19557) | our only LoRA run was the preference arm; no matched SFT LoRA-vs-full-FT ablation at 0.8B on math | full FT vs rank-64 vs rank-256 all-linear vs rank-64 MLP-only; α ladder | LoRA never within 1 point of full FT at its best α | LoRA's lower memory allows a larger batch/LR; match effective batch and schedule | 2 | 0.25 | 17 | **0.029** |
| 8 | **G6** | **Post-hoc interpolation of the failed preference delta** | WiSE-FT (arXiv:2109.01903); task arithmetic (arXiv:2212.04089); our S69 collapse (`docs/S69-RESULT.md`) | the arm was discarded; nobody asks whether a small fraction of a collapsed preference delta is a useful correction | α∈{0.02,0.05,0.1,0.2} of the preference delta into the soup; score MATH-cal + EOS/cap | no α restores EOS with MATH-cal ≥ soup | at small α the model is nearly the soup; a "gain" may be noise | 0.5 | 0.20 | 4 | **0.025** |

### G.1 Why the ranking and the recommendation diverge

Five candidates (G3, G4, G2, G7, G5) have better ratios than G1. If the
objective were "maximise information per GPU-hour", G3 would be first: it is a
zero-training diagnostic that tells us whether the useful part of our own delta
is low-rank. But the program's objective is a **threshold**: MATH-500 ≥ 50 and
full MATH ≥ 50. The champion from this delta is 40.60 / 42.12, and the best
single-seed MATH-500 point is 42.80; neither is 50. G3,
G6 and G7 operate on that delta; their ceiling is the delta's ceiling, about
+2.5 full MATH. G4 changes the data but keeps a 2B/student teacher; its ceiling
is plausibly +3–5. G2's ceiling is the student's own K-sample coverage, plausibly
+4–6. G5 is a stability experiment on 214 pairs and cannot exceed +1–2. Only G1
changes the teacher and therefore the delta's content, and only G1 has a
published analogue at 1.5B that reaches 83.9 MATH-500.

So: **ranked by the requested ratio, G3 first; recommended, G1, sixth.** The
operator should know that this is a deliberate departure from the ranking. The
ratio rewards cheap diagnostics; the threshold rewards the only mechanism with
a ceiling above the gap. If the operator's rule is "never spend two GPU-days on
one arm", then G3 + G4 + G2 in sequence is the right reading of this table and
the target should be treated as a two-phase programme rather than a single run.

If the operator wants a cheap gate before spending 48 GPU-hours, run G6 (4
GPU-hours) and G3 (5.5 GPU-hours) first. They do not test the same hypothesis,
but they cost less than a day and they close two live questions: whether the
preference direction has any signal, and whether our own delta's harm is
high-rank. Neither can reach 50.

---

## H. The recommended next experiment

### S72 — strong-teacher distillation with a measured delta-fraction ladder

**One-sentence claim under test.** Replacing the 2B teacher with the strongest
teacher that fits 16 GB and 4,096 tokens, at a matched 10.94M-token budget,
produces a delta whose best safe fraction beats the 2B-teacher delta's best safe
fraction by enough to move MATH-500 toward 50.

This is one experiment with a qualification stage, a generation stage, a
training stage, and a selection ladder. It is registered as one question, with
one selection instrument and one one-shot test.

**Execution.** Every stage runs through `scripts/docker-court.sh` with an
explicit, workload-sized `SELVEDGE_MEM` derived from the preceding smoke receipt,
plus `SELVEDGE_CPUS`, `SELVEDGE_PIDS` and a bounded `SELVEDGE_TMP`. No stage may
inherit the 2g CPU-court default. Each receipt records the cgroup
`memory.peak`/`memory.events` and the GPU peak, and a mismatch fails closed.

#### H.1 Stage 0 — teacher qualification (before any long generation)

**Purpose.** Do not spend 30–45 GPU-hours generating traces from a teacher that
cannot solve our problems at our cap. The published headline numbers for 4B/7B
teachers are at their own generation settings, not at 4,096 tokens with our
template.

**Candidates.** (a) Qwen3-4B, non-thinking; (b) Qwen3-4B, thinking; (c)
Qwen2.5-Math-7B-Instruct; (d) the existing 2B teacher, as the control. All at
4-bit weight-only quantization, bf16 KV, on the same GPU.

**Instrument.** The first 250 MATH-train and 250 GSM8K-train problems after the
same exclusions as S58 (S58 holdout, S66 calibration ids). Protocol: S54 user
template, system instruction "Please reason step by step, and put your final
answer within \boxed{}.", greedy, `max_new_tokens=4096`, batch 8, one sample per
problem. Verify with `frontier.rft.verify` against the train label.

**Gate.** The chosen teacher must solve **≥ 70% of MATH-train** and **≥ 90% of
GSM8K-train** at 4,096. If no candidate clears, stop: the 4,096 cap, not the
teacher, is the binding constraint, and the right next experiment is a
cap/format study, not a distillation run.

**Receipt.** Teacher id and revision, quantization, tokenizer hash, per-problem
correctness, generated tokens, wall time, throughput, peak VRAM, and the
selected teacher. This receipt also sizes Stage 1's memory and time.

**Cost.** 0.5M generated tokens per candidate at roughly 100–150 tok/s for a
4-bit 4B/7B model ≈ 1–1.4 h per candidate; four candidates ≈ **4–6 GPU-hours**.

#### H.2 Stage 1 — trace generation

**Data.** 4,000 problems drawn from GSM8K-train and MATH-train, excluding the
S58 holdout and the S66 calibration ids, stratified by source and MATH level.
The problem set is frozen before generation and hashed.

**Generation.** The Stage-0 winner, same template, `max_new_tokens=4096`,
temperature 0.7 / top-p 0.8 for a general teacher; greedy for a math-specialist
teacher. One sample per problem, then top-up samples only for problems the
teacher fails, up to K=4, until 4,000 verified solutions exist or the budget is
spent. Keep only traces that are verified correct **and** EOS-terminated within
4,096; reject truncated traces rather than training on them. Deduplicate by
normalised final answer plus a sketch hash.

**Matched budget.** Target **10.94M tokens** of supervised completion, the S58
budget. If the traces are shorter, add problems from the same frozen pool until
the token budget matches; if longer, do not truncate — reject and resample.

**Receipt.** Teacher id, sampling parameters, per-problem attempts, verified
correctness, EOS status, token counts, the accepted set hash, and the exact
token total. The S58 control is the same 4,084-problem/10.94M-token corpus
already on disk.

**Cost.** 10.94M accepted tokens; at 150–250 tok/s and a ~1.5× rejection/top-up
factor, **18–30 GPU-hours** for generation, plus the top-up pass. Stage 0
replaces this estimate with a measured rate. The ranking table uses a
conservative **30 GPU-hours** for Stage 1.

#### H.3 Stage 2 — training arms

All arms are **full-backbone SFT** on the 498,113,344 trainable text-tower
elements, tied embedding frozen, exactly the S58 optimiser and schedule: Adam8bit,
cosine to 0.1, warmup 200, clip 1.0, lr 1e-5, 10,940,000 processed tokens, seed
fixed. The step count is whatever the new packing makes of 10,940,000 tokens
(S58's packing gave 4,376); the registration records the computed count before
launch. No LoRA: §C says full FT is the capability default, and the capacity
question is G8, not this experiment.

| arm | data | purpose |
|---|---|---|
| **A (new teacher)** | 4,000 verified traces from the Stage-0 winner, 10.94M tokens | the treatment |
| **B (mix, optional)** | 50/50 by token of A and short student-correct traces (reuse the S69 rollout corpus where the prompt ids match; otherwise generate from the champion at K=4, cap 2,048, keep correct) | the "Small Model Learnability Gap" hedge; also a control for trace length |
| **C (control)** | the existing S58 2B-teacher run, 4,084 problems / 10.94M tokens | matched-budget, matched-objective control; already exists |

**Memory.** S58's shape peaked at 12.728 GiB allocated / 13.086 GiB reserved GPU
and a 15,032,385,536-byte container cgroup peak against a 14g limit
(`docs/S58-STAGE2A.md` §3). This run must be given **`SELVEDGE_MEM=20g`** and a
GPU memory check before launch; the 14g court default is known to be too small
for this shape and must not be reused. Fail closed if the smoke peak plus 1 GiB
exceeds 15.5 GiB reserved.

**Cost.** 10.94M / 2,081 tok/s = 5,257 s ≈ **1.5 GPU-hours per arm**; arms A and
B ≈ **3 GPU-hours**. The 48-hour total below is for arm A plus the control. If
arm B needs a full 50/50 mix rather than the 650-problem S69 rollout reuse, add
about **9 GPU-hours** of student generation (3,000 traces × K=4 × ~1.8K tokens ≈
21.6M tokens at 668 tok/s), making the full three-arm registration ≈ **57
GPU-hours**.

#### H.4 Stage 3 — the delta-fraction and rank ladder

For each new arm, form `δ = θ_arm − θ_post` in float32 over the 498,113,344
trained elements, and apply `θ(α) = θ_post + α δ`, rounded back to bf16.

* **α grid:** {0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 1.0}. α=0 is the
  post-trained baseline; α=0.375 is the S67 champion's coefficient.
* **Rank grid at the best α:** per-matrix SVD of δ, keep the top-r singular
  directions, renormalise the truncated delta to the L2 norm of `αδ`, and apply.
  r ∈ {64, 256, full}. This separates "the delta is too big" from "the delta's
  harmful part is high-rank".
* **Scoring instrument:** the registered 1,000-item GSM8K-cal and MATH-cal
  splits and the 2,000-item MMLU-cal, with the exact S69 scorer
  (`scripts/s69_milestone_calibration.py`, greedy, non-thinking, 2,048 cap) so
  the rows are paired with the soup's 0.609 / 0.326 / 0.5215. The 4,096-cap test
  instrument is **not** read during selection.
* **Pairing:** per-problem paired bootstrap against the soup re-score, 4,000
  resamples, seed 53, as in S67/S69.

**Selection rule (fixed before scoring).** Highest MATH-cal among points with
MMLU-cal ≥ soup − 1.0 point; tie-break higher GSM8K-cal, then smaller α, then
smaller rank.

**Adoption gate.** MATH-cal ≥ **0.346** (soup + 0.02) and GSM8K-cal ≥ **0.599**
(soup − 1.0) and MMLU-cal ≥ **0.5115** (soup − 1.0). If met, score the selected
checkpoint **once** on GSM8K test, MATH-500, full MATH, full MMLU and MMLU-Pro
4,000.

**Win bar (test).** MATH-500 ≥ **50.0** and full MATH ≥ **50.0**, with paired
95% intervals above zero against the soup, and GSM8K ≥ 62.38 and MMLU ≥ 51.07
(the 1.0-point retention bounds). If the point estimates are below 50 but the
intervals include 50, report the result as **unresolved**, not a win.

**Cost.** Eleven ladder points × ~1.0 GPU-hour = **11 GPU-hours**, or ~4
GPU-hours if a 300-item screen is used first and only the top two points are
confirmed at 1,000 items. Total for G1: Stage 0 4–6 h + Stage 1 30 h + Stage 2
3 h + Stage 3 11 h ≈ **48 GPU-hours** (2 GPU-days), which is the number in the
ranking table. The screen's intervals are wider; the registration
must say so.

#### H.5 Abandonment condition

Abandon the strong-teacher route if **any** of the following holds:

1. No α/rank point reaches MATH-cal ≥ 0.346 with retention (the adoption gate).
2. The best new-teacher point is not better than the S58 2B-teacher arm's best
   point at matched budget by at least 0.01 MATH-cal.
3. The new delta is safe only at α ≤ 0.125 and its best MATH-cal is below the
   soup's 0.326. This means the student cannot absorb the teacher at our token
   budget, and more teacher quality will not help without more data.
4. Generation diagnostics at the selected point show cap fraction > 0.6 or EOS
   < 0.4 on MATH-cal (the S69 collapse signature at a smaller scale).

If condition 1 or 2 holds, the next registered experiment is **G2**
(self-consistency → weights), not another teacher/dataset variant. If condition
3 holds, the next experiment is **G4** (short/long mix) or a data-volume ramp,
not a new objective.

#### H.6 Why this and not the alternatives ranked above it

* **G3 (delta-SVD, ratio 0.159)** is the best information-per-GPU-hour buy, and
  it should be run as a side diagnostic if the operator wants one. It cannot
  reach 50 because it operates on the 2B-teacher delta, whose best measured
  effect is +2.5 full MATH. It is a safety experiment, not a capability
  experiment.
* **G4 (short/long mix, ratio 0.126)** is the best hedge on the "Small Model
  Learnability Gap" and is included as arm B. Running it alone would leave the
  teacher-quality hypothesis untested, and its ceiling is a 2B/student signal.
* **G2 (TTC→weights, ratio 0.116)** is the strongest fallback and the only other
  candidate with a plausible +5 ceiling. It ranks third by ratio but is
  subordinated here because its teacher is the student: the trace set is bounded by the student's own K-sample coverage, and
  the published TTC-to-weights effects (STaR, ReST^EM) are modest at this scale.
  If G1 shows the student can absorb a stronger teacher, G1 dominates; if it
  shows the student cannot, G2 is the right next move.
* **G7 (ratio 0.094) and G5 (ratio 0.080)** also rank above G1. G7 is a
  zero-training merge test on the three existing seeds: it can recover at most
  a point or two, because the mean is already the soup. G5 is a stability test
  on 214 pairs: its best realistic outcome is +1–2 MATH-cal, and its main value
  is closing the preference question, not reaching 50. Both are worth running
  as gates; neither changes the delta's content.
* **G6 and G8** are the cheap collapse diagnostic and the capacity diagnostic.
  They are the right second and third experiments if G1 is funded, because they
  explain *why* it worked or failed.

#### H.7 The three ways this recommendation could be wrong

1. **The teacher may not be the bottleneck.** If the 0.8B student's capacity,
   not the trace quality, is binding, G1 will produce a safe delta with a small
   gain, and the ceiling argument collapses. The mix arm B and the α ladder are
   designed to detect this (a safe delta with a flat curve).
2. **The 4,096 cap may discard the teacher's advantage.** A 4B thinking teacher
   may need 8K–16K tokens to reach its headline accuracy. Stage 0 is designed to
   detect this before the 30–45-hour generation run; if it fires, the
   recommendation is wrong for this protocol and the next experiment is a
   cap/format study.
3. **The calibration selection may not transfer to the test.** 1,000 items give
   a MATH-cal standard error of about 1.5 points; a +2-point calibration gain can
   be selection noise. The one-shot 5,000-problem full-MATH instrument and the
   paired interval are the protection, and the win bar is the program target,
   not a lower bar invented after the fact.

---

## I. What I could not verify

1. **MobileLLM-R1-950M's 74.0 MATH-500 decoding details.** The card says 0-shot
   pass@1 but does not state temperature, cap, or whether the answer was parsed
   with a verifier. The paper may; this session read the abstract and the card,
   not the full experimental section.
2. **rStar-Math's per-model MATH numbers and search budget.** The abstract gives
   7B 90.0 and 3.8B 86.4 and says MCTS; the number of rollouts per problem, the
   PPM's role, and the exact 3.8B/7B split were not read in the full paper.
3. **Qwen2.5-Math-1.5B-Instruct's evaluation cap.** Table 3 reports zero-shot
   pass@1; the report states 4,096 for *training*, not for evaluation. The 75.8
   is greedy but the cap is not verified.
4. **Qwen3-0.6B's 55.2/77.6 evaluation settings.** The report's benchmark
   section was not read in full; the tables do not state the generation cap in
   the HTML this session read.
5. **SimpleRL-Zoo's GPU count and wall-clock hours.** The hyperparameters and
   the MATH-500 deltas are verified from the paper; the hardware and total
   compute are not in the text this session read.
6. **arXiv:2503.16219's best-checkpoint MATH-500.** The text says 83 → 84 within
   50–100 steps and degradation after 200; a final table might report a
   different selected checkpoint. The +1 is the number the text supports.
7. **The 2026 on-policy-distillation papers** (CCE-OPD arXiv:2610.21195, SCOUT
   arXiv:2609.38360, SFD arXiv:2605.30833) are preprints; their effect sizes are
   not peer-reviewed and their student/teacher configurations are not identical
   to ours.
8. **TRL's `rpo_bc` name.** Current TRL and v0.9.6 document `rpo_alpha`, which
   points to arXiv:2404.19733 (Iterative Reasoning Preference Optimization) and
   adds the SFT/NLL term on the chosen. I did not find an `rpo_bc` loss type in
   either version; if the operator's `rpo_bc` refers to a different fork or an
   older alias, this file has not verified it.
9. **cDPO.** TRL's `label_smoothing` documentation names "the cDPO report and
   Robust DPO paper". The peer-reviewed robust-DPO result is arXiv:2403.00409;
   the cDPO note itself is not a paper I could verify, and no cDPO effect size is
   quoted here.
10. **Robust DPO's benchmark effect size.** The abstract gives a sub-optimality
    bound, not a MATH or GSM8K delta.
11. **Tokenizer compatibility** between Qwen3-4B and Qwen3.5-0.8B. It is
    plausible (same family) and is **not verified**; it matters for token-level
    OPD and does not matter for text-level SFT.
12. **No new measurement.** The GPU driver is down; no court was run; every
    throughput number is a quotation from S58/S67, and every cost in §G/H is
    arithmetic on those quotations plus the cited literature. The costs are
    **estimates**, not receipts; Stage 0 of the recommendation exists to replace
    the generation rate with a measured one before the expensive stage starts.

---

## J. One-paragraph bottom line

The binding constraint is not the preference objective, not the LoRA rank, and
not the merge recipe. It is the delta's content: 4,084 verified 2B-teacher
traces, interpolated to 37.5%, buy +2.5 full MATH and no more. Every published
sub-1B model at 50+ MATH paid for it with 200–1,500× more post-training data,
strong-teacher long CoT, or test-time search. The next GPU-days should buy a
better delta — the strongest teacher that fits 16 GB and 4,096 tokens, at the
same 10.94M-token budget — and should apply it through the one mechanism this
program has already validated: a measured fraction of the delta, selected on the
calibration instrument, with the test read once. If that delta is also unsafe at
every fraction, the answer is not a sixth objective; it is that 0.8B with 10M
tokens cannot reach 50 on this protocol, and the program should say so.
