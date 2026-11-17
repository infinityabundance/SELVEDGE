# S45 focused literature — on-policy distillation, rejection-sampling fine-tuning, and small-scale math SFT

Status: **focused verification, 2026-11-17 (S45).** Each entry names the URL,
the fetch date, the claim the source itself states, and a verdict for the
question this session had to decide. `VERIFIED` means the claim is in the
fetched record; `PARTIAL` means the source supports part of the claim or a
different setting; `UNSUPPORTED` means no located source states the claim. An
abstract does not settle a paper, and the verdicts are scoped to what was read.
Where S43/S44 already read a source it is named rather than re-read.

The session had four questions:

1. is on-policy distillation / GKD established against supervised KD, and at
   which student scale;
2. is rejection-sampling fine-tuning (STaR/RFT) established for math;
3. what is known about verifier-filtered math SFT at small scale;
4. does any source state that a **20M-token** distillation budget moves GSM8K.

One local measurement frames the reading. The S45 forward analysis
(`results/stage2/s45-forward-analysis-20261117T082511Z.json`) found that on the
correct solution path the memory residual adds a small positive CE at 80.0% of
positions, dominated at the class level by letter/space tokens (letter +0.557,
space +0.060, digit +0.003, operator -0.010 nats per class mean on the five
probed items): the residual moves prose, not arithmetic. The 2026 on-policy
line below independently reports that style positions carry much more
teacher-student signal than math positions, which is the same failure shape at
a different mechanism.

## 1. On-policy distillation and GKD

| source | fetched | claim as stated | verdict |
| --- | --- | --- | --- |
| Agarwal et al., *On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes* (GKD), `https://arxiv.org/abs/2306.13649` (ICLR 2024) | 2026-11-17 | "Instead of solely relying on a fixed set of output sequences, GKD trains the student on its self-generated output sequences by leveraging feedback from the teacher on such sequences"; demonstrated on "summarization, translation, and arithmetic reasoning tasks" | **VERIFIED** that student-generated sequences plus teacher feedback is a published improvement over supervised KD on arithmetic reasoning. **PARTIAL** for our setting: the paper's students are encoder-decoder/small models, not a frozen Qwen3.5-0.8B-Base at a 20M-token budget. |
| Chen et al., *CoRD: Coupled Rollouts Fill the Blind Spot of On-Policy Distillation*, `https://arxiv.org/abs/2611.07959` (2026-11-05) | 2026-11-17 | "Across six base students from two model families (Qwen3 and Qwen3.5, 0.6B to 4B), CoRD raises average pass@1 over on-policy distillation by 0.9 to 2.0 points and pass@8 by 1.2 to 3.4 points on five public math benchmarks at equal steps and data"; standard OPD "seldom supervise[s]" teacher-reached states the student does not reach | **VERIFIED** that OPD is a working math post-training baseline at **0.6B–4B Qwen3.5** scale, and that its blind spot is states the student never samples. **PARTIAL** for us: they change the rollout distribution, we first change the data (verified solutions) and keep the rollout policy fixed. |
| Fang et al., *Reinforcing What the Student Has Learned: Teacher-Guided On-Policy Distillation* (SD-OPD), `https://arxiv.org/abs/2611.07722` (2026-11-05) | 2026-11-17 | "on math reasoning responses, style-related positions exhibit 3.5 times the mean absolute teacher–student log-probability difference of mathematical positions"; SD-OPD "improves six-task mathematics average pass@1 (avg@16) over OPD by 7.76 to 10.37 percentage points" | **VERIFIED** that the teacher's token-level signal is concentrated in style positions and that using the student's own base as the reference improves OPD. **PARTIAL** for us: Qwen3 1.7B–8B students, and their fix is a reweighting of the OPD update; ours is a data filter plus the registered preservation term. |
| Chen et al., *Proposing Better Rollouts for On-Policy Distillation* (MPOPD), `https://arxiv.org/abs/2610.36477` (2026-10-28) | 2026-11-17 | "the rollout policy in OPD is usually treated as fixed… this choice can create a compounding teacher–student mismatch"; on math reasoning, MPOPD "improves average accuracy by 6.91 over OPD on Qwen3-4B-Base while reducing average generation length by 41%" | **VERIFIED** that rollout-policy design is a first-order lever in OPD. **PARTIAL** for us: we register a single student-policy sampling pass, not a coupled rollout policy; this is a candidate next route if the single-pass KD arm loses. |
| Lin et al., *Do We Really Need KL Divergence for On-Policy Distillation of Large Language Models?*, `https://arxiv.org/abs/2609.33791` (2026-09-27) | 2026-11-17 | "simply assigning a reward of (+1) to tokens where the teacher probability is higher than the student probability and (-1) where it is lower… reproduces almost the same training mode as OPD with reverse KL"; "only the direction of a small subset of tokens with large teacher–student disagreement is critical" | **VERIFIED** that OPD's useful signal is concentrated in a small disagreement subset. **PARTIAL** for us: their objective is a signed token reward, ours is a bounded signed log-probability advantage inside the correction objective. |
| S43/S44 read (not re-read): MiniLLM `https://arxiv.org/abs/2306.08543`, DistiLLM `https://arxiv.org/abs/2402.03898`, AOPD `https://arxiv.org/abs/2610.31563`, asymmetric OPD `https://arxiv.org/abs/2605.06387` | 2026-11-17 (S43/S44) | reverse-KL / on-policy distillation, advantage-weighted supervision, and preservation in the non-positive region | **VERIFIED** in `docs/S43-LITERATURE.md` §3 and `docs/S44-LITERATURE.md` §1–§2; those verdicts stand. |

**Decision.** OPD is established and, in the 2026 line, already at Qwen3.5
0.6B scale. The local question is not whether OPD exists but whether a single
student-policy sampling pass followed by verified-only optimisation moves our
0.8B base on GSM8K at 20M tokens. SD-OPD's style-versus-math amplitude result is
the strongest external corroboration of the diagnosis: the residual's damage is
in prose tokens, and RFT's exact-solution CE is the direct counter-pressure.

## 2. Rejection-sampling fine-tuning and STaR

| source | fetched | claim as stated | verdict |
| --- | --- | --- | --- |
| Zelikman et al., *STaR: Bootstrapping Reasoning With Reasoning*, `https://arxiv.org/abs/2203.14465` (2022) | 2026-11-17 | "generate rationales to answer many questions… if the generated answers are wrong, try again to generate a rationale given the correct answer; fine-tune on all the rationales that ultimately yielded correct answers; repeat" | **VERIFIED** that self-generated, answer-verified rationales can be fine-tuned back into the model. **PARTIAL** for us: STaR's loop uses the model's own rationales; our SFT arm uses the 2B teacher's verified solutions (closer to RFT), and the KD arm uses student rollouts. |
| Yuan et al., *Scaling Relationship on Learning Mathematical Reasoning with Large Language Models* (RFT), `https://arxiv.org/abs/2308.01825` (2023-08) | 2026-11-17 | "RFT uses supervised models to generate and collect correct reasoning paths as augmented fine-tuning datasets"; "with augmented samples containing more distinct reasoning paths, RFT improves mathematical reasoning performance more"; "RFT brings more improvement for less performant LLMs"; LLaMA-7B reaches "49.3% on GSM8K" versus "SFT accuracy of 35.9%" | **VERIFIED** that verified-solution rejection sampling is a real GSM8K lever, and that the gain is larger for weaker models — the direction that matters for a 0.8B base. **PARTIAL** for our budget and scale: 7B/13B models, many generated samples, no 20M-token cap. |
| Ye et al., *LIMO: Less is More for Reasoning*, `https://arxiv.org/abs/2502.03387` (COLM 2025) | 2026-11-17 | "sophisticated mathematical reasoning can emerge with only a few examples"; LIMO "achieves 63.3% accuracy on AIME24 and 95.6% on MATH500… while using only 1% of the training data required by prior approaches" | **VERIFIED** that a small, curated, high-quality solution set can elicit math reasoning. **PARTIAL** for us: the base models are large (the method is a template hypothesis for models whose pre-training already encodes the domain), and the paper does not test a 0.8B student or a 20M-token budget. |
| Muennighoff et al., *s1: Simple test-time scaling*, `https://arxiv.org/abs/2501.19393` (2025) | 2026-11-17 | "we curate a small dataset s1K of 1,000 questions paired with reasoning traces"; SFT on s1K plus budget forcing makes s1-32B exceed o1-preview on competition math | **VERIFIED** that 1,000 curated traces can move a 32B model. **PARTIAL** for us: 32B and test-time budget forcing are both outside the 0.8B / 20M-token setting. |

**Decision.** RFT is the best-matched published lever for a weak student: the
gain grows as the model gets weaker. The risk the literature also names is
diversity and hard-problem coverage (§3). The registered RFT arm keeps the
teacher sampling temperature high (0.8), retains all distinct verified
solutions, and evaluates on the anchored per-item basis rather than pass@k
alone.

## 3. Verifier-filtered math SFT at small scale

| source | fetched | claim as stated | verdict |
| --- | --- | --- | --- |
| Liu et al., *TinyGSM: achieving >80% on GSM8k with small language models*, `https://arxiv.org/abs/2312.09241` (2023-12) | 2026-11-17 | a 12.3M-problem synthetic dataset plus "a duo of a 1.3B generation model and a 1.3B verifier model can achieve 81.5% accuracy" on GSM8K | **VERIFIED** that small models plus a large verified synthetic corpus and a verifier can break 80% GSM8K. **PARTIAL** for us: 12.3M problems is a far larger data budget than 20M tokens, and the verifier is used at inference, not only as a training filter. |
| Yang et al., *Qwen2.5-Math Technical Report*, `https://arxiv.org/abs/2409.12122` (2024-09) | 2026-11-17 | a reward model built from "massive sampling" is "applied to the iterative evolution of data in supervised fine-tuning"; the series includes Qwen2.5-Math-Instruct-**1.5B**/7B/72B | **VERIFIED** that reward/verifier-guided iterative math SFT is used at 1.5B scale. **PARTIAL** for us: the data and compute budget is not 20M tokens, and the reward model is learned, not an exact answer checker. |
| Wang et al., *Math-Shepherd: Verify and Reinforce LLMs Step-by-step without Human Annotations*, `https://arxiv.org/abs/2312.08935` (2023-12) | 2026-11-17 | automatic process supervision; step-by-step PPO improves Mistral-7B "77.9%→84.1% on GSM8K and 28.6%→33.0% on MATH"; verification lifts to "89.1% and 43.5%" | **VERIFIED** that a learned step-level verifier improves math post-training. **PARTIAL** for us: our verifier is an exact answer extractor, not a process reward model; we do not have a step-level signal. |
| Dorner et al., *ROC-n-reroll: How verifier imperfection affects test-time scaling*, `https://arxiv.org/abs/2507.12399` (2026-08 revision) | 2026-11-17 | "the instance-level accuracy of these methods is precisely characterized by the geometry of the verifier's ROC curve"; "it is generally impossible to predict the high-compute performance of either method based on observations in the low-compute regime" | **VERIFIED** that verifier quality and compute regime change the conclusion. **PARTIAL** for us: this is test-time selection, not training-data filtering, but it is the right caution against reading a low-acceptance-rate sample as a high-budget prediction. |
| Rath et al., *What pass@k Cannot Measure: Evaluating Diversity and Capability Retention after Post-Training*, `https://arxiv.org/abs/2610.07405` (2026-10) | 2026-11-17 | training Qwen2.5-1.5B-Instruct with RFT (own shortest verifier-passed rollout) moves diversity measures in the opposite direction from GRPO; "no trained arm significantly improves hard-problem coverage: RFT is significantly worse"; pass@8/32 show "no consistent winner on GSM8K" | **VERIFIED** that RFT can lose diversity and hard-problem coverage at 1.5B. **PARTIAL** for us: their RFT uses the model's own shortest correct rollout, our SFT arm uses teacher solutions and our KD arm keeps all distinct verified rollouts; our anchored eval is per-item, not pass@k. This is the strongest reason the registration keeps a hard held-out and knowledge condition. |
| Korman, *Inverse Reasoning Models for Efficient Post-Training and Verification from Answer-Only Data*, `https://arxiv.org/abs/2610.18878` (2026-10) | 2026-11-17 | with Qwen3-0.6B "rejection sampling by construction has no signal on problems the model never solves on its own"; a single offline distillation round from an inverse model "roughly doubles pass@1 on hard prompts" after GRPO saturation, "with smaller gains on GSM8K" | **VERIFIED** for Qwen3-0.6B that rejection sampling is limited by the solver's own coverage. **PARTIAL** for us: our teacher-policy SFT arm is not limited by the student's coverage (the 2B teacher can solve problems the 0.8B cannot); the KD arm is. This is exactly why the registration separates the two policies. |

**Decision.** Verifier-filtered math SFT is established at 1.3B–7B with much
larger data, and the Qwen3-0.6B result warns that student-policy rejection
sampling has no signal outside the student's coverage. The registered route
therefore uses **two policies**: teacher solutions for coverage (SFT) and
student solutions for on-policy states (KD), then a 1:1 mixture. The exact
answer verifier is weaker than a process reward model but is the only verifier
this program can audit end to end.

## 4. Is a 20M-token distillation budget known to move GSM8K?

| source / search | fetched | claim as stated | verdict |
| --- | --- | --- | --- |
| arXiv API query `all:"20M tokens" AND all:distillation`, `http://export.arxiv.org/api/query?search_query=all:%2220M%20tokens%22%20AND%20all:distillation` | 2026-11-17 | the query returned 0 records | **UNSUPPORTED** that any indexed paper states a 20M-token distillation budget moves GSM8K. |
| RFT (`2308.01825`), STaR (`2203.14465`), GKD (`2306.13649`), LIMO (`2502.03387`), s1 (`2501.19393`), TinyGSM (`2312.09241`) | 2026-11-17 | all report math gains, but at 1K–12.3M examples, 1.3B–32B models, or 800k-sample distillation sets; none states a 20M-token budget | **PARTIAL** for the direction (verified data can move math) and **UNSUPPORTED** for the exact budget. The 20M local budget is a registered local experiment, not a reproduction. |
| S43/S44 read (not re-read): math-mixture and small-budget literature (`docs/S43-LITERATURE.md` §1–§2, `docs/S44-LITERATURE.md` §4) | 2026-11-17 | math-heavy mixtures and proxy-scale mixture prediction, at billions of tokens or from-scratch pretraining | **VERIFIED** in those files; **PARTIAL** for 57% math in 20M online-distillation tokens. |

**Decision.** No external source sizes our budget. The local arithmetic is in
`results/stage2/s45-budget-plan-20261117T082853Z.json`: at the bar's measured
1,836.0 tok/s, 20M / 40M / 60M / 100M are 3.026 / 6.052 / 9.077 / 15.129 hours,
with checkpoints at 20/40/60/100M so one 100M run can draw the curve. The
sampling pass is bounded at 61.3M generated tokens per full teacher pass; the
acceptance rate is unmeasured until the job runs.

## What this file does not establish

- That any cited gain transfers to a frozen 0.8B base or to a 20M-token budget.
  Every source above either trains a larger model, uses much more data, changes
  the rollout policy, or all three.
- That the exact answer verifier is as good as a process reward model. It is
  strictly weaker: it cannot reward a correct step inside a wrong solution.
- That RFT will not reduce diversity or hard-problem coverage in our setting.
  `2610.07405` is the published warning, and our registration keeps a per-item
  anchored evaluation plus a held-out condition rather than pass@k alone.
- That teacher-policy SFT cannot hurt the knowledge family. The 20M full arm
  trains every text-tower identity except the tied embedding; the registered
  win bar requires the knowledge family not to fall below base beyond its
  combined standard error.
- That a 20M-token run is sufficient. No source states that, and the local
  budget plan is exactly the experiment that would answer it.
