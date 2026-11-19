# S60 — literature: LoRA, reward gating, difficulty weighting, relay and token importance

Every entry names the URL, the date it was read, the exact claim being checked,
and one of **VERIFIED** / **PARTIAL** / **UNSUPPORTED**. `VERIFIED` means the
claim is visible in the named primary source; it does not mean the paper is
right, or that it transfers to this student and this corpus. `PARTIAL` means the
source supports part of the claim, or supports it in a different setting.
`UNSUPPORTED` means no source for the claim was found in this session; it is not
a claim that no source exists.

Sources were read on **2026-11-19** through the arXiv API (`export.arxiv.org`)
and the named pages. The retrieval was metadata/abstract level unless a page is
quoted below.

## 1. LoRA matches full fine-tuning in the low-regret regime

| source | claim checked | status |
| --- | --- | --- |
| [LoRA Without Regret](https://thinkingmachines.ai/blog/lora/), Thinking Machines Lab, 2025-09-29 | Applied to **all** weight matrices (especially MLP/MoE), LoRA matches full fine-tuning for supervised fine-tuning in the small-to-medium post-training regime; attention-only LoRA underperforms even at higher rank. | **VERIFIED** |
| same | The optimal LoRA learning rate is approximately **10x** the full-fine-tuning rate, for both SFT and RL; the ratio is empirical and the article says it lacks a full theory. | **VERIFIED** |
| same | For policy-gradient RL, LoRA matches full fine-tuning even at very low rank (as low as rank 1 on MATH/GSM8K). | **VERIFIED** |
| same | LoRA is less tolerant of large batch sizes than full fine-tuning, independent of rank. | **VERIFIED** |
| Inference from the source | "Rank 64 on every linear layer is sufficient for a 6000-row SFT corpus" | **PARTIAL** — the article gives a capacity condition (trainable parameters exceed the information in the data), not a rank for this corpus. The rank is a registered choice, not a source-supported constant. |
| Inference from the source | "A 10x learning rate is the right starting point for this reverse-KL objective" | **PARTIAL** — the 10x ratio is measured for SFT and policy-gradient RL; our reward-gated reverse-KL OPD is closer to the RL setting than to SFT, but it is not one of the paper's experiments. The fixed-batch KL gate, not the paper, decides the rate. |

**What this changes for S60.** The revised stage-2 registration applies rank-64
LoRA to every linear layer and starts the LR screen at 1e-4 (10x the full-FT
1e-5). Both are supported as a *starting point*; neither is a result. The
all-linear choice is the part the source is emphatic about, and it is why the
registration does not use attention-only LoRA.

## 2. Reward-gated on-policy distillation

| source | claim checked | status |
| --- | --- | --- |
| [RG-OPD: Reward-Gated On-Policy Distillation](https://arxiv.org/abs/2607.04037), submitted 2026-07-04 | A teacher can assign high likelihood to plausible-but-wrong solutions and low likelihood to correct but differently-reasoned student solutions; using verifier feedback to decide when teacher logits are trusted preserves dense token supervision while filtering misleading signals, and beats vanilla reverse-KL. | **VERIFIED** |
| This repository's rule | A verified final answer (train label) gates the reverse-KL supervision | **PARTIAL** — RG-OPD gates *teacher logits* by verifier feedback, which is the same mechanism at the rollout level. Our exact form (a binary verified-answer gate on reverse KL, with no teacher-likelihood term) is a registered instantiation, not a copied result. |

**What this changes for S60.** The reward gate is registered as the stage's
central mechanism and its own matched control (gate off, every sampled rollout)
is required, because RG-OPD's evidence is for its own gate.

## 3. Difficulty weighting: p(1-p)

| source | claim checked | status |
| --- | --- | --- |
| [PACED: Distillation and On-Policy Self-Distillation at the Frontier of Student Competence](https://arxiv.org/abs/2603.11178), submitted 2026-04-09 | Weighting each problem by `w(p) = p(1-p)`, where `p` is the student's empirical pass rate, concentrates training on the zone of proximal development; it needs only student rollouts and no hyperparameters, and improves over unweighted distillation on MATH-500/AIME. | **VERIFIED** |
| [Restoring the Sweet Spot: Pass-Rate Weighted Self-Distillation](https://arxiv.org/abs/2605.27765), submitted 2026-05-26 | For SDPO, the analysis yields a `sqrt(p(1-p))` per-question weight (SC-SDPO), not the un-squared weight. | **VERIFIED** |
| Inference from the sources | "p(1-p) is the right weight for our reverse-KL OPD" | **PARTIAL** — PACED measures the un-squared weight in its distillation setting; SC-SDPO derives the square root for a different objective. The two must not be conflated, and our run uses the un-squared PACED weight as registered. |

**What this changes for S60.** The registration estimates `p` from K=8 rollouts
per problem and uses `p(1-p)`; `p=0` and `p=1` carry zero weight and are counted.
The K=8 estimate is coarse near 0 and 1, which is the regime where the weight is
small anyway; the receipt records the histogram so the approximation is visible.

## 4. Relay: teacher continuation from the student's prefix

| source | claim checked | status |
| --- | --- | --- |
| [Pass the Baton: Trajectory-Relayed On-Policy Distillation (Relay-OPD)](https://arxiv.org/abs/2607.26057), submitted 2026-07-28 | Failed student prefixes create a teacher-student continuation asymmetry; a label-free handoff trigger lets the teacher briefly take over, after which the student resumes and is optimised on the relayed trajectory. Reported +5.73% over OPD on eight math benchmarks for a 1.7B student. | **VERIFIED** |
| [Learning from Teacher Continuations at Student States (OLIVE)](https://arxiv.org/abs/2609.36246), submitted 2026-10-12 | At each iteration the student generates a prefix, the teacher continues it autoregressively, and the student is updated by **cross-entropy on the teacher-generated tokens**. | **VERIFIED** |
| Our trigger | "Cut at **loop onset** and let the teacher continue" | **UNSUPPORTED by these sources** — Relay-OPD's trigger is a failed-prefix asymmetry, not a loop detector; OLIVE regenerates prefixes without a loop trigger. The loop-onset trigger is this repository's registered extension. |
| Our loss on relay tokens | "Train the relay tokens with reverse KL" | **UNSUPPORTED by these sources** — OLIVE trains by cross-entropy on teacher tokens; Relay-OPD optimises the resulting trajectory under its own OPD loss. Reverse KL on relay tokens is a registered choice, not a published result. |

**What this changes for S60.** The relay is registered with its own control
(relay off) and is described as a hypothesis. The report must not cite Relay-OPD
or OLIVE as evidence that *loop-onset reverse-KL relay* works.

## 5. Termination and stopping in distillation

| source | claim checked | status |
| --- | --- | --- |
| [Solving Without Stopping: On-Policy Distillation at Small Scale](https://arxiv.org/abs/2609.37326), submitted 2026-09-29 | Distillation transfers *solving* but in **thinking mode** not *stopping*: students stop ending their reasoning early, and the teacher teaches no new stops. The paper also reports that **non-thinking** students keep stopping. | **VERIFIED** |
| Inference | "Our non-thinking stage will lose its stopping ability" | **PARTIAL / contrary** — the source says non-thinking students keep stopping, so its stopping warning applies to thinking mode, not to our protocol. Our termination/truncation/loop measurement is still required, but the source does not predict a non-thinking stopping regression. |

**What this changes for S60.** The 0.5M/1M/2M/5M cadence reports termination,
truncation and loop rates at every point. The S56/S59 loop cut and the DAPO
truncation tension remain as registered; this paper is not evidence for a
non-thinking stopping failure.

## 6. Token importance

| source | claim checked | status |
| --- | --- | --- |
| [TIP: Token Importance in On-Policy Distillation](https://arxiv.org/abs/2604.14084), submitted 2026-05-21 | Informative tokens come from two regions: high student entropy, and low student entropy with high teacher-student divergence (overconfident and wrong). Retaining 50% of tokens by entropy matches all-token training and cuts peak memory by up to 47%. | **VERIFIED** |
| Inference | "TIP's selection rule should be applied in this stage" | **UNSUPPORTED as a claim about our run** — the revised stage does not add token selection; TIP is recorded as the reason a token-selection ablation exists, not as evidence that selection helps here. |

**What this changes for S60.** None to the registered recipe. TIP is context for
a later ablation; it is not used to change the loss, the mask or the budget.

## 7. Failure profiling and reachable trajectories

| source | claim checked | status |
| --- | --- | --- |
| [FPRT: Failure Profiling and Reachable Trajectory Selection for Reasoning Distillation](https://arxiv.org/abs/2611.09042), submitted 2026-11-06 (NeurIPS 2026) | Repeated sampling profiles each problem's failure behaviour (error rate and error consistency); the diagnosed mode guides teacher candidate generation; among correct candidates, the one whose **first corrective steps** are most reproducible by the student is selected, because the first steps after divergence are the bottleneck. | **VERIFIED** |
| Inference | "FPRT supports loop-onset relay" | **PARTIAL** — FPRT supports *profiling failures by repeated sampling* and *selecting reachable continuations*, which is the motivation for the pass-rate weight and the relay admission rule. It does not contain a loop-onset detector or a reverse-KL relay loss. |

**What this changes for S60.** The rollout pass keeps the per-problem pass rate
and the verification verdicts, so the failure profile FPRT describes is
available in the receipt without a new instrument.

## 8. Effect on the S60 design (summary)

| S60 decision | literature position |
| --- | --- |
| rank-64 LoRA on every linear layer, merged | supported as the all-layer low-regret regime (LoRA Without Regret); the rank is a registered choice |
| LR screen starts at 10x the full-FT rate | supported as the measured LoRA/FullFT ratio; the fixed-batch KL gate, not the source, selects the rate |
| verified-answer gate on reverse KL | supported in mechanism by RG-OPD; our binary form is a registered instantiation |
| per-problem `p(1-p)` weight from 8 rollouts | supported by PACED; do not substitute SC-SDPO's `sqrt(p(1-p))` |
| loop-onset teacher relay with reverse KL | **not supported by Relay-OPD or OLIVE** as registered; it is a hypothesis with its own control |
| termination/truncation/loop cadence | required by the S56 registration; Solving Without Stopping does **not** predict a non-thinking stopping loss |
| no token selection in this stage | TIP supports a later ablation, not this run |
