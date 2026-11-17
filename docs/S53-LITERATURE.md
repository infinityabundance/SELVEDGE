# S53 focused literature — post-trained distillation, loops, and the vLLM recipe

Read 2026-11-17 (UTC). URLs were read through the arXiv API or the vLLM
documentation on that date. Each row names the exact claim the source makes and
whether it supports what this session's registration needs. Nothing here is a
SELVEDGE measurement.

## 1. Verdicts

| work | URL / date | claim as stated | verdict | bearing on S53 |
| --- | --- | --- | --- | --- |
| **CANON** — *Consensus as Privileged Context for Label-Free Self-Distillation*, Gkountouras, Jukić, Titov | `https://arxiv.org/abs/2607.13643` (v1 2026-07-15) | "For each unlabeled prompt, CANON samples multiple solutions, extracts the majority answer, and conditions a frozen snapshot of the model on a solution that reaches it; this consensus-anchored teacher then supervises the model on its own rollouts at every token." "Improves pass@1 by up to 12 points, outperforming label-free reinforcement learning by 6 points at a seventh of its compute and approaching a teacher conditioned on gold solutions"; trained on pooled unlabeled data it "matches training methods that use gold labels". | **VERIFIED** (abstract). **PARTIAL** for our exact setting: their teacher conditions on a consensus solution only; stage 3 conditions on a solution that is *both* the maj@16 consensus and train-label-verified. | Direct published precedent for the stage-3 teacher construction. The "up to 12 points" is a pass@1 gain; "6 points" is the margin over label-free RL; "a seventh" is compute. These are three different quantities and must not be merged. |
| **D-LoopOPD** — *Recurrent Self-Improvement: Dynamic Cross-Loop On-Policy Distillation for Looped Language Models*, Wang, Qian, Li, Yao, Wang | `https://arxiv.org/abs/2610.10623` (v2 2026-10-23) | LoopOPD "uses a frozen terminal loop policy as a compute-privileged teacher for an intermediate loop student on student generated rollouts, providing dense supervision without an external teacher or privileged information". D-LoopOPD "continually refreshes the terminal loop teacher as the shared model parameters are updated"; "LoopOPD improves mathematical reasoning, while D-LoopOPD yields further gains"; trained only on math, the models "also improve on general reasoning and code generation benchmarks". | **VERIFIED** (abstract). **PARTIAL** for the teacher construction: their teacher is the *terminal loop* of the same model, not a frozen R=1 snapshot with a consensus+verified privileged solution. | Published precedent for cross-loop on-policy distillation. Our stage-3 teacher is an information-privileged R=1 snapshot, which the paper explicitly says it avoids; that difference must be stated, not hidden. |
| **Small Model Learnability Gap** — *Small Models Struggle to Learn from Strong Reasoners*, Li et al. | `https://arxiv.org/abs/2502.12143` (v3 2025-11-13) | "small models (≤3B parameters) do not consistently benefit from long chain-of-thought (CoT) reasoning or distillation from larger models. Instead, they perform better when fine-tuned on shorter, simpler reasoning chains that better align with their intrinsic learning capacity"; Mix Distillation "significantly improves small model reasoning performance". | **VERIFIED** (abstract). **PARTIAL** for Qwen3.5-0.8B: the paper's smallest models are Qwen-family ≤3B but not the Qwen3.5-0.8B post-trained checkpoint. | Directly relevant risk: the 2B teacher's long solutions may exceed the 0.8B student's learnability. The stage-2 registration uses a 1,024-token generation cap and reports length; it does not assume long CoT helps. |
| **MiCoTA** — *MiCoTA: Bridging the Learnability Gap with Intermediate CoT and Teacher Assistants*, Ding et al. | `https://arxiv.org/abs/2507.01887` (v1 2025-07-02) | "SLMs often struggle to learn long-form CoT reasoning due to their limited capacity, a phenomenon we refer to as the 'SLMs Learnability Gap'." MiCoTA uses intermediate-sized teacher assistants and intermediate-length CoT; Qwen2.5-7B-Instruct and Qwen2.5-3B-Instruct improve "3.47 and 3.93 respectively on average" on AIME2024, AMC, Olympiad, MATH-500 and GSM8K. | **VERIFIED** (abstract). **PARTIAL** for us: the +3.47/+3.93 are Qwen2.5-7B/3B-Instruct averages across five tasks, not Qwen3.5-0.8B and not a single-task gain. | Supports the "short reasoning / intermediate teacher" mitigation if stage 2 loses on length. Do not quote +3.47 as an expected 0.8B gain. |
| **Retrofitted recurrence** — *Teaching Pretrained Language Models to Think Deeper with Retrofitted Recurrence*, McLeish et al. | `https://arxiv.org/abs/2511.07384` (v1 2025-11-10) | A "curriculum of recurrences" converts pretrained non-recurrent LMs into depth-recurrent ones, "preserves performance while reducing total computational cost"; on mathematics, "converting pretrained models to recurrent ones results in better performance at a given compute budget than simply post-training the original non-recurrent language model". | **VERIFIED** (abstract). **PARTIAL** for the claim that it is compute-matched at inference: the internal reading in `docs/PRIOR-ART-AND-NOVELTY.md` §19.1 records the GSM8K `46.2 -> 52.0` gain at **32× inference compute**, not matched inference compute; the paper's matched claim is about *training* FLOPs. | Stage 3 must report FLOPs per answer; it must not quote `46.2 -> 52.0` as a compute-matched result. |
| **Retrofitted recurrence at two budgets** — *Retrofitting Recurrent Depth into a Pretrained Language Model*, Shapiro | `https://arxiv.org/abs/2608.11233` (v1 2026-07-31) | Qwen2.5-0.5B-Instruct split into Prelude/weight-tied Recurrent Block/Coda with an "identity-preserving one-loop path and a re-entry bridge"; at loop 1 "non-inferior to its base"; installs at "6M trained parameters over frozen base weights and 180M full-block"; adapter "83.8% versus 84.0%" of the full block; "retained 53% versus 2.5% beyond depth 10, and answered 7.6 times faster". | **VERIFIED** (abstract). **PARTIAL** for Qwen3.5-0.8B post-trained: different model, 0.5B, ARC battery, not GSM8K/MATH. | The "identity-preserving one-loop path" is exactly the bit-identical R=1 requirement; the paper is the precedent for it. Its numeric gains are not transferable to our model or task. |
| **Distilling System 2 into System 1** — Yu, Xu, Weston, Kulikov | `https://arxiv.org/abs/2407.06023` (v3 2024-07-24) | "self-supervised methods to 'compile' (distill) higher quality outputs from System 2 techniques back into LLM generations without intermediate reasoning token sequences"; "several such techniques can be successfully distilled, resulting in improved results compared to the original System 1 performance, and with less inference cost than System 2". | **VERIFIED** (abstract). **PARTIAL** for us: the paper distils inference-time techniques, not a teacher model's consensus+verified solution; it reports no single universal point gain. | The "less inference cost than System 2" claim is the ancestor of the stage-3 FLOPs condition. It is a qualitative claim; no number may be imported. |
| **1-shot RLVR** — *Reinforcement Learning for Reasoning in Large Language Models with One Training Example*, Wang et al. | `https://arxiv.org/abs/2504.20571` (v3 2025-10-24) | One verifiable-reward example elevates Qwen2.5-Math-1.5B on MATH500 "from 36.0% to 73.6% (8.6% improvement beyond format correction)", and the six-benchmark average "from 17.6% to 35.7% (7.0% non-format gain)"; this "matches the performance obtained using the 1.2k DeepScaleR subset"; two examples "slightly exceed" it (MATH500 74.8%, average 36.6%). | **VERIFIED** (abstract), with a number-reading warning below. | A data-efficiency precedent, not a stage-2 recipe: our stage 2 is distillation, not RLVR. |
| **On-policy distillation (GKD)** — Agarwal et al. | `https://arxiv.org/abs/2306.13649` (ICLR 2024) | "Instead of solely relying on a fixed set of output sequences, GKD trains the student on its self-generated output sequences by leveraging feedback from the teacher on such sequences"; evaluated on "summarization, translation, and arithmetic reasoning". | **VERIFIED** (abstract; also read at S43/S45). **PARTIAL** for our setting: their students are encoder-decoder/small models, not a post-trained Qwen3.5-0.8B at a 20M-token budget. | Stage 2's on-policy reverse KL is the GKD family. The 2026 Qwen3.5-scale line (CoRD, SD-OPD, MPOPD) is recorded in `docs/S45-LITERATURE.md` and is not re-read here. |
| **vLLM Qwen3.5 recipe** | `https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html` (page updated 2026-06-18) | Serves `Qwen/Qwen3.5-397B-A17B` and `Qwen/Qwen3.6-35B-A3B` with `--language-model-only`, `--reasoning-parser qwen3`, `--default-chat-template-kwargs '{"enable_thinking": false}'`, prefix caching and optional MTP-1. | **VERIFIED** for the 397B-A17B/35B-A3B recipe. **UNSUPPORTED** as a recipe for the 0.8B/2B chat checkpoints: the page does not name them. | Stage 1 may use the flags, but it may not claim the page is a recipe for the small checkpoints. The flags must be confirmed by a stage-1 smoke. |
| **vLLM supported models** | `https://docs.vllm.ai/en/latest/models/supported_models.html` (read 2026-11-17) | Lists `Qwen3_5ForConditionalGeneration` for Qwen3.5, with example `Qwen/Qwen3.5-9B-Instruct`; notes "hybrid-only models such as Llama-4, Step3, Mistral-3 and Qwen-3.5" need `--language-model-only` for text-only. | **VERIFIED** that the architecture family is supported. **PARTIAL** for the 0.8B/2B checkpoint: the example is 9B-Instruct, and the local eval image does not contain vLLM. | Stage 1 pins vLLM 0.34.0 and must smoke the exact checkpoints; a family-level support row is not a checkpoint-level guarantee. |

## 2. Claimed numbers that do not match the source

1. **Retrofitted recurrence `46.2 -> 52.0` is not compute-matched.** The gain is
   at test recurrence 32; at recurrence 1 the same table reports `25.6`. The
   internal reading in `docs/PRIOR-ART-AND-NOVELTY.md` §19.1 already records
   this. Stage 3 must report FLOPs per answer and may not call this a
   compute-matched gain.

2. **1-shot RLVR: `8.6` is not the MATH500 gain.** MATH500 moves `36.0 -> 73.6`
   (+37.6 points); `8.6` is the improvement *beyond format correction*. The
   six-benchmark average moves `17.6 -> 35.7` (+18.1), of which `7.0` is the
   non-format gain. A sentence that says "1-shot RLVR improves MATH500 by 8.6
   points" is wrong by a factor of about four.

3. **CANON: `12`, `6` and `1/7` are three different quantities.** `12` is the
   maximum pass@1 gain; `6` is the margin over label-free RL; `1/7` is the
   compute fraction of that comparison. Stage 3's registered threshold is
   `<= 0.30` of stage-2 maj@16 FLOPs, which is *not* CANON's `1/7`; the two
   numbers must not be interchanged.

4. **vLLM has no published 0.8B/2B Qwen3.5 recipe on the recipe page.** The
   page is for the 397B-A17B and 35B-A3B models. The supported-models page is
   family-level (`Qwen3_5ForConditionalGeneration`, example 9B-Instruct). A
   claim that the recipe page covers the small checkpoints is **UNSUPPORTED**.

5. **MiCoTA's `+3.47 / +3.93` are not 0.8B numbers.** They are Qwen2.5-7B/3B-
   Instruct averages over AIME2024, AMC, Olympiad, MATH-500 and GSM8K. They are
   an existence proof for the length/teacher-assistant mitigation, not an
   expected gain for our target.

6. **D-LoopOPD does not use a consensus+gold privileged teacher.** Its teacher is
   the terminal loop of the same model, explicitly "without an external teacher
   or privileged information". Our stage-3 teacher is a frozen R=1 snapshot
   conditioned on a consensus-and-verified solution. That is an extension of the
   published mechanism, and the registration labels it as such.

7. **Distilling System 2 into System 1 reports no universal point gain.** Its
   claim is that several System 2 techniques can be compiled into System 1 with
   "less inference cost than System 2". Quoting a single point gain for it would
   not be supported.

## 3. What this page does not establish

- It does not re-read the 2026 Qwen3.5-scale OPD papers already recorded in
  `docs/S45-LITERATURE.md`; those verdicts stand and are referenced, not
  re-derived.
- It does not verify the vLLM 0.34.0 wheel against a downloaded hash; the
  PyPI record was read on 2026-11-17 and the wheel hash is recorded in the
  registration. Stage 1 must record the installed version.
- It does not establish that any of these mechanisms will transfer to the
  post-trained Qwen3.5-0.8B at a 20M-token budget. That is the experiment.
