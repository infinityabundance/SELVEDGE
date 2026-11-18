# S56 — literature: termination, loops, truncation, and the vLLM/Qwen3.5 question

Every entry names the URL, the date it was read, the exact claim being checked,
and one of **VERIFIED** / **PARTIAL** / **UNSUPPORTED**. `VERIFIED` means the
claim is visible in the named primary source (abstract, registry, or metadata),
not that the paper is right. `PARTIAL` means the source supports part of the
claim or supports it in a different setting. `UNSUPPORTED` means this session
found no source for the claim; it is not a claim that no source exists.

Sources were fetched on 2026-11-18 from inside
`scripts/court --net python`, so the retrieval is CPU-only and offline from the
repository's point of view. The arXiv API was queried by title/abstract; the
abstracts below were read from `export.arxiv.org` or `arxiv.org/abs`.

## 1. Termination and looping in reasoning models

| source | claim checked | status |
| --- | --- | --- |
| [Circular Reasoning: Understanding Self-Reinforcing Loops in Large Reasoning Models](https://arxiv.org/abs/2601.05693), submitted 2026-01-09 | Reasoning models have a distinct failure mode, *Circular Reasoning*: a self-reinforcing trap where generated text becomes a premise for its own recurrence. `LoopBench` covers numerical and statement loops. The paper reports that **semantic repetition precedes textual repetition** and uses CUSUM to predict loop onset. | **VERIFIED** |
| [Stop Overthinking: A Survey on Efficient Reasoning for LLMs](https://arxiv.org/abs/2503.16419), submitted 2025-03-20 | Long chains of thought contain redundant/verbose output ("overthinking"), and efficient-reasoning work spans model-based, output-based and training/decoding interventions. | **VERIFIED** |
| [Solving LLM Repetition Problem in Production](https://arxiv.org/abs/2512.04419), submitted 2025-12-04 | In production batch code interpretation, greedy decoding gets stuck in repetitive loops via self-reinforcement; beam search with `early_stopping=True` and `presence_penalty` are practical mitigations. | **VERIFIED** |

**What this changes for S56.** The detector we ship is *textual*: a loop must
already be visible in the tail. The Circular Reasoning result says semantic
repetition starts earlier. So the detector is a **compute-saving cut**, not an
early correctness signal: a loop it misses costs tokens, and a loop it catches
is still measured for false cuts. We do not claim it detects the onset of
circular reasoning.

## 2. Repetition penalties versus training-time termination

| source | claim checked | status |
| --- | --- | --- |
| [LZ Penalty: An information-theoretic repetition penalty for autoregressive language models](https://arxiv.org/abs/2504.20131), submitted 2025-04-28 (v4 2026-08-18) | The industry-standard frequency penalty and repetition penalty are **ineffective**, with degenerate repetition rates up to 4%; an LZ77-codelength penalty lets state-of-the-art reasoning models run greedy without degenerate repetitions and without capability loss. | **VERIFIED** |
| [Neural Text Generation with Unlikelihood Training](https://arxiv.org/abs/1908.04319), submitted 2019-08-12 | The likelihood objective assigns too much probability to repeated sequences; unlikelihood training lowers it at training time and reduces repetition. | **VERIFIED** |
| [Repetition In Repetition Out](https://arxiv.org/abs/2310.10226), submitted 2023-10-16 | Repetition in the training data correlates with degeneration; dropping attention to repetitive training words reduces it. | **VERIFIED** |
| [Learning to Break the Loop](https://arxiv.org/abs/2206.02369), submitted 2022-06-06 | Greedy/maximisation decoding gets stuck in sentence-level loops; repeated sentences are self-reinforcing in the context. | **VERIFIED** |
| [Penalty Decoding](https://arxiv.org/abs/2310.14971), submitted 2023-10-23 | A repetition penalty suppresses self-reinforcement but its optimal value is hard to choose; forgetting distant tokens and a length penalty are proposed to compensate. | **VERIFIED** |

**What this changes for S56.** The registered decision not to add a repetition
penalty to the eval (and not to tune decoding to raise a score) is supported by
LZ Penalty: the standard penalties are weak, and changing decoding would change
the quantity being measured. Training-time objectives (unlikelihood, data-side
deduplication, attention dropout) are the supported alternative, and our S55
corpus already deduplicates exact generated token sequences. A repetition
penalty is not a substitute for either.

## 3. Compression-based loop detection

| source | claim checked | status |
| --- | --- | --- |
| LZ Penalty (above) | Compression (LZ77 codelength) is a usable signal for *penalising* degenerate repetitions at decoding time. | **PARTIAL** — it penalises, it does not detect or early-stop, and it uses codelengths, not a tail-vs-whole zlib ratio. |
| Circular Reasoning (above) | Loop onset can be predicted from internal/semantic signals (CUSUM). | **PARTIAL** — supports "detect before the text repeats", but with hidden-state statistics, not compression. |
| This repository, `oracle/frontier/termination.py` | A zlib level-9 ratio of the last 2,000 characters to the whole answer, combined with word n-gram repetition and an exact-period test, separates loop tails from long derivations on the stored stage-1 corpus. | **UNSUPPORTED by the literature found here**; it is an original implementation, calibrated on our data (`results/stage1/s56-termination-calibration.json`). The "prediction-compression duality" framing in LZ Penalty is the nearest published support. |

**What this changes for S56.** We do not present the zlib detector as a known
method. It is a cheap, calibrated baseline; the Circular Reasoning paper is the
reason not to over-claim it.

## 4. On-policy distillation with truncation-aware rewards

| source | claim checked | status |
| --- | --- | --- |
| [On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes (GKD)](https://arxiv.org/abs/2306.13649), submitted 2023-06-23 | Standard KD suffers distribution mismatch; training the student on its own generated sequences with teacher feedback fixes it. | **VERIFIED** |
| [MiniLLM: On-Policy Distillation of Large Language Models](https://arxiv.org/abs/2306.08543), submitted 2023-06-14 | Replacing forward KL with reverse KL suits generative KD and supports an on-policy optimisation procedure. | **VERIFIED** |
| [Less is More: Early Stopping Rollout for On-Policy Distillation](https://arxiv.org/abs/2605.27028), submitted 2026-05-26 | Full rollouts suffer "Off-policy Teacher Decay" for later tokens; restricting the rollout to the first response tokens (ESR) improves OPD performance, GPU efficiency and stability. | **VERIFIED** |
| [Learning from Partial Chain-of-Thought via Truncated-Reasoning Self-Distillation](https://arxiv.org/abs/2603.13274), submitted 2026-02-27 | Training a student on a truncated reasoning prefix against a teacher target built from the full trace improves robustness to truncated inference and shortens outputs. | **VERIFIED** |
| [DAPO: An Open-Source LLM Reinforcement Learning System at Scale](https://arxiv.org/abs/2503.14476), submitted 2025-03-18 | In long-CoT RL, punitive rewards for truncated samples introduce reward noise; **Overlong Filtering** (mask the loss of truncated samples) and **Soft Overlong Punishment** stabilise training. | **VERIFIED** |
| [Understanding R1-Zero-Like Training: A Critical Perspective (Dr. GRPO)](https://arxiv.org/abs/2503.20783), submitted 2025-03-26 | GRPO has an optimisation bias that artificially increases response length, especially for incorrect outputs. | **VERIFIED** |
| "On-policy distillation with truncation-aware rewards" as one published method | A single method that combines OPD with a truncation-aware reward was not found. | **UNSUPPORTED** — GKD/MiniLLM establish OPD, ESR/TRSD establish truncation-aware variants, DAPO establishes truncation-aware RL rewards; the combination is a registered hypothesis here, not a published result. |

**What this changes for S56.** Two points are in tension with our registration
and are recorded rather than smoothed:

1. **Truncated = wrong conflicts with DAPO's finding.** DAPO reports that
   penalising truncated samples adds reward noise, and masks their loss instead.
   Our registration counts a truncated rollout as wrong in the *verified
   reward* used to admit or weight stage-2b targets. That is defensible for
   corpus admission (a truncated solution is not a verified target), but it is
   not evidence that a punitive dense reward is better for policy gradients.
   Stage 2b must compare `truncated = wrong` against `mask the loss` before
   claiming a win; the S56 amendment registers this as an open choice, not a
   settled result.
2. **ESR cuts by position; our cut is by loop.** ESR reports that later tokens
   are the problem and truncates the rollout to the first response tokens.
   Our cut keeps the whole prefix and stops only when the tail degenerates.
   These are different interventions and should be compared, not conflated.

## 5. vLLM and Qwen3.5 support

| source | claim checked | status |
| --- | --- | --- |
| [vLLM v0.34.0 model registry](https://raw.githubusercontent.com/vllm-project/vllm/v0.34.0/vllm/model_executor/models/registry.py), read 2026-11-18 | `Qwen3_5ForCausalLM` maps to `vllm.models.qwen3_5`, and `Qwen3_5MoeForCausalLM` is registered. | **VERIFIED** |
| [vLLM supported-models documentation](https://raw.githubusercontent.com/vllm-project/vllm/v0.34.0/docs/models/supported_models.md), read 2026-11-18 | The table lists `Qwen3_5ForConditionalGeneration` for the Qwen3.5 family (text + image/video variants). | **VERIFIED** |
| [PyPI metadata for vllm 0.34.0](https://pypi.org/pypi/vllm/0.34.0/json), read 2026-11-18 | 0.34.0 requires `torch==2.14.0` and `transformers<5.23.0,>=5.16.1`; uploaded 2026-11-16. No vLLM release requires torch 2.15.0. | **VERIFIED** |
| Running our post-trained Qwen3.5-0.8B/2B checkpoints under vLLM 0.34.0 in this repository | Not run: the image was not built (disk), and the smoke is queued. | **UNSUPPORTED in this repository** — the model support is verified upstream, the throughput is not measured here. |

**What this changes for S56.** vLLM 0.34.0 does support Qwen3.5 checkpoints
upstream, so the smoke is worth queueing. It still cannot share the scoring
image because of the torch pin, and no throughput number may be quoted until the
smoke receipt exists (`docs/S56-VLLM.md`).

## 6. Effect on the S56 design (summary)

| S56 decision | literature position |
| --- | --- |
| textual loop detector used for compute-saving early cut | supported as a cheap baseline; **not** an early semantic detector (Circular Reasoning) and not a published zlib method |
| no repetition penalty, decoding unchanged | supported: standard penalties are weak (LZ Penalty); changing decoding changes the measurement |
| truncated rollout is not a verified target | supported for corpus admission; **contradicted** as a dense reward by DAPO — registered as an ablation |
| same-prefix teacher KL | consistent with MiniLLM's reverse KL and GKD's student-state objective; not a novel claim |
| early cut rather than full rollout | consistent with ESR's finding that later rollout tokens hurt OPD; the cut rule differs (loop vs position) |
