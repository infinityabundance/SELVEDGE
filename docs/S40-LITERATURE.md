# S40 focused literature — distillation scale, frozen embeddings, 8-bit optimizers, decoding parity

Status: **focused verification, 2026-11-17 (S40).** Each entry names the URL, the
fetch date, the claim the source itself states, and a verdict for the question
this session had to decide. `VERIFIED` means the claim is in the fetched abstract
or record; `PARTIAL` means the source supports part of the claim or a different
setting; `UNSUPPORTED` means no source was located that states the claim. An
abstract does not settle a full paper, and the verdicts are scoped to what was
read.

## 1. Full-backbone distillation versus adapters at equal tokens

| source | fetched | claim as stated | verdict |
| --- | --- | --- | --- |
| Houlsby et al., *Parameter-Efficient Transfer Learning for NLP*, `https://arxiv.org/abs/1902.00751` (ICML 2019) | 2026-11-17 | adapters "attain within 0.4% of the performance of full fine-tuning, adding only 3.6% parameters per task" on 26 text-classification tasks including GLUE | **VERIFIED** for classification. It is not evidence about decoder generation or about distillation at a fixed token budget. |
| Biderman et al., *LoRA Learns Less and Forgets Less*, `https://arxiv.org/abs/2405.09673` (TMLR 2024) | 2026-11-17 | "in the standard low-rank settings, LoRA substantially underperforms full finetuning" on programming and mathematics, in both instruction tuning (~100K pairs) and continued pretraining (20B tokens); "full finetuning learns perturbations with a rank that is 10-100X greater than typical LoRA configurations" | **VERIFIED**. The continued-pretraining regime is token-matched, which is the regime our control approximates. |
| Shuttleworth et al., *LoRA vs Full Fine-tuning: An Illusion of Equivalence*, `https://arxiv.org/abs/2410.21228` (v3, 2025) | 2026-11-17 | LoRA and full fine-tuning produce spectrally different weight matrices; LoRA has "intruder dimensions" absent from full fine-tuning, and its forgetting localises there | **VERIFIED**. Supports treating a low-rank update as a different solution class, not a cheaper full fine-tune. |
| Azimi et al., *KD-LoRA*, `https://arxiv.org/abs/2410.20777` (ENLSP-IV 2024) | 2026-11-17 | LoRA combined with knowledge distillation "achieves performance comparable to full fine-tuning (FFT) and LoRA" on GLUE, with lower cost, on BERT/RoBERTa/DeBERTaV3 | **PARTIAL**. Encoder classification, not decoder generation; it shows the gap is not universal, which is why the comparison instrument reports both families rather than assuming one direction. |

**Decision.** The frozen-backbone adapter is not the bar. Published evidence says
low-rank/frozen adaptation can be close on classification-shaped tasks and
substantially behind full fine-tuning on code/mathematics at matched tokens. Our
full-backbone arm is therefore the bar, and the anchored harness reports
knowledge and math separately so a mechanism cannot win by being close on one
family only. **Verdict for the exact setting (0.8B student ← 2B teacher, 20M
tokens, online KL): PARTIAL** — no located paper runs this exact pair and budget,
so the arm is necessary rather than redundant.

## 2. Embedding-frozen distillation

| source | fetched | claim as stated | verdict |
| --- | --- | --- | --- |
| Dettmers et al., *QLoRA*, `https://arxiv.org/abs/2305.14314` (NeurIPS 2023) | 2026-11-17 | backpropagates "through a frozen, 4-bit quantized pretrained language model into Low Rank Adapters"; reports preserving "full 16-bit finetuning task performance" | **PARTIAL**. It is the nearest precedent for a frozen embedding space with the remaining weights adapted, but the rest is LoRA, not full-backbone, and the embedding is frozen as part of the whole quantized base. |
| Sanh et al., *DistilBERT*, `https://arxiv.org/abs/1910.01108` (NeurIPS 2019 workshop) | 2026-11-17 | distils a smaller model during pretraining with a triple loss; the student is trained, including its embeddings | **VERIFIED**, and it is *not* a precedent for freezing the embedding: DistilBERT trains it. |
| search: `all:"frozen embeddings" AND all:"language model"` (`https://export.arxiv.org/api/query?...`) | 2026-11-17 | five results, none about freezing the tied embedding during full-backbone LM distillation | **UNSUPPORTED**. No located paper studies freezing only the tied token embedding while training every other text-tower parameter. |

**Decision.** Freezing the tied embedding is a registered engineering choice for
this arm, not a published result. Its justification is local: the vocabulary is
unchanged, the teacher and student tokenizers are byte-identical, and the tied
output projection is frozen with it. The literature does not establish that this
is harmless, and the registration says so.

## 3. 8-bit optimizers for LLM finetuning

| source | fetched | claim as stated | verdict |
| --- | --- | --- | --- |
| Dettmers et al., *8-bit Optimizers via Block-wise Quantization*, `https://arxiv.org/abs/2110.02861` (ICLR 2022 spotlight) | 2026-11-17 | "the first optimizers that use 8-bit statistics while maintaining the performance levels of using 32-bit optimizer states"; block-wise dynamic quantization, a stable embedding layer; demonstrated on 1.5B-parameter language modeling, GLUE finetuning, WMT'14, ImageNet and RoBERTa pretraining | **VERIFIED** for memory saving at comparable performance. |
| Dettmers et al., *QLoRA*, `https://arxiv.org/abs/2305.14314` | 2026-11-17 | "paged optimizers to manage memory spikes" | **PARTIAL**. Paging is not 8-bit moment storage; it is a different memory technique. |

**Decision and honest limit.** The arm uses block-wise int8 moments with a
float32 scale per 256 elements in `oracle/frontier/optim8bit.py`. That is the
published construction, not bitsandbytes' dynamic non-linear quantization, and
bitsandbytes is absent from the pinned image. The claim is therefore
"8-bit-statistics Adam of the published construction, CPU-tested against AdamW",
not "bitwise equal to bitsandbytes". **UNSUPPORTED** for bitwise equivalence.

## 4. Memory-table train/inference equivalence and cached-decoding parity

| source | fetched | claim as stated | verdict |
| --- | --- | --- | --- |
| NGM, *A Plug-and-Play Training-Free Memory Module for LLMs*, `https://arxiv.org/abs/2605.16893v2` | 2026-11-17 | a causal n-gram encoder over the pretrained token-embedding space plus a cosine-gated memory injector, training-free; evaluated on ten benchmarks and six backbones | **PARTIAL**. The mechanism is stated; the abstract does not report cached-decoding parity or a statefulness test. |
| Roy et al., *N-Grammer: Augmenting Transformers with latent n-grams*, `https://arxiv.org/abs/2207.06366` | 2026-11-17 | augments a Transformer with n-grams from a discrete latent representation of the sequence; evaluates language modeling on C4 and SuperGLUE | **UNSUPPORTED** for decoding parity: the abstract states none. |
| Wu et al., *Memorizing Transformers*, `https://arxiv.org/abs/2203.08913` (ICLR 2022 spotlight) | 2026-11-17 | approximate kNN lookup into a non-differentiable memory of recent (key, value) pairs at inference; memory up to 262K tokens | **UNSUPPORTED** for decoding parity: the abstract states none, and an inference-time memory makes the question acute rather than answered. |
| Pope et al., *Efficiently Scaling Transformer Inference*, `https://arxiv.org/abs/2211.05102` | 2026-11-17 | analytical treatment of generative inference, KV-cache partitioning, multi-query attention, per-token latency | **VERIFIED** for incremental decoding as the standard serving path, but it does not discuss memory modules outside attention. |

**Decision.** No checked published memory-augmented model reports that its cached
decoding reproduces its full-sequence forward. The absence is in abstracts, so it
is not proof about full texts; it is enough to make the parity test mandatory
here. The S40 tests do exactly what the literature does not: a non-identity
convolution and non-zero gates, full-sequence output versus token-by-token
cached output, and full-sequence logits versus cached continuation logits.
`results/stage2/memory-2m-compat-20261117T023857Z.json` extends that to the
registered checkpoint.

## What this document does not establish

- It does not establish that the 20M-token budget transfers the published
  matched-token findings; 20B continued-pretraining tokens (Biderman et al.) is a
  different regime.
- It does not establish bitsandbytes parity, or that freezing the tied embedding
  costs nothing.
- It does not establish that any published memory model is *incorrect* during
  cached decoding; only that the checked sources do not report the check.
