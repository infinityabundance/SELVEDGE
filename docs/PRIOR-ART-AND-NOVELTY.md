# Prior art and novelty ledger

Status: **live document, first complete pass 2026-11-15.** Required by
`docs/MANDATE.md` ("REQUIRED LITERATURE FORENSICS" and "REQUIRED DOCUMENTS").
This ledger exists to stop the program from claiming a mechanism that is already
published, and to stop it citing a paper that does not say what the strategy
note says it says.

## 1. Method and how to check this document

Every entry below was resolved by an actual HTTP request made in the Docker
python court with network explicitly enabled
(`scripts/court --net python -- python3 scripts/prior_art_fetch.py`), on
2026-11-15 (UTC). The raw bytes that back each status are in
`results/prior-art/prior-art-fetch-20261115*.jsonl`: for every source, the URL,
final URL after redirects, HTTP status, retrieval timestamp, returned title and
the first part of the abstract (or page description) are recorded.

Four methods were used:

- `arxiv_id` — authoritative metadata from the arXiv Atom API for a known id.
- `arxiv_search` / `arxiv_all` — title-phrase and all-field searches. **All**
  candidate titles are recorded, so a near-miss cannot silently become a
  citation.
- `page` / `page_abstract` — publisher landing pages (PMLR, ACL Anthology,
  AAAI OJS, ICLR, IJCAI, PMC, alphaXiv, vLLM docs).
- `github` / `raw_grep` — repository pages and raw files.

Three sources could not be read at their canonical location and are recorded as
such rather than guessed:

- `doi.org/10.1145/3694715.3695964` and `dl.acm.org/doi/10.1145/3694715.3695964`
  both returned **HTTP 403** (bot protection). PowerInfer was verified instead
  through the Semantic Scholar DOI record and the project repository.
- `proceedings.iclr.cc` and `ijcai.org` expose only a truncated description in
  metadata; the Monet and DoMoE abstracts were therefore resolved through arXiv
  (`2412.04139`) and the proceedings page body respectively.

**Status vocabulary.**

| status | meaning |
| --- | --- |
| `VERIFIED` | the source exists and says what the claim needs it to say; the evidence is quoted or tightly paraphrased from the fetched record |
| `PARTIAL` | the work exists but the fetched evidence covers only part of the claim, or only a citing work could be read |
| `UNSUPPORTED` | the searches performed did not locate a source supporting the claim; the mechanism must be treated as `PROPOSED` |
| `PROPOSED` | no prior art is claimed for this mechanism; it is the program's own hypothesis |

**Counts.** 53 named works were checked, plus 4 supplementary works found during
verification:

- `VERIFIED`: 52
- `PARTIAL`: 3 (Engram, D²-MoE, CMoE)
- `UNSUPPORTED`: 2 (NorSA, HOPE)
- `PROPOSED`: every SELVEDGE mechanism in §5 that is not in this list

**Second pass, 2026-11-15 (S9).** The WS7–WS12 mechanism list named 42 further
works. They were fetched in rounds 9–17
(`results/prior-art/prior-art-fetch-20261115-round{9..17}.jsonl`); their verdicts
are in §10, with 15 supplementary works in §10.1. Totals after that pass — 95
named works and 19 supplementary works:

- `VERIFIED`: 52 + 24 = 76 named, + 7 supplementary = **83**
- `PARTIAL`: 3 + 6 = 9 named, + 8 supplementary = **17**
- `UNSUPPORTED`: 2 + 12 = **14**
- `PROPOSED`: every SELVEDGE mechanism in §5 and in `docs/PROGRAM-PLAN.md`
  WS7–WS12 that is not in either list

The 8 supplementary `PARTIAL` rows are works whose *identity* (title, id, date)
is verified from a fetched search or page but whose abstract was not read, so only
the existence of the mechanism is supported. That distinction is kept out of the
named counts on purpose: a named work gets `VERIFIED` only when the fetched text
supports the mechanism, not merely the title.

A source that is `UNSUPPORTED` is not thereby non-existent; it is not located by
the recorded searches (§8). The distinction matters for this pass because 12 of
the 42 names could not be located at all, and several of them are load-bearing
citations for a proposed experiment.

**Third pass, 2026-11-16 (S16C, WS13).** Twenty named works were added with the
eleven P13.6–P13.21 inventions and the four published mechanisms WS13 stages.
They were fetched in rounds 18–18d
(`results/prior-art/prior-art-fetch-20261116-round18{,b,c,d}.jsonl`); the verdicts
are in §12. New rows: `VERIFIED` 9, `PARTIAL` 4, `UNSUPPORTED` 7, plus
one supplementary `VERIFIED` (PonderNet).

**Correction to the third-pass counts, 2026-11-16 (S20).** Rows 115 (MoSE),
117 (GMoE) and 118 (TYLER) were false negatives caused by arXiv-only search and
by landing pages that had not yet served an abstract. All three were re-fetched
directly at their venues and are now `VERIFIED` (§12.1, §12.1b). The third pass
is therefore `VERIFIED` 12, `PARTIAL` 2, `UNSUPPORTED` 6, plus one supplementary
`VERIFIED` (PonderNet). The pre-correction counts above are retained as history,
not as the current tally.

**Count discrepancy, recorded not repaired.** The second-pass paragraph above
says "95 named works" while its own category counts (76 + 9 + 14) sum to 99. The
eighteen-point row table in §10 and §10.1 is the authoritative record; the prose
total in §1 is short by four. This pass does not rewrite the earlier arithmetic
(it would be editing a previous verdict's bookkeeping without re-reading its
receipts). Counting rows instead: 114 numbered rows before this pass, 135 after.

## 2. Verification table — novelty-boundary list

| # | work | id / venue | status | recorded evidence | overlap with SELVEDGE |
| ---: | --- | --- | --- | --- | --- |
| 1 | Drop-Upcycling | arXiv 2502.19261 | VERIFIED | abstracts the upcycling slowdown and proposes "partial re-initialization" | dense→sparse upcycling, the P1.3 boundary |
| 2 | SIMoE (Sparse Interpolated MoE) | ACL 2025 long 816 | VERIFIED | "experts are structurally sparse subsets of the seed LLM's parameters"; router learns an input-dependent merging | expert upcycling + conditional merging |
| 3 | Memory Layers at Scale | ICML 2025 (PMLR v267) | VERIFIED | "trainable key-value lookup mechanism to add extra parameters ... without increasing FLOPs" | learned memory path (P3.1) |
| 4 | Engram | DeepSeek (primary not located) | PARTIAL | four 2026 works describe it: "Deepseek's Engram, a conditional memory module ... token-level N-gram hashing" (arXiv 2607.29065); "Engram conditional memory ... decoupling static knowledge lookup from dynamic computation" (arXiv 2603.10087). **2026-11-15 (S9):** an ACL 2026 long paper named directly as the Engram/Conditional Memory source was fetched at `https://aclanthology.org/2026.acl-long.226/` — "Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models", Xin Cheng et al., Proceedings of the 64th Annual Meeting of the ACL (Volume 1: Long Papers), 2026. Title, venue and authors are verified; the landing page serves no abstract, so the method is still described only through citing works | hashed conditional memory (P3.1) |
| 5 | SeqMoE | arXiv 2609.12978 | VERIFIED | recasts expert prediction as sequence modelling + "Job Sequencing with Deadlines" prefetch scheduling | predictive loading (P3.7) |
| 6 | Prefetch-Steered Routing | arXiv 2610.34721 | VERIFIED | "cache-aware routing ... makes the model's outputs depend on transfer timing"; proposes a deterministic logical cache; 2.2–2.3× faster than exact routing with LRU on OLMoE | cache-aware routing and its semantic hazard (P3.7) |

## 3. Verification table — activation sparsity and routing

| # | work | id / venue | status | recorded evidence | overlap |
| ---: | --- | --- | --- | --- | --- |
| 7 | Deja Vu | ICML 2023 (PMLR v202) | VERIFIED | "contextual sparsity ... can be accurately predicted" with an asynchronous hardware-aware implementation | learned pre-execution selection |
| 8 | PowerInfer | SOSP 2024 (DOI 10.1145/3694715.3695964) | VERIFIED | "power-law distribution in neuron activation ... hot neurons preloaded ... cold neurons computed on CPU", plus an adaptive predictor | hot/cold residency, activation prediction |
| 9 | Attribution-based Sparse Activation | PMC PMC13570667 (2026) | VERIFIED | "quantified the large errors of existing attribution metrics ... due to the interdependency among attribution scores of different neurons, and ... a new attribution metric that can provably correct such errors" | attribution scoring + the group-dependence caveat |
| 10 | SEAP | AAAI-26 (OJS 40463) | VERIFIED | "task-relevant expert activation paths ... training-free pruning ... clustering patterns of hidden states and neuron activations" | task-conditioned selection |
| 11 | SparseInfer | ICML 2026 (PMLR v306) | VERIFIED | "core neurons ... sentence-level prediction" | semantic, pre-execution activation prediction |
| 12 | UpIT | ACL 2025 long 637 | VERIFIED | "intermediate checkpoints ... are naturally suitable for specialized experts ... genetic algorithm and parameter merging" | expert upcycling |
| 13 | PEER | arXiv 2407.04153 | VERIFIED | "product key technique for sparse retrieval from a vast pool of tiny experts (over a million)" | fine-grained retrieval over many small experts |
| 14 | Monet | ICLR 2025 / arXiv 2412.04139 | VERIFIED | "scaling the expert count to 262,144 per layer while total parameters scale proportionally to the square root of the number of experts" | very many experts, fine-grained indexing |
| 15 | MoSE | ICML 2026 (PMLR v306) | VERIFIED | "nested, slimmable ... executed at variable widths" | variable-width conditional compute |
| 16 | DAG-MoE | ICML 2026 (PMLR v306) | VERIFIED | replaces weighted-sum aggregation with structural aggregation | expert-output composition |
| 17 | ProbMoE | ICML 2026 (PMLR v306) | VERIFIED | "models expert selection as a distribution over cardinality-constrained expert subsets" | differentiable routing |
| 18 | MoTE | arXiv 2506.14435 | VERIFIED | ternary experts for memory-efficient MoE | low-bit expert storage |
| 19 | PuzzleMoE | ICML 2026 (PMLR v306) | VERIFIED | "fine-grained element-wise merging" + bit-packed inference | redundancy-aware expert storage |
| 20 | MoE-Infinity | GitHub EfficientMoE/MoE-Infinity | VERIFIED | README: "keeps hot experts resident on the GPU, while tracing and prefetching hide transfer cost" | offloading + prefetch |
| 21 | SpecMD | arXiv 2602.03921 | VERIFIED | benchmarks cache policies and finds "MoE expert access is not consistent with temporal locality assumptions (e.g LRU, LFU)" | cache-policy evidence **against** locality assumptions |
| 22 | SpecPrefetch | arXiv 2607.24787 | VERIFIED | "a shared lightweight adapter to predict next-layer expert candidates only for asynchronous transfer, while the frozen native router still determines the final executed experts" | **exactly** P3.7's predictor/router separation |
| 23 | DoMoE | IJCAI 2026 (proceedings 657) | VERIFIED | "domain-aware semantic expert prediction ... under expert offloading"; effectiveness "critically depends on accurate and efficient expert prediction" | domain-level prediction for offloading |
| 24 | Task-aware coactivation grouping | arXiv 2606.01007 | VERIFIED | "expert co-activation is strongly task-conditioned"; task-aware placement instead of one globally aggregated plan | task-conditioned co-access placement |
| 25 | MoE-CORE | arXiv 2610.01950 | VERIFIED | coordinated offloading/residency; "an optional score-based substitution path handles eligible low-score misses" | residency + an approximate-substitution path |
| 26 | OLMoE | arXiv 2409.02060 | VERIFIED | abstract says "uses only 1B per input token"; the paper body contains "1.3B" (HTML match) | the 1B-active MoE baseline; confirms the mandate's warning about the name |
| 27 | Huginn | arXiv 2502.05171 | VERIFIED | iterated recurrent block, arbitrary test-time depth | recurrent depth axis (P5.11) |
| 28 | Critical recurrent analysis | arXiv 2507.02199 | VERIFIED | "limited evidence of interpretable latent CoT ... significant problems" in Huginn-3.5B | the caution on P5.11 |

## 4. Verification table — mechanisms named in WS5 and the infrastructure list

| # | work | id / venue | status | recorded evidence | overlap |
| ---: | --- | --- | --- | --- | --- |
| 29 | vLLM prefix caching | vLLM docs v0.13.0 | VERIFIED | "block hash of hash(tuple[components])" with "parent hash value" and "block tokens" | exact-prefix KV identity |
| 30 | GGUF spec | ggml docs/gguf.md | VERIFIED | "GGUF is a file format for storing models for inference with GGML ... binary format designed for fast loading and saving" | on-disk format (not a novelty claim) |
| 31 | Candle | GitHub huggingface/candle | VERIFIED | "Minimalist ML framework for Rust" | Rust inference primitives |
| 32 | lm-evaluation-harness | GitHub EleutherAI | VERIFIED | "framework for few-shot evaluation of language models" | independent evaluation |
| 33 | SpAx | arXiv 2610.02598 | VERIFIED | "replace the binary choice of whether or not to read a weight with three options: fully retain it, approximate ..." for offloaded weights | **three-way retain/approximate/omit** (P4.3) |
| 34 | R-Sparse | arXiv 2504.19449 | VERIFIED | "training-free activation sparsity ... rank-aware" | training-free selection baseline |
| 35 | La RoSA | arXiv 2507.01299 | VERIFIED | "layerwise orthogonal rotations to transform input activations ... Top-K selection within the rotated activations" | rotation before sparsification (P5.3) |
| 36 | Sirius | arXiv 2409.03856 | VERIFIED | contextual sparsity "significantly degrades ... for reasoning, deduction, and knowledge-based tasks"; adds token corrections | correction path for sparse execution |
| 37 | BlockFFN | arXiv 2507.08771 | VERIFIED | sparsely activated models "exhibit low chunk-level sparsity ... the union of multiple consecutive tokens activates a large ratio of parameters" | chunk-level union (P5.2) |
| 38 | NeuroPrefetcher | arXiv 2608.22643 | VERIFIED | "storage-aware sparse LLM inference via delta prefetching"; model-exceeds-memory setting where "storage becomes an active source of weights on the critical path" | delta prefetch from storage (P4.5/P5.1) |
| 39 | ToMoE | arXiv 2501.15316 | VERIFIED | dynamic structural pruning that converts MLPs into an MoE at a fixed active parameter count | dense→MoE conversion |
| 40 | Cluster-aware upcycling | arXiv 2604.13508 | VERIFIED | partitions input activations into semantic clusters; experts initialised from cluster subspaces via truncated SVD; router initialised to cluster centroids | **exactly** P3.6's semantic-cluster initialisation |
| 41 | DeRS | arXiv 2503.01359 | VERIFIED | "decomposes the experts into one expert-shared base weight and multiple expert-specific delta weights" | shared-base + delta experts (P5.3) |
| 42 | D²-MoE | primary not located | PARTIAL | only through a citing work: RFID-MoE (arXiv 2602.09316) compares against "state-of-the-art methods like MoBE and D2-MoE" | expert compression baseline |
| 43 | MASKerade | arXiv 2610.07809 | VERIFIED | "learns experts as sparse subnetworks of a frozen pretrained FFN ... a token-level router selects which masked FFNs to execute" | masked expert bank over frozen weights (P5.8) |
| 44 | Mixture-of-Recursions | arXiv 2507.10524 | VERIFIED | shared layer stack across recursion steps; routers assign per-token depths; "selectively caching only their key-value pairs" | adaptive depth + recursion-wise KV (P5.8) |
| 45 | WINA | arXiv 2505.19427 | VERIFIED | weight-informed neuron activation; argues magnitude-only criteria give "high approximation errors" | magnitude vs weight-aware scoring (P5.5) |
| 46 | NorSA | not located | UNSUPPORTED | `ti:"NorSA"` and `all:"NorSA"` both returned zero entries from the arXiv API | must be treated as PROPOSED |
| 47 | SGA-Sparse | arXiv 2611.11223 | VERIFIED | trains "the existing gate projection directly as a top-k router"; the gate-vs-up ordering "reverses after sparsity-aware training" | native gate as router (P5.6) |
| 48 | LoPA / Model Casting | arXiv 2609.31975 | VERIFIED | "mid-training recipe that drastically sparsifies the activations within the FFN"; introduces "LoPA Gating" | low-parameter gating (P5.6) |
| 49 | HOPE | not located | UNSUPPORTED | `ti:"HOPE"` and `all:"HOPE" AND all:"mixture of experts"` returned only unrelated works (quantum outreach, zero-shot learning, speech detection) | must be treated as PROPOSED |
| 50 | SeqTopK | arXiv 2511.06494 | VERIFIED | "Sequence-level TopK ... shifts the expert budget from the token level to the sequence level ... less than 1% overhead" | sequence-level budget allocation (P5.7) |
| 51 | CMoE | primary not located | PARTIAL | MLPMoE (arXiv 2511.21089) names "CMoE, ToMoE, and Mo..." among dense→MoE upcycling methods; no primary CMoE upcycling paper was located by title or all-field search | upcycling prior art |
| 52 | Fast Feedforward | arXiv 2308.14711 | VERIFIED | log-time alternative to FFNs with "noiseless conditional execution" | conditionally executed nodes |
| 53 | SPON | arXiv 2512.12744 | VERIFIED | "Spontaneous Neurons (SPON) ... a small set of learnable, input-independent activation vectors that act as persistent representational anchors" | resting/anchor vectors (P5.9) |

### Supplementary works found during verification

| # | work | id | status | why it matters |
| ---: | --- | --- | --- | --- |
| 54 | Temporally Extended Mixture-of-Experts | arXiv 2604.20156 | VERIFIED | options-framework controller that "reduces switch rates from over 50% to below 5% while retaining up to 90% of base-model accuracy" on MATH, MMLU and MMMLU; directly relevant to P3.7's negative-result claim |
| 55 | Pooling Engram Conditional Memory using CXL | arXiv 2603.10087 | VERIFIED | Engram tables in a CXL pool with prefetching, "near-DRAM end-to-end performance"; prior art for memory-tier placement |
| 56 | RFID-MoE | arXiv 2602.09316 | VERIFIED | SVD-based MoE compression keyed on routing frequency and information density; the citing work for D²-MoE |
| 57 | MLPMoE | arXiv 2511.21089 | VERIFIED | zero-shot dense→static-MoE metamorphosis; names CMoE/ToMoE among upcycling methods |

## 5. Feature-overlap matrix

For each SELVEDGE candidate mechanism: the closest antecedent, the overlapping
features, and what is genuinely uncombined.

| SELVEDGE candidate (item) | closest antecedent(s) | overlapping features | what is genuinely uncombined |
| --- | --- | --- | --- |
| Hashed prefix memory path (P3.1) | Engram; Memory Layers at Scale; PEER; Monet | n-gram hashed keys, learned value rows, sparse retrieval over 2^18–2^20 rows, a few reads per token | the **strict per-token union accounting of retrieved rows plus evaluated index parameters**, and scoring the memory against a parameter-matched dense adapter and an expert bank in one matrix |
| Conditional residual specialist bank (P3.2) | PEER, Monet, MoSE, Drop-Upcycling, ToMoE | small conditional experts, residual addition, top-k routing | the non-duplication audit (residual correlation with the backbone) and the union-accounted comparison against a dense adapter |
| Error-directed router on `u_e(x)` (P3.3) | Deja Vu, PowerInfer, Sirius, attribution-based sparse activation, SparseInfer, SEAP | predicting pre-execution usefulness of a conditional computation from the hidden state | measuring utility as **incremental benefit of added capacity** (not omission loss) and testing it against magnitude/probability routing at equal union budget |
| Five-arm first matrix (P3.4) | standard MoE ablations; Deja Vu/PowerInfer sparse-vs-dense comparisons | dense vs specialised vs hybrid at matched compute | memory-only / expert-only / naive-hybrid / error-directed arms with one backbone, one data order and independent union audit |
| Cold-start initialisation (P3.6) | Cluster-aware upcycling; Drop-Upcycling | cluster-centroid router init, SVD-derived experts, symmetry breaking | a measured specialisation comparison across random / dense-derived / semantic-cluster init, with a "more tokens does not fix it" test |
| Separate prefetch predictor, miss = stall (P3.7) | **SpecPrefetch**; SeqMoE; SpecMD; Prefetch-Steered Routing; Temporally Extended MoE | predictor separate from the router; deterministic cache; next-layer candidate prediction; deadline scheduling | scoring the predictor's bytes inside the same per-token union/bytes ledger as the selector, and the explicit refusal of router stickiness with the semantics-preservation court |
| Three-way per-group decision (P4.3) | **SpAx** | retain / approximate / omit for offloaded weights | tier assignment by **predicted error contribution** (from P3.3's utility model) rather than magnitude, at the group granularity |
| Bytes/bit-width equivalence reporting (P4.4) | offloading papers report transfer volume | bytes per token, residency | unique-parameter + bytes + **equivalent bit-width** as a mandatory triple for every quality row |
| Resident low-rank + sparse exact delta (P5.1) | DeRS; NeuroPrefetcher (delta prefetch); PuzzleMoE | shared base, delta representation, compressed storage | the scoring rule `|x|·||W_col − Q(W_col)||` and the pre-registered 15% bytes kill criterion |
| Chunk-tiered tiering (P5.2) | BlockFFN | chunk-union activation sparsity | using the chunk union as the **fetch unit** and measuring chunk-PPL degradation against per-token PPL |
| Rotated shared-base delta experts (P5.3) | La RoSA + Cluster-aware upcycling + DeRS | rotation, cluster init, delta experts | the combination, with an inter-expert-similarity kill criterion |
| Adaptive dense correction from delta mass (P5.4) | Sirius (token corrections) | a correction pass for sparse execution | the correlation-based build/kill gate (Spearman ≥ 0.5 build, < 0.3 kill) |
| Quantisation-error tiering (P5.5) | WINA; PuzzleMoE; MoTE | weight-aware scoring, low-bit storage | the test of whether magnitude already proxies quantisation error |
| Native gate as router + low-parameter gating (P5.6) | **SGA-Sparse**; Model Casting / LoPA Gating | gate projection used as the selector; parameter-free routing | measuring whether the selector cost is **removed** under the union budget, and the post-training ordering reversal |
| Sequence-level budget (P5.7) | **SeqTopK**; Temporally Extended MoE | shifting the budget from token to sequence | per-output-token union compliance at the worst token, not just on average |
| Masked bank + adaptive depth (P5.8) | **MASKerade** + **MoR** | masked subnetwork experts over frozen weights; per-token recursion depth; recursion-wise KV | the combination with a profiler-verified "reads only selected elements" audit |
| Cheap interaction signals (P5.9) | SPON (verified); M2's linear bilinear surrogate | input-independent anchors; pairwise interaction scores | tree routing and online centroids at matched selector cost, with `HOPE` treated as PROPOSED because it could not be verified |
| Oracle-in-the-loop distillation (P5.10) | M2's own offline oracle; Deja Vu predictor training | an offline oracle labels, a cheap model imitates | applying it to **added capacity** selection and reporting the distillation gap to the oracle bound |
| Recurrent latent computation (P5.11) | Huginn; critical recurrent analysis; MoR | latent recurrence, adaptive depth | nothing claimed; explicitly deferred and separately pre-registered if ever started |
| Joint physical execution frontier (P6.2) | MoE-Infinity, MoE-CORE, SpecMD, SeqMoE, Prefetch-Steered Routing | same-GPU-memory comparisons, TPOT/p50/p95, residency | the joint operating point: error-directed capacity + retrieval + physical execution under one union ledger, reported as a Pareto frontier |

## 6. Consequential findings

These are the findings that change what the program may claim or must do.

1. **P3.7's design is already published.** SpecPrefetch (arXiv 2607.24787)
   predicts next-layer experts with a separate lightweight adapter while "the
   frozen native router still determines the final executed experts". The
   program's contribution cannot be the predictor/router separation or the
   "miss is a stall" rule; it has to be the joint budget accounting and the
   measured operating point.
2. **P4.3's three-way decision is already published.** SpAx (arXiv 2610.02598)
   replaces read/skip with "fully retain, approximate, omit" for offloaded
   weights. Only the *predicted-error* tier assignment is open, and it depends
   on P3.3 delivering a calibrated error model.
3. **The 2026 stickiness negative result is not the last word.** Temporally
   Extended MoE (arXiv 2604.20156) reports switch rates from over 50% to below
   5% while retaining up to 90% of base-model accuracy on MATH/MMLU/MMMLU, using
   an options-framework controller. The program must cite this and argue why a
   *no-stickiness* rule is still right, rather than presenting its own failed
   experiment as the state of the art.
4. **Locality is not established, and SpecMD says so.** SpecMD (arXiv
   2602.03921) finds MoE expert access "is not consistent with temporal locality
   assumptions (e.g LRU, LFU)". Any part of WS3/WS4 that assumes temporal
   locality — including a prefetch predictor trained on history — must be
   treated as an empirical question, not a premise.
5. **The predictive-loading space is crowded.** SeqMoE, SpecMD, SpecPrefetch,
   DoMoE, Prefetch-Steered Routing, MoE-CORE, NeuroPrefetcher and MoE-Infinity
   each hold part of it. A SELVEDGE claim of "predictive loading" alone would be
   unsupportable.
6. **`HOPE` and `NorSA` do not resolve to the claimed mechanisms.** The arXiv
   API returned no title match for either, and all-field searches returned only
   unrelated papers. They are marked `UNSUPPORTED`; if the strategy note's
   mechanisms are wanted they must be specified from first principles and called
   `PROPOSED`.
7. **`Engram`, `D²-MoE` and `CMoE` are known only through citing works.** Engram
   is described consistently by four 2026 papers (conditional memory, n-gram
   hashing, sparse access, prefetchable), but the primary report was not
   located; D²-MoE is named as a state-of-the-art compression baseline by
   RFID-MoE; CMoE is named as an upcycling method by MLPMoE. That is enough to
   call them prior art, not enough to quote their methods.
8. **The "1B active" naming trap is real.** OLMoE-1B-7B's abstract says "1B per
   input token", while the paper body says 1.3B; the mandate's warning is
   confirmed by the fetched HTML match. This is why P1.2 must use the strict
   union count and never a model's name.
9. **`gate` vs `up` ordering reverses after sparsity-aware training.**
   SGA-Sparse (arXiv 2611.11223) reports that the ordering that holds for
   training-free routing reverses after training the native gate as a router.
   Any comparison that ranks gate-based and up-based routing from a *pretrained*
   checkpoint is therefore not transferable to a trained model.
10. **Very many tiny experts is established.** Monet reports 262,144 experts per
    layer; PEER over a million. "Dormant capacity" is not itself novel; the
    organisation of retrieval under a union and I/O budget is the only open
    question.

## 7. The target claim, stated plainly

SELVEDGE does **not** claim to be first at: causal attribution, activation
sparsity, expert upcycling, learned memory, expert prefetching, cache-aware
routing, sequence-level expert budgets, masked experts, delta-compressed
experts, or low-bit experts. Those are prior art, and the ledger above names the
closest antecedent for each.

The target claim is a **joint operating point**: a trained, error-directed system
that selects among learned memory and learned specialist capacity, predicts the
physical cost of executing that selection, and is measured under a strict
per-token unique-parameter union with bytes per token and wall-clock reported
alongside quality. The novelty, if any, is in the joint design and in the
measured causal-cost advantage under that accounting — not in any single
mechanism.

## 8. What this ledger does not establish

- It does not prove the *absence* of prior art. arXiv searches cannot cover
  closed venues, non-English work, or papers whose title does not contain the
  named acronym. `UNSUPPORTED` means "not found by the recorded searches", not
  "does not exist".
- It does not verify numbers inside the papers beyond what was fetched. Where a
  paper's advantage is quoted, it is the paper's own claim, not a reproduction.
- It does not resolve `doi.org` 403s. PowerInfer is the one entry where the
  canonical DOI was unreadable; it is supported by the Semantic Scholar record
  and the project repository instead.
- It is not a substitute for reading the papers in full before building on
  them. §5's "genuinely uncombined" column is a claim about *published
  mechanism descriptions*, and a full-text reading could narrow it further.

## 9. Refresh procedure

```bash
# re-fetch everything (writes results/prior-art/prior-art-fetch-<date>*.jsonl)
scripts/court --net python -- python3 scripts/prior_art_fetch.py \
    --out results/prior-art/prior-art-fetch-$(date -u +%Y%m%d).jsonl

# one source only
scripts/court --net python -- python3 scripts/prior_art_fetch.py \
    --out results/prior-art/prior-art-fetch-check.jsonl --only specprefetch
```

Re-run at every milestone advance and whenever an item in
`docs/PROGRAM-PLAN.md` moves from `PROPOSED` to implementation.

**Second-pass procedure (uses the HTML fallbacks).** `export.arxiv.org` rate-limits
by caller (HTTP 429, rounds 9–11). When that happens, do **not** read a 429 as "no
such paper": rerun the key with `arxiv_page` (known id) or `arxiv_html_search`
(name), and record the 429 in the same JSONL so the attempt is visible:

```bash
# one name, HTML search, when the API is refusing
scripts/court --net python -- python3 scripts/prior_art_fetch.py \
    --out results/prior-art/prior-art-fetch-20261115-round17.jsonl \
    --only foam-title,lookahead-offload-search --sleep 4

# one known id, arxiv.org/abs page
scripts/court --net python -- python3 scripts/prior_art_fetch.py \
    --out results/prior-art/prior-art-fetch-check.jsonl \
    --only engram-acl-226-page --sleep 4
```

An `UNSUPPORTED` verdict in §10 must list the queries attempted; that list is the
evidence, not the absence of a result.

---

## 10. Verification table — WS7–WS12 works (second pass, 2026-11-15)

This section covers the 42 works named by the WS7–WS12 mechanism list. It is a
separate table rather than an extension of §2–§4 because it includes the
`UNSUPPORTED` verdicts that remove citations, and those must be visible as a
group.

**Method addendum.** `export.arxiv.org` answered **HTTP 429** for most requests
from the court container in rounds 9–11; the 429s are recorded in those JSONL
files as findings, not discarded. Two methods were added to
`scripts/prior_art_fetch.py` for this reason and are used from round 12 on:

- `arxiv_page` — reads `https://arxiv.org/abs/<id>` directly (title, abstract,
  submission history). Different host, no API quota.
- `arxiv_html_search` — reads `https://arxiv.org/search/?...` and records every
  candidate id and title, so a miss is visible as a list of near-misses.

Every row below was resolved by one of: `page_abstract` on the publisher page
(ACL Anthology), `arxiv_id`/`arxiv_page` for a known id, or
`arxiv_html_search` for a name. Where a row is `UNSUPPORTED`, the **queries
tried** are listed in the "does not support" column — that list is the evidence.

| # | work as named | located (id / venue) | status | what it supports | what it does not support | SELVEDGE addition |
| ---: | --- | --- | --- | --- | --- | --- |
| 58 | Engram / Conditional Memory (ACL 2026 long 226) | ACL 2026 long 226: "Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models", Cheng et al. | PARTIAL | title, venue, authors, and that the paper frames conditional memory as a sparsity axis | the method: the landing page serves no abstract, so n-gram hashing and sparse lookup still come from citing works (entry 4, 2607.07388) | a union-and-latency ledger for retrieved rows **plus the index parameters evaluated to find them** |
| 59 | Memory Grafting | 2605.20948 | VERIFIED | frozen hidden states of a grafting model as conditional n-gram memory; exact longest-match suffix lookup; hash-based Engram fallback; beats MoE and vanilla Engram at 2.8B (paper's claim) | any admission criterion for *new* memory entries beyond frequency | P11.3's verified-novelty admission test |
| 60 | Counterfactual Routing (ACL 2026 long 2187) | ACL 2026 long 2187: "Awakening Dormant Experts: Counterfactual Routing to Mitigate MoE Hallucinations" | VERIFIED | static top-k favours high-frequency patterns; causally important specialists are under-gated; training-free intervention | a *trained* router for causal importance, and any cost accounting | P9.3's interaction-aware utility target, trained, under the union budget |
| 61 | Agent Memory Distillation (EMNLP findings 2407) | ACL Anthology `2026.findings-emnlp.2407` | VERIFIED | training-free transfer of hierarchical teacher memory (workflow, subtask) into a small agent | any capability representation other than text-like memory, and any cost accounting | P9.1's transfer-retention test and a byte-accounted comparison against weights |
| 62 | SkillCache | 2611.10022: "How Much Skill Orchestration Can LLM Agents Compile Away? Execution-Grounded Specialization..." | VERIFIED | zero-shot skill specialisation is brittle; a one-trace compiled fast path works | that the artefact is a weight-space capability (it is an executable plan) | P9.6's compile/verify/solve cost split, and P11.14's fallback contract |
| 63 | the consumer-scale Engram study | nearest: 2607.07388 (TF-Engram), 2608.17050 | PARTIAL | a train-free Engram with SSD-backed tables, GPU–DRAM–SSD hierarchy and early-exit guided predictive prefetching on Qwen3-0.6B | that a *study* exists as the note implies; one system paper is not a study, and the note's claim is stronger than the record | a measured consumer-residency claim inside the union/bytes ledger |
| 64 | What Is This Memory Useful For? | 2611.03930 | VERIFIED | task-conditioned utility via a task-blind writer and frozen reader with a cost ledger; answer-bearing memories beat matched controls with an interval excluding zero (paper's claim) | anything about marginal utility of **added capacity**, or interactions between two memories | P9.3's interaction term `I(m,e;x)` applied to capacity rather than memory |
| 65 | Infinite-Parameter LLMs | 2609.18842 | VERIFIED | generating and adapting weights from live data | physical cost under a union budget | P10.1's generated-vs-stored expressivity test at matched storage |
| 66 | Shaping Latent MoE | 2610.14644 | VERIFIED | local Jacobian rank bounded jointly by expert input and intermediate width; near-square experts at fixed per-expert parameters; layerwise capacity allocation | intrinsic-dimension-guided allocation from measured data; any union ledger | P10.6's measured-ID allocation at matched totals |
| 67 | LatentMoE / Nemotron | 2601.18089; 2602.04870; 2610.40577; Nemotron-3 reports | VERIFIED | accuracy per FLOP and per parameter; adopted into Nemotron-3; O(1)-communication variant; multi-view routing | bytes per token, residency, cold-start cost | P10.2/T3: take the geometry as given and measure what it costs physically |
| 68 | Titans | 2501.00663 | VERIFIED | a neural long-term memory module that learns at test time | versioning, rollback, tier separation | P10.3's stable/episodic/working split with rollback |
| 69 | MIRAS | nearest 2504.13173: "It's All Connected: A Journey Through Test-Time Memorization, Attentional Bias, Retention, and Online Optimization" | PARTIAL | Transformers, Titans and linear RNNs unified as associative memories with an internal objective | the acronym MIRAS (absent from the fetched text), so the note's naming is stronger than the record | P7.7's objective as a design constraint on the boundary, not a framework citation |
| 70 | TTT-E2E | 2512.23675 | VERIFIED | continues learning at test time by next-token prediction, compressing context into weights; constant latency vs context (paper's claim) | per-token union accounting or residency | charging the write inside the arm ledger |
| 71 | TTT-NTP | 2606.21803 | VERIFIED | fast-weight writes supervised by the model's own next contextual hidden state | tiering or rollback | P10.3's tier split and the document-removal target task |
| 72 | Select-TTT | 2610.36794 | VERIFIED | uniform write schedules dilute query-relevant information; query-conditioned write strength from the frozen head, no new parameters | multi-stage relevance change or consolidation | P10.7's staged version and the write→consolidate transition |
| 73 | LOTUS | 2606.31779: "Bridging the Gap Between Latent and Explicit Reasoning with Looped Transformers" | VERIFIED | looped padded transformer with CE on each latent position's gold CoT step; matches explicit CoT at 3B; large thought-phase latency reduction (paper's claim) | the note's title — LOTUS is the method inside that paper, not a paper title | matched-wall-clock comparison against search, and trajectory→operator compilation (P11.14) |
| 74 | Latent Reasoning with Supervised Thinking States | 2602.08332 | VERIFIED | thinking tokens generated while the input is processed, transformed to embedding space, supervised by teacher forcing | cost accounting against CoT at matched wall clock | P10.4's four-arm matched-clock court |
| 75 | Multi-View MoE (29 Oct 2026) | 2610.40577, v1 submitted Thu 29 Oct 2026 | VERIFIED | single-router rankings disagree with expert-utility rankings, especially in middle and deep layers | any accounting of the multi-view selection's own cost | P10.5's router-parameter accounting inside the active budget |
| 76 | intrinsic-dimension expert allocation | nearest 2610.14644 | PARTIAL | expert geometry ↔ Jacobian rank, and layerwise allocation | a paper that allocates by *measured intrinsic dimension of the data* | P10.6 as specified, with ID as the measured allocation signal |
| 77 | MemCredit | 2611.15600 | VERIFIED | counterfactual intervention credits distilled into a utility estimator; relevance for recall, utility for arbitration | weight-space capacity, or any pairwise interaction term | P9.3's four-condition interaction utility is the pair version of this idea |
| 78 | Memory-Aware Routing | 2610.23583 (MARS) | VERIFIED | session state as shared control interface between memory construction and model selection | union accounting of router plus memory parameters | P10.5's counted-router comparison |
| 79 | Doc2Weight | **not located** | UNSUPPORTED | — | `all:"Doc2Weight"`, `ti:"Doc2Weight"`, `all:"Doc2" AND all:"weight"`, `all:"document to weight" AND all:"adaptation"` returned no related entry | must be specified from first principles and called PROPOSED; nearest published relatives are P9.1/P10.10 |
| 80 | MemAgent | 2609.32521 | VERIFIED | 13 memory methods, none generalises; agent memory formulated as a routing problem over providers | whether provider routing composes under a byte budget | P8.2/P9.3 composition accounting |
| 81 | KV-Skill | 2608.05475 | VERIFIED | external factorised operators read by a frozen model through a lightweight interface; registration from text and reward-learned operators | rollback/versioning; a document-removal target | P11.11 leasing and P10.3 tiering on top of the same interface |
| 82 | ReaComp | **not located** | UNSUPPORTED | — | `all:"ReaComp"` (twice), `all:"reasoning compression"`, and a latent variant returned no entry for the name; adjacent ReCUT (2506.10822) is a different method | the citation cannot be used; PROPOSED |
| 83 | Mixture-of-Recursions | 2507.10524 (entry 44) | VERIFIED | shared stack, per-token depth, selective KV caching | union accounting of retained KV | P10.9's KV byte accounting |
| 84 | MI-Distillation | **not located** | UNSUPPORTED | — | `all:"MI-Distillation"`, `all:"mutual information" AND all:"distillation"`, and the reasoning variant returned no such method | P10.10 is stated from first principles |
| 85 | One-Token Verification | **not located** | UNSUPPORTED | — | `all:"one-token" AND all:"verification"`, `AND all:"correctness"`, `all:"one-token"`, and a token/verifier/confidence query returned homonyms only | P10.11 is PROPOSED; the citation is unsupported |
| 86 | CORE | **not located** | UNSUPPORTED | — | `ti:"CORE" AND all:"reasoning"`, `all:"CORE" AND all:"advantage"`, `all:"CORE" AND all:"chain-of-thought"` returned unrelated works | the name attracts homonyms; any CORE claim must be dropped or re-specified |
| 87 | LAPO | **not located** | UNSUPPORTED | — | `all:"LAPO" AND all:"reasoning"`, `all:"LAPO" AND all:"policy"`, `all:"LAPO"` returned zero entries | as above |
| 88 | ScMoE | 2404.05019 | VERIFIED | shortcut-connected MoE to overlap expert-parallel communication | any capability-side use (skipping work rather than hiding latency) | P10.14's capability-side shortcut, claimed narrowly and with re-tested dependencies |
| 89 | RFID-MoE | 2602.09316 | VERIFIED | SVD compression keyed on heterogeneous inter-group routing frequency and information density | allocation by verified capability loss | P10.15's allocation objective |
| 90 | Meta-Learning Where to Allocate Experts | **not located** | UNSUPPORTED | — | `ti:"Allocate" AND ti:"Experts"`, `all:"expert allocation" AND all:"meta-learning"`, and a budget variant (25 results, none the named work) | P10.13 is stated from first principles |
| 91 | Fast-weight Product Key Memory | **not located as a combination** | UNSUPPORTED | the components: product-key retrieval (2407.04153, 2412.04139) and fast weights (Titans, MIRAS, Select-TTT); `all:"product key memory"` returned only 2101.11685, which is static | the combination, and any writable-PKM claim | P10.17 as specified, with the static-store control and rollback |
| 92 | FOAM (12 Nov 2026) | **not located** | UNSUPPORTED | — | `ti:"FOAM"`, `all:"FOAM" AND all:"optimizer"`, `AND all:"AdamW"`, `AND all:"Muon"`, `AND all:"Lion"`, `AND all:"orthogonal"` returned foam-material physics or nothing | P10.18 keeps FOAM conditional and runs Muon vs tuned AdamW meanwhile |
| 93 | Manifold-Constrained Hyper-Connections | 2609.05309 (analysis on DeepSeek-V4-Flash); 2608.05549 (speech domain) | VERIFIED | mHC widens the residual pathway to n streams; a typical attention/FFN site effectively uses about two; read/write routing is concentrated and depth-dependent | capacity claims: the analysis says the *effective* stream count is low | P12.14's wildcard becomes a measured effective-stream test, not a capacity claim |
| 94 | xHC | **not located** | UNSUPPORTED | — | `all:"xHC"`, `all:"xHC" OR all:"extended hyper-connections"`, `all:"hyper-connections" AND all:"streams"` returned mHC/MSAR works and no xHC | the note's xHC claim is stronger than any located source; the family (mHC, MSAR) is published | the family is prior art; an xHC-specific claim needs its own definition |
| 95 | low-rank logits (ICLR 2026) | **not located** | UNSUPPORTED | — | `ti:"low-rank" AND ti:"logits"`, `all:"low-rank" AND all:"logits" AND all:"vocabulary"`, `AND all:"softmax"`, and an output-layer variant returned unrelated works | P12.14's low-rank-logits wildcard is PROPOSED, not a replication |
| 96 | SemPIC | **not located** | UNSUPPORTED | — | `all:"SemPIC"` (twice), `all:"SemPIC" OR all:"semantic prefix cache"`, `all:"semantic cache" AND all:"LLM"` (other semantic-cache works only) | semantic caching is prior art (supplementary rows); the named method is not |
| 97 | Routing in Gradient Space | no paper by that title; the mechanism is 2610.00493 | PARTIAL | routing should score the *update*: usefulness is the inner product of an expert's update with the negative loss gradient, i.e. gradient-space routing under another title | the note's title and citation; any full-width-expert version | P9.3 can cite the right paper; the note's citation must be corrected |
| 98 | ProbMoE | PMLR v306 (entry 17) | VERIFIED | expert selection as a distribution over cardinality-constrained subsets | union/bytes accounting | P11.9's constrained selection builds on it |
| 99 | recent future-aware offloading (11 Nov 2026) | nearest: 2611.11364 (v1 8 Nov 2026, training-side), 2603.09983 (MoE-SpAc, speculative activation), 2410.17954, 2609.12978, 2608.21614, 2610.33404 | PARTIAL | predictive and future-aware offloading is thoroughly published, including predictive caching with token scheduling and speculative activation | the specific 11 Nov 2026 work the note names: no fetched record supports that date-plus-claim pair | nothing claimed; the family is crowded (§6.5) |

### 10.1 Supplementary works located in the second pass

| # | work | id | status | why it matters |
| ---: | --- | --- | --- | --- |
| 100 | Frozen Memory Is Not Enough | 2608.17050 | VERIFIED | asks whether the frozen Engram memory or its *reader* matters when moved across backbones; the design template for P12.13's transfer test |
| 101 | TF-Engram | 2607.07388 | VERIFIED | train-free Engram, SSD-backed tables, early-exit guided predictive prefetching; reports a gain over a parameter-matched LoRA on Qwen3-0.6B (paper's claim). The closest thing to the note's "consumer-scale" item |
| 102 | Multi-Head LatentMoE | 2602.04870 | VERIFIED | O(1) communication regardless of k; deterministic expert-parallel traffic |
| 103 | Descent-Aligned Routing | 2610.00493 | VERIFIED | the correct citation for the note's "routing in gradient space" (row 97) |
| 104 | Multi-Stream Attention Residuals | 2610.13630 | VERIFIED | multi-head depth attention over residual streams; part of the mHC/xHC family |
| 105 | ExpertFlow | 2410.17954 | VERIFIED | predictive expert caching plus token scheduling for single-GPU offloading; the strongest published form of the note's "future-aware offloading" |
| 106 | OOM (Offload-Optimized MoE) | 2611.11364 | VERIFIED | offload-optimised full-parameter MoE training with memory/execution co-design; v1 8 Nov 2026 |
| 107 | MemoryAthena | 2609.25853 | PARTIAL | identity only (adaptive routing over latent and generated memories) |
| 108 | SkillMoV | 2606.17615 | PARTIAL | identity only (mixture-of-view routing with prototype-conditioned gating) |
| 109 | ReCUT | 2506.10822 | PARTIAL | identity only; the nearest published relative of the unfound "ReaComp" |
| 110 | MoE-SpAc | 2603.09983 | PARTIAL | identity only (speculative activation utility for edge MoE inference) |
| 111 | From Exact Hits to Close Enough | 2603.03301 | PARTIAL | identity only (semantic caching for LLM embeddings); the nearest relative of the unfound "SemPIC" |
| 112 | vCache | 2502.03771 | PARTIAL | identity only (verified semantic prompt caching) |
| 113 | CNN with large memory layers | 2101.11685 | PARTIAL | identity only; the only "product key memory" hit, and it is static memory |
| 114 | Multi-View MoE with vision-language reranking | 2609.18139 | PARTIAL | identity only (a homonym of the note's Multi-View MoE, different task) |

## 11. Consequential findings from the second pass

1. **Twelve named works do not exist as named.** `Doc2Weight`, `ReaComp`,
   `MI-Distillation`, `One-Token Verification`, `CORE`, `LAPO`,
   `Meta-Learning Where to Allocate Experts`, `Fast-weight Product Key Memory` (as
   a combination), `FOAM`, `xHC`, `low-rank logits (ICLR 2026)` and `SemPIC` were
   not located by any recorded search. In every case the note's claim is stronger
   than the record. Two of them are load-bearing: **P10.18's optimiser court names
   FOAM as an arm**, and **P10.11's mechanism is described as if published**. Both
   items now say so explicitly and carry a PROPOSED label.
2. **One citation is right about the mechanism and wrong about the paper.**
   "Routing in Gradient Space" is not a title; the mechanism is published by
   2610.00493 as descent-aligned routing. The idea stands, the citation does not.
3. **Three names are method names, not paper titles.** LOTUS is the method in
   2606.31779; SkillCache is the method in 2611.10022; TF-Engram is a system paper.
   Citing the acronym as a title would be a false citation even though the work
   exists.
4. **MIRAS is an acronym that the located paper does not use.** 2504.13173 is the
   unifying test-time-memorisation framework, and it is the right source for the
   *content*; the *name* is not in the fetched text.
5. **The Engram primary source is now located but not readable as an abstract.**
   ACL 2026 long 226 is "Conditional Memory via Scalable Lookup", verified by
   title, venue and authors. Any statement about its method still rests on citing
   works. Entry 4 is updated accordingly.
6. **Two of the note's most useful items are already published in stronger form
   than the note implies.** MemCredit (2611.15600) already attributes memory by
   counterfactual intervention, and the 2611.03930 study already measures
   task-conditioned memory utility with a cost ledger. SELVEDGE's novelty is the
   **pairwise interaction term** and its use as a *capacity* selection target, not
   the utility idea.
7. **mHC's own analysis undercuts a capacity reading of it.** 2609.05309 finds a
   typical site effectively uses about two of four streams. A SELVEDGE mHC/xHC
   wildcard must therefore be an *effective-stream* measurement, not a capacity
   claim. This is the single most useful negative finding of the pass.
8. **The latent-MoE geometry axis is fully published** (2601.18089,
   2610.14644, 2610.40577) including the near-square result and the
   router-ranking/utility-ranking disagreement. P10.2/T3 must therefore be
   presented as an accounting replication with a union ledger, never as a
   discovery.
9. **The fast-weight/adaptation axis is crowded but the target task is not.**
   Titans, MIRAS, TTT-E2E, TTT-NTP, Select-TTT and KV-Skill all write state at
   test time. None of the fetched abstracts reports a **document-removal** test
   with rollback and versioning. That is the part of P10.3 worth claiming.
10. **arXiv's Atom API is not reliable enough to be a single source.** Round 9–11
    returned HTTP 429 for most queries. The ledger now uses two HTML fallbacks and
    records the 429s. Any future refresh must expect this and must not read a 429
    as "no such paper".

---

## 12. Verification table — WS13 mechanisms and the sixteen inventions (third pass, 2026-11-16)

This pass covers the four published mechanisms that `docs/PROGRAM-PLAN.md` §18
stages (P13.1–P13.4) and the nearest prior art for each of the sixteen P13.6–
P13.21 inventions. Raw evidence:
`results/prior-art/prior-art-fetch-20261116-round18{,b,c,d}.jsonl`.

**Method addendum, and why there are four round-18 files.** Six arXiv ids were
recalled from memory in round 18 and resolved to *different papers* (one to a
physics paper on dilepton final states, one to a Heun-algebra paper). Those
wrong-page records are kept: an id that resolves to another work is evidence
that the id was wrong, not evidence that the work does not exist. Round 18b then
asked for the same works with `ti:` field queries; the HTML search endpoint this
script uses does not accept `ti:`, so 18b's zeros are a query artefact and are
recorded as such. Rounds 18c/18d re-asked with all-field phrases, which is the
form that worked in rounds 1–17. Where a work is still missing after that, the
status is `UNSUPPORTED` and the queries attempted are listed in the row.

### 12.1 The four published mechanisms WS13 stages

| # | work | id / venue | status | recorded evidence | what it supports |
| ---: | --- | --- | --- | --- | --- |
| 115 | **MoSE** — Mixture of Slimmable Experts | ICML 2026, PMLR **306:119816–119836** (`https://proceedings.mlr.press/v306/tastan26a.html`) | **VERIFIED** (2026-11-16, S20; supersedes the round-18 `UNSUPPORTED` verdict) | fetched 2026-11-16 UTC. *MoSE: Mixture of Slimmable Experts for Efficient and Adaptive Language Models*, Nurbek Tastan, Stefanos Laskaridis, Karthik Nandakumar, Samuel Horváth. Abstract: "each expert has a **nested, slimmable structure that can be executed at variable widths**"; conditional computation "not only over **which** experts are activated but also over **how much** of each expert is utilized"; "a single pretrained MoSE model can support a more continuous spectrum of accuracy-compute trade-offs at inference time". Training = "multi-width training with standard MoE objectives"; inference maps "router confidence/probabilities to expert widths under a fixed budget" (incl. lightweight test-time training). Code: `https://github.com/tnurbek/mose`. **Round-18 history retained:** three arXiv queries found no widths work; that was a **venue false negative** — the paper lives in PMLR volume 306, not in arXiv's front search. | exactly P13.1: nested/variable expert width under a fixed compute budget. P13.1 is a **reproduction and extension of published MoSE**, not a proposed SELVEDGE mechanism |
| 116 | **MoRE** — Mixture of Reused Experts | arXiv 2609.18176 (COLM 2026 as cited; the arXiv record does not name a venue) | **VERIFIED** | abstract: "shares expert pools across groups of adjacent layers. Each layer retains its own router but selects from a larger shared pool, expanding the diversity of routing combinations without additional parameters"; "lightweight learnable depth embeddings" let shared experts distinguish layers | exactly the P13.2 mechanism: shared pool over adjacent layers + per-layer router + depth embeddings |
| 117 | **GMoE** — Global Mixture of Experts with Logit Propagation | ACL 2026 long 2065, pp. 44599–44614 (`https://aclanthology.org/2026.acl-long.2065/`; DOI 10.18653/v1/2026.acl-long.2065) | **VERIFIED** (2026-11-16, S20; supersedes the round-18 `PARTIAL` identity-only verdict) | fetched 2026-11-16 UTC; the full abstract is now served (Geonwoo Hong, Taehwan Kim): "GMoE uses **Global Experts shared across all layers** and adds a **Local Expert per layer**"; "a Global Router with a **GRU-based recurrent component shared across layers** and layer-specific routing heads that **propagate routing logits across layers**"; this "couples routing decisions across layers, progressively refines routing paths". **Round-18 history retained:** the landing page served no abstract on the earlier pass; a page that does not serve an abstract is *not* evidence the mechanism is unstated. | exactly P13.3: a recurrent, state-carrying router plus global + per-layer local experts, with routing logits propagated across layers. P13.3 is a **reproduction and extension of published GMoE**, not a proposal |
| 118 | **TYLER** — Typed Latent Reasoning for Language Models | EMNLP 2026 main 2004, pp. 45650–45670 (`https://aclanthology.org/2026.emnlp-main.2004/`; DOI 10.18653/2026.emnlp-main.2004) | **VERIFIED** (2026-11-16, S20; supersedes the round-18 `PARTIAL` identity-only verdict) | fetched 2026-11-16 UTC; the full abstract is now served (Hanyu Lin, Min Cai, Jiawei Wen, Haodi Zhang): "TYLER **learns a policy** that, at each decoding step, chooses between emitting a visible token and **invoking a typed latent operator**"; the operators "support **global planning, local state updates, or reusable procedural abstraction**"; +14.49 over CoT and +4.30 over the strongest baseline, and it "generalizes ... with the lowest forgetting". **Round-18 history retained:** no abstract was served on the earlier pass; that is a transient fetch gap, not evidence the mechanism is unstated. | exactly P13.4: a learned policy over typed latent operators with per-step allocation. P13.4 is a **reproduction and extension of published TYLER**, not a proposal |

**Verdict on the WS13 premise (corrected 2026-11-16, S20).** "Reproduce four
published mechanisms" is supported for **all four** at full evidence: MoSE
(row 115), MoRE (row 116), GMoE (row 117), TYLER (row 118). The round-18
verdict below is retained only as superseded history; it was produced by
arXiv-only searches and by landing pages that served no abstract, and neither
is evidence of non-existence.

> *Superseded history (2026-11-16, S16C round 18, retained verbatim):* "Reproduce four
> published mechanisms" is supported for **one** of the four at full evidence (MoRE),
> *identity-only* for two (GMoE, TYLER), and **not locatable** for one (MoSE).
> WS13 is therefore written as four competing programmes with their own controls,
> not as a replication of a known result set.
>
> *Correction:* this was a **retrieval false negative**, not a factual finding. WS13
> is therefore built as **four published baselines to reproduce and extend**, each
> with its own controls and a measured SELVEDGE delta.

### 12.1b Retrieval false negatives (method lesson, 2026-11-16 S20)

A failed fetch is not evidence of non-existence, and this ledger must never
convert one into a negative verdict. The round-18 failures were produced by
four mechanisms, each of which can silently return zero hits for a real paper:

1. **Wrong index.** MoSE is an ICML 2026 paper whose canonical record is the PMLR
   volume index (`proceedings.mlr.press/v306/`), not arXiv. arXiv field searches
   cannot see it.
2. **Abstracts not yet on a landing page.** ACL Anthology serves title/venue/authors
   before (or while) the abstract is populated; a title that renders without an
   abstract is a fetch gap, not an abstract-less paper.
3. **Field-query syntax.** `ti:` is not accepted by the HTML search endpoint
   (`scripts/prior_art_fetch.py`), so `ti:` rounds return artefactual zeros.
4. **Homonym capture.** "MoSE" also names Mixture-of-Shape-Experts (2504.09601)
   and Swin2-MoSE (2404.18924); a homonym hit is not the target and a homonym
   *only* result does not prove the target absent.

**Rule.** `UNSUPPORTED` means "not located by the recorded searches"; it must
carry the queries, and it must be re-tested against a venue-native index before
it is allowed to shape a plan. Rows 115/117/118 were re-fetched directly at
their venues and all three flipped to `VERIFIED`.

**Consistency check.** `scripts/prior_art_fetch.py` is a *fetch* tool and has no
ledger-consistency mode, so there is no automated check to re-run. The ledger
counts in §1 were therefore recounted by hand over the §12 rows: sixteen named
rows in §12.2 plus four in §12.1 is 20; with rows 115/117/118 moved, the third
pass is **VERIFIED 12, PARTIAL 2, UNSUPPORTED 6**, plus one supplementary
`VERIFIED` (PonderNet). A row-level audit (`scripts/prior_art_audit.py`, added
this pass) reproduces that arithmetic and fails closed on future drift.

### 12.2 Nearest prior art for the sixteen inventions (P13.6–P13.21)

| # | invention | nearest prior art | id | status | recorded evidence |
| ---: | --- | --- | --- | --- | --- |
| 119 | P13.6 Capability Seed Compilation | task vectors / task arithmetic | arXiv 2212.04089 | **VERIFIED** | "A task vector specifies a direction in the weight space of a pre-trained model, such that movement in that direction improves performance on the task" |
| 120 | P13.7 Constraint-Debt Reasoning | Reflexion | arXiv 2303.11366 | **VERIFIED** | "reinforce language agents not by updating weights, but instead through linguistic feedback"; agents "maintain" a verbal reflection across attempts |
| 121 | P13.8 Metamorphic Quotient Routing | ReCode (robustness under code rewrites) | not located | **UNSUPPORTED** | three queries; the only ReCode hit is a different paper, "ReCode: Updating Code API Knowledge with Reinforcement Learning" 2506.20495, which is not about semantics-preserving transforms. The nearest *located* relative is a code-generation benchmark (EvoCodeBench, 2404.00599) that appears only as a search candidate, identity only, and is not a robustness study |
| 122 | P13.9 Semantic Refinement Ladders | MatFormer (nested granularity, elastic inference) | arXiv 2310.07707 | **VERIFIED** | "MatFormer ... provide[s] elastic inference across diverse deployment constraints"; nested submodels at several granularities trained jointly |
| 123 | P13.10 Verified Constraint Distillation | Self-Debugging | arXiv 2304.05128 | **VERIFIED** | teaches a model to debug "its predicted program" from execution feedback ("rubber duck debugging") |
| 124 | P13.11 Certified Read Elision | CROWN (certified bounds) | not located | **UNSUPPORTED** | four queries over three rounds (a recalled id resolved to a Heun-algebra paper; two phrase searches returned unrelated works). The *concept* of certified bound propagation is standard and the item's requirement "must be a real bound, not a CI" is stricter than the ledger can currently evidence. Treat the bound machinery as PROPOSED until read |
| 125 | P13.12 Interaction Compilation | SMEAR (soft merging of experts into one operator); function vectors (composition) | arXiv 2306.03745; function vectors not located | **PARTIAL** | SMEAR: "avoids discrete routing by using a single `merged` expert constructed via a weighted average of all of the experts' parameters" — a *published* compilation of a mixture into one executed operator, which is exactly the shape of P13.12's mechanism. What it does **not** do is condition the compiled operator on a *measured pairwise interaction*; that residual is the invention. The two `function vectors` queries located only "Improving Activation Steering ... with Mean-Centring" 2312.03813 |
| 126 | P13.13 Conditional Representation Expansion | CoLT5 (conditional token-selective width) | not located | **UNSUPPORTED** | three queries, including `all:CoLT5` bare (0 hits). The conditional-computation idea is supported elsewhere in this ledger (§3: DejaVu, MoD-adjacent works); *this specific reference* is unverified |
| 127 | P13.14 Omission-Conditioned Recovery | GPTQ (layer-wise output reconstruction with error compensation) | arXiv 2210.17323 | **VERIFIED** | post-training quantisation that minimises layer-wise output error; the exact "strong layer-wise distribution alignment" baseline P13.14 must beat |
| 128 | P13.15 Counterexample-Preserving Compression | HAWQ (sensitivity-driven mixed precision) | not located | **UNSUPPORTED** | two queries (a recalled id resolved to a dilepton physics paper; the phrase search found an unrelated spectral-flatness quantisation paper). Mixed-precision-by-sensitivity is standard; the *citation* is unverified |
| 129 | P13.16 Latent Branch-and-Reconcile | self-consistency | arXiv 2203.11171 | **VERIFIED** | "samples a diverse set of reasoning paths ... and then selects the most consistent answer by marginalizing out the sampled reasoning paths" — sampling + reconcile, the baseline P13.16 must beat at equal cost |
| 130 | P13.17 Resident-Side-Information Coding | LoRA | arXiv 2106.09685 | **VERIFIED** | "injects trainable rank decomposition matrices into each layer"; a module delta carried as a low-rank factorisation |
| 131 | P13.18 Phase-Transition Capability Switching | SMEAR (nearest *located* work; different mechanism) | arXiv 2306.03745 | **UNSUPPORTED** | the located SMEAR is 2306.03745 (2310.17680 is CodeFusion), and its abstract is about *soft merging of expert parameters*, **not** sequence-level or phase-boundary routing. No fetched record supports "switch at learned phase boundaries"; the per-token conditional-depth family (§3 domain) is per-token by construction, which is the control P13.18 exists to beat. The mechanism is PROPOSED |
| 132 | P13.19 Executable Capability Contracts | selective classification (reject option at a declared risk) | arXiv 1705.08500 | **VERIFIED** | "allow[s] a user to set a desired risk level. At test time, the classifier rejects instances as needed, to grant the desired risk (with high probability)" — the abstain-on-violation analogue of a pre/postcondition contract |
| 133 | P13.20 Training for Future Locality | SeqMoE (predictability as a training target) | arXiv 2609.12978 | **PARTIAL** | carried from §2 row 5 (verified there): recasts expert prediction as sequence modelling, with deadline scheduling. It supports *predictive loading*; it does **not** evidence a router trained to be **both accurate and predictable with cold bytes and p95 as the acceptance metric**, which is P13.20's claim. That joint objective is PROPOSED |
| 134 | P13.21 Verifier-Calibrated Internal Stopping | DEER (early exit from convergence/uncertainty) | not located | **UNSUPPORTED** | three queries (a recalled id resolved to a cooperative-society paper; two phrase searches returned unrelated works). The *adjacent* verified prior art is PonderNet |

### 12.3 Supplementary work located in this pass

| # | work | id | status | why it matters |
| ---: | --- | --- | --- | --- |
| 135 | PonderNet | 2107.05407 | VERIFIED | "learns end-to-end the number of computational steps to achieve an effective compromise between training prediction accuracy, computational cost and generalization"; bounded expected computation is already published, so P13.21's claim must be the **independent verifier evidence**, not halting itself |

## 13. Consequential findings from the third pass

1. **MoSE is real and is the direct basis for P13.1.** *(Corrected 2026-11-16, S20;
   the round-18 text is retained as superseded below.)* MoSE = *Mixture of
   Slimmable Experts* (Tastan et al., ICML 2026, PMLR 306:119816–119836) and it
   states nested/slimmable experts executable at variable widths under a fixed
   budget. P13.1 is therefore a **reproduction and extension of a published
   baseline**, not a proposal.

   > *Superseded:* "MoSE is the weakest citation ... Three searches find no work
   > matching 'adaptive expert widths' under that name; the only MoSE in the corpus
   > is a shape-dictionary prompt method for image segmentation. P13.1 stays in the
   > plan, but as a MatFormer-style nested-width expert experiment with a MatFormer
   > citation, not as a MoSE reproduction." — This was a venue false negative; the
   > paper is in PMLR, not arXiv.
2. **All four WS13 mechanisms are fully evidenced.** *(Corrected 2026-11-16, S20.)*
   GMoE (ACL 2026 long 2065) states a GRU-based recurrent router shared across
   layers with routing-logit propagation, plus global and per-layer local experts;
   TYLER (EMNLP 2026 main 2004) states a learned policy over typed latent
   operators with per-step allocation. Both abstracts are now served and both
   support their P13 claims in full. P13.3 and P13.4 are **reproductions and
   extensions**, not proposals.

   > *Superseded:* "Two of the four 'published' WS13 mechanisms are title-only ...
   > their landing pages serve no abstract ... now marked PROPOSED." — The pages
   > were fetched again and serve full abstracts; a missing abstract on one fetch
   > is a transient gap, not a mechanism that was never stated.
3. **Six of the sixteen nearest-prior-art citations could not be located.**
   CROWN, CoLT5, function vectors, HAWQ, DEER and the robustness ReCode are all
   standard works; the ledger nonetheless cannot cite them, and the plan must not
   present them as verified baselines. Each affected item (P13.8, P13.11–P13.13,
   P13.15, P13.21) says so in its own text. P13.18 is the seventh: its mechanism
   has no located prior art at all, and the work chosen as its neighbour (SMEAR)
   turned out to be about soft expert merging, not phase switching.
4. **One homonym is a live hazard, one is retired.** *(Corrected 2026-11-16, S20.)*
   "ReCode" = Updating Code API Knowledge (2506.20495) is still a homonym hazard.
   "MoSE" is **no longer** a hazard in the negative sense: the target MoSE
   (PMLR v306) is real and distinct from Mixture-of-Shape-Experts (2504.09601);
   the shape-expert paper is an unrelated homonym, not evidence against MoSE.
5. **One strategy-note identity was wrong; MoSE's was not.** *(Corrected
   2026-11-16, S20.)* SMEAR is 2306.03745 (soft merging of experts) and
   2310.17680 is CodeFusion. MoSE is correctly identified as PMLR v306 and needs
   no re-specification. **The general lesson stands:** an identity is wrong
   until a fetched primary record supports it — but a failed fetch is not that
   record, so it can never by itself falsify an identity.
6. **The most defensible novelty in the sixteen is the conjunction, not any one
   mechanism.** Every invention has a located neighbour for its *component*
   (task vectors, Reflexion, MatFormer, GPTQ, self-consistency, LoRA, selective
   classification, PonderNet, SMEAR) or no neighbour at all (six of sixteen). What
   no fetched record shows is the specific *measured* target SELVEDGE sets —
   verified task success at a fixed active budget, bytes per output token and
   cold-byte p95, with a falsifying control per mechanism. That is the claim to
   make, and it is a claim about evidence, not about novelty of parts.
