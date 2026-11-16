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

**Fourth pass, 2026-11-16 (S24, WS15–WS18).** Thirty-five works were added with
the WS15–WS18 workstreams: ten alternatives to sparse MoE, fifteen storage,
eviction, scheduling and control-theory works, eight biological control
principles and the two cross-project repositories. They were fetched in rounds
19–24 (`results/prior-art/prior-art-fetch-20261116-round{19,20,21,22,23,24}.jsonl`);
the verdicts are in §14 and their consequences in §15. Tally: `VERIFIED` 17,
`PARTIAL` 14, `UNSUPPORTED` 4. The row-level arithmetic is checked by
`scripts/prior_art_audit.py`, which fails closed on drift.

**The fourth pass is weaker evidence than the third, and says so.** Semantic
Scholar answered one request and then returned HTTP 429 for every later request
in the pass; `en.wikipedia.org`'s REST summary endpoint did the same twice. No
row below is `VERIFIED` on a 429: a row is `VERIFIED` only when a primary record
(an arXiv abstract page, a venue landing page with an abstract, a PMLR page, or a
NIST publication page) was fetched and supports the mechanism. The eight
biological rows and three of the control-theory rows are secondary records and
are marked `PARTIAL` for that reason alone; they support that a principle exists
and is described as the plan claims, never that it transfers quantitatively to a
neural system.

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

## 14. Verification table — WS15–WS18 works (fourth pass, 2026-11-16)

This pass covers the works named by the four workstreams appended to
`docs/PROGRAM-PLAN.md` in S24: the alternatives to sparse MoE (WS15), the storage,
eviction and control-theory literature WS16 borrows from, the biological control
principles (P16.32), and the two cross-project repositories (P16.35, WS17). Raw
evidence: `results/prior-art/prior-art-fetch-20261116-round{19,20,21,22,23,24}.jsonl`.

**Method note for this pass.** Three source classes are recorded separately
because the class is part of the verdict: a **primary** record (arXiv abstract
page, ACL Anthology page *with* an abstract, PMLR page, NIST publication page); an
**index search**, whose hit may be a homonym; and a **secondary** record
(Wikipedia), which can never carry `VERIFIED` for a mechanism claim. Two request
faults occurred and are recorded rather than papered over: Semantic Scholar
answered the first `s2_search` (`kv-evict-s2`, HTTP 200) and returned HTTP 429 for
the remaining twelve; the Wikipedia REST endpoint returned 429 on two of
seventeen. Every 429 row is a *finding about the pass*, and the work behind it was
re-asked through arXiv, its venue, or its repository.

### 14.1 Alternatives to sparse MoE (WS15)

| # | work | id / venue | status | recorded evidence | what it bounds in WS15 |
| ---: | --- | --- | --- | --- | --- |
| 136 | **Expert Choice Routing** | arXiv 2202.09368 | **VERIFIED** | fetched 2026-11-16 UTC. "Prior work allocates a fixed number of experts to each token using a top-k function regardless of the relative importance of different tokens. To address this, we propose a heterogeneous mixture-of-experts employing an expert choice" — experts choose tokens, so per-expert load is variable | that compute allocated per *token* may vary is published; P15.1/P15.6 novelty cannot be "variable per-token compute" |
| 137 | **DA-MoE** | arXiv search `all:"DA-MoE"` | **UNSUPPORTED** | one hit: arXiv 2411.03025, *DA-MoE: Addressing Depth-Sensitivity in Graph-Level Analysis through Mixture of Experts*, whose abstract is about "graph neural networks ... the optimal depth of GNN layers depends on the scale of the graph data". That is a graph-learning mixture, not an LLM mixture. The name is a homonym | no located sparse-MoE system named DA-MoE exists; the plan must not cite it as an LLM alternative |
| 138 | **DynaMoE** | arXiv search `all:"DynaMoE"` | **UNSUPPORTED** | 25 candidates returned, all solar/stellar **dynamo** physics ("The Complex Magnetic Anatomy of a White Dwarf Chromosphere", "Convergence of Roberts flow dynamos"). The name does not resolve to a neural mixture. `s2_search` for the same name returned HTTP 429 | not located; treat as PROPOSED, or cite the nearest located dynamic-SMoE work instead |
| 139 | **Mixture-of-Depths** | arXiv 2404.02258 | **VERIFIED** | "transformers can instead learn to dynamically allocate FLOPs ... to specific positions in a sequence ... enforces a total compute budget by capping the number of tokens (k) ... The tokens to be processed are determined by the network using a top-k routing mechanism" | conditional depth under an explicit FLOP budget is published; P15.2 and P16.13 must be FLOP-matched against it, not against a dense baseline alone |
| 140 | **HyperNetworks** | arXiv 1609.09106 | **VERIFIED** | "an approach of using one network, also known as a hypernetwork, to generate the weights for another network ... trained end-to-end with backpropagation" | generating resident weights from a compact code is published (2016); P15.7 and P10.1 must claim the *accounting* of the generation cost, not generation itself |
| 141 | **Product-Key Memory** | arXiv **1907.05242** (*Large Memory Layers with Product Keys*) | **VERIFIED** | "The memory is very large by design ... up to a billion parameters with a negligible computational overhead. Its design and access pattern is based on product keys, which enable fast and exact nearest neighbor search." **Identity correction:** the id 1611.01587 recalled for this work resolved to *A Joint Many-Task Model*, a different paper (round 19, row retained in the receipt) | the product-key addressing P15.7 and P16.31 use is published; the plan's citation must be 1907.05242 |
| 142 | **Titans** | arXiv 2501.00663 | **VERIFIED** | "We present a new neural long-term memory module that learns to memorize historical context and helps attention to attend to the current context" — a learned test-time memory alongside attention | learned test-time memory is published; P15.7's separation of long-term storage from resident compute is the part with no located neighbour |
| 143 | **Dynamic Expert Sharing (2026)** | arXiv search `all:"expert sharing" AND all:"mixture of experts"` | **UNSUPPORTED** | one hit: 2610.09342, *Shared Low-rank Basis Factorization for Data-free Mixture-of-Experts Compression* — shares a **factorised basis** across experts for compression, which is not dynamic sharing of computation across requests. `s2_search` returned HTTP 429 | not located by name; P16.35's cross-request sharing claim must be stated as PROPOSED |
| 144 | **DExperts** | ACL 2021 long 522 | **VERIFIED** | "a decoding-time method for controlled text generation that combines a pretrained language model with 'expert' LMs and/or 'anti-expert' LMs in a product of experts. Intuitively ... tokens only get high probability if they are considered likely by the experts, and unlikely by the anti-experts" | subtractive composition at decoding time is exactly the precedent for P15.3 (negative terms in the objective); the mechanism is prior art, the typed evidence-grounded constraints are not |
| 145 | **PARROT** | PMLR v119 **liu20f**, *An Imitation Learning Approach for Cache Replacement* | **VERIFIED** | "we focus on the problem of cache replacement: choosing which cache line to evict ... we propose an imitation learning approach to automatically learn cache access patterns by leveraging Belady's, an oracle policy" (Evan Liu, Milad Hashemi, Kevin Swersky, Parthasarathy Ranganathan, Junwhan Ahn) | learned eviction with an offline Belady oracle is published (2020). P16.6 and P16.34 must claim the residual/fingerprint features and the safety constraint, not the learned controller or the oracle |

### 14.2 Storage, eviction, scheduling and control theory (WS16)

| # | work | id / venue | status | recorded evidence | what it bounds in WS16 |
| ---: | --- | --- | --- | --- | --- |
| 146 | **MoE-Infinity** | arXiv 2401.14361 | **VERIFIED** | "on personal machines ... batch size of one ... MoE models exhibit a high degree of activation sparsity ... we design a sparsity-aware expert cache, which can trace the sparse activation of experts during inference" | activation-aware expert caching/prefetch is published; P16.1's novelty is the joint multi-item risk-constrained objective, not prefetch |
| 147 | **Learning to Evict from Key-Value Cache** | arXiv **2602.10238** (Moschella, Manduchi, Sener, 2026) | **VERIFIED** | "Existing eviction or compression methods ... rely on heuristics, such as recency or past attention scores, which serve only as indirect proxies for a token's future utility ... We reframe KV cache eviction as an RL problem: learning to rank tokens by their predicted usefulness for future decoding ... KV Policy (KVP), a framework of lightweight per-head RL agents" | learned, future-utility-scored eviction is published; P16.1's *risk constraint* (P(ΔL>ε)<δ) and its cost accounting are the residual claims. Note the title is attested; the Apple affiliation is **not** stated in the fetched abstract and is therefore not asserted |
| 148 | **LKV** | EMNLP 2026 main 1183 | **VERIFIED** | "LKV (Learned KV Eviction) ... formulates KV compression as an end-to-end differentiable optimization problem. LKV integrates LKV-H to learn task-optimized global budgets, and LKV-T to derive intrinsic KV importance" (Enshuai Zhou et al., 2026) | learned *budgets* as well as selection are published; P16.23 (layerwise error allowances) and P16.1's per-component allocation have a direct neighbour |
| 149 | **TextNCA** | arXiv 2608.02050 | **VERIFIED** | "Can a strictly local, iterated, weight-shared computation primitive support language modelling ... The model **does not match a parameter-matched Transformer** at this scale (Hier-TextNCA 60.3 vs. ...)" | this is the **negative result** P15.2 cites. It supports the warning, not the mechanism: no fetched record shows an attractor-field computer matching a dense baseline |
| 150 | **MoE cache paging** | arXiv 2509.02408 (*Cache Management for Mixture-of-Experts LLMs*) | **VERIFIED** | "we introduce and study a new paging problem that models expert management ... frequently used experts should be stored in the fast cache rather than in the slower secondary memory" | the paging abstraction P16.17 formalises is published; the fetch-to-recompute decision under equal outputs is not shown in the abstract |
| 151 | **MoE cache evaluation** | arXiv 2608.07911 | **VERIFIED** | "we find the measurement fragile. With a trace-driven, event-atomic simulator over three MoE models ... we isolate three evaluation axes that change conclusions, not just numbers. Replay semantics: under a fused-event traffic contract, an inconsistent per-access replay inflates rec..." | external corroboration for P16.34: replay semantics and workload contamination decide conclusions. It also means P16.34's novelty is not "simulate a cache" |
| 152 | **LLM I/O bottleneck survey (Springer 2026)** | four queries: `s2_search`, Springer search page, two arXiv all-field searches | **UNSUPPORTED** | the Springer search page returns only "Search research articles and books"; `s2_search` returned 429; arXiv returned generative-AI education, vector-database and power-systems works, none a survey of LLM I/O bottlenecks | not located. The plan must not cite it; the primary substitute is row 151 |
| 153 | **Red-blue pebble game** (Hong & Kung, MIT) | arXiv `all:"red-blue pebble"` | **PARTIAL** | six primary works use the model in their own titles, e.g. *On Fine-Grained I/O Complexity of Attention Backward Passes*, *Red-Blue Pebbling with Multiple Processors: Time, Communication and Memory Trade-offs*, *On the Hardness of Red-Blue Pebble Games*; the 1981 primary record itself was not fetched and `s2_search` returned 429 | the I/O-complexity framing of P16.17 is attested by citing works; the classical attribution is **not** verified in this ledger and the item must say "the red-blue pebble model" rather than cite a page it did not read |
| 154 | **Rent-or-Buy with expert advice** | PMLR v97 **gollapudi19a** | **VERIFIED** | "we consider the classical rent-or-buy problem (also called ski rental), and obtain algorithms that provably improve their performance over the adversarial scenario by using these predictions. We also prove matching lower bounds" | bounded-regret rent-vs-load with learned advice is published; P16.15's novelty is the *load* arm and the SELVEDGE-specific recovery cost, not the online algorithm |
| 155 | **Successive refinement** | arXiv 1802.07458 (primary work in that line) | **PARTIAL** | "In the successive refinement problem, a fixed-length sequence ... is encoded into two codewords by two encoders in order to give two reconstructions ... One of two reconstructions is obtained by one of two codewords, and the other reconstruction is obtained by all two codewords" | the coarse-plus-refinement structure P16.13/P16.20 use is attested by a primary work in that line; the attribution to **Cover** is not verified by a fetched record, and the plan must not present it as one |
| 156 | **CFAR** | Wikipedia REST summary, *Constant false alarm rate* | **PARTIAL** | secondary record only: "Constant false alarm rate (CFAR) detection is a common form of adaptive algorithm" with sections "Principle", "Cell-averaging CFAR", "Sophisticated CFAR approaches". The `s2_search` that would have located a radar primary returned 429 | the principle P16.22 borrows exists and is described as an adaptive threshold; no primary record was read, so the item is an analogy, not a baseline |
| 157 | **NIST combinatorial interaction testing** | NIST SP 800-142, *Practical Combinatorial Testing* | **VERIFIED** | "The key insight underlying t-way combinatorial testing is that not every parameter contributes to every fault and most faults are caused by interactions between a relatively small number of parameters." (two CSRC project URLs returned 404; the SP page served the text) | t-way covering arrays P16.24/P16.12 use are published **for software**; the transfer to neural capability combinations is exactly what P16.24 says must not be assumed |
| 158 | **Braess' paradox** | Wikipedia summary + arXiv `all:"Braess" AND all:"paradox"` | **PARTIAL** | secondary: "the observation that adding one or more roads to a road network can slow overall traffic flow"; primary-computing neighbours: *Braess's Paradox in Uniform Affine Grid Networks*, *Braess's Paradox of Generative AI*, *Network Topology That Excludes Braess's Paradox*. The specific **"Springer 2023"** treatment named in the plan was not identified: the Springer search page returned only its chrome | the paradox exists and has computational analogues; the named 2023 citation is unverified, and P16.26 stays HYPOTHESIS for that reason |
| 159 | **IEA demand response 2026** | iea.org report page | **PARTIAL** | `iea.org` returned **HTTP 403** ("Just a moment..." — Cloudflare bot protection) on two URLs; the secondary record states "Demand response is a change in the power consumption of an electric utility customer to better match the demand for power with the supply" | the demand-shifting principle is attested secondarily; the IEA 2026 report itself is **not read**, so P16.27 must cite the principle, not the report |
| 160 | **Preisach model** | Wikipedia REST summary, *Preisach model of hysteresis* | **PARTIAL** | secondary: "the Preisach model of hysteresis is a model of magnetic hysteresis" with sections "Nonideal relay", "Discrete Preisach model", "Preisach plane"; `s2_search` returned 429 | the hysteresis model P16.11/P16.22 borrow exists; the superposition-of-relays construction is attested only by the secondary record's section list |

### 14.3 Biological control principles (P16.32) — secondary records only

Every row in this table is a Wikipedia record. That is deliberate and it caps
the verdict at `PARTIAL`: an encyclopedia entry supports that a principle
exists and is described as the plan claims. It does not support any quantitative
transfer to a neural system, and P16.32 is `HYPOTHESIS` for that reason.

| # | principle | record | status | recorded evidence | what it may inform |
| ---: | --- | --- | --- | --- | --- |
| 161 | microglial pruning | *Synaptic pruning*, *Microglia* (REST summaries) | **PARTIAL** | "Synaptic pruning is the process of synapse elimination or weakening"; microglia "are constantly scavenging the CNS for plaques, damaged or unnecessary neurons and synapses". The first extract does **not** attribute pruning to microglia, and the second does not name pruning — the attribution is **not** quoted by either | negative selection of capacity (P16.12, P16.32); the attribution is weaker than the plan's wording |
| 162 | proteasome | *Proteasome* (REST summary) | **PARTIAL** | "Proteasomes are essential protein complexes responsible for the degradation of p..." (tagged/selective degradation) | destroy-before-recompute (P16.32): a selective degradation channel, not a cache |
| 163 | autophagy | *Autophagy* (REST summary) | **PARTIAL** | "Autophagy is the natural, conserved degradation of a biological cell that remove[s] ..." | bulk degradation under scarcity (P16.32) |
| 164 | bacterial chemotaxis | *Chemotaxis* (REST summary, section "Bacterial chemotaxis / Signal transduction / Receptor regulation") | **PARTIAL** | "Chemotaxis is the movement of an organism or entity in response to a chemical st..."; the section list attests receptor regulation | adaptive thresholding (P16.22): the analogy for a threshold that tracks a background level |
| 165 | quorum sensing | *Quorum sensing* (REST summary) | **PARTIAL** | "quorum sensing or quorum signaling (QS) is the process of cell-to-ce[ll communication]" | agreement-gated action (P16.10): the analogy for requiring independent signals to agree |
| 166 | microbial dormancy | *Dormancy*, *Persister cells* (REST summaries) | **PARTIAL** | "Dormancy is a period in an organism's life cycle when growth, development, and p[hysical activity] are temporarily suspended"; "Persister cells are subpopulations of cells that survive under lethal treatment" | the dormant state in P16.9 and the rare-capability reserve in P16.29 |
| 167 | sleep replay | *Hippocampal replay* (REST summary) | **PARTIAL** | "During sleep or awake rest, replay refers to the re-occurrence of a sequence of cell[s] ..." | offline consolidation in P16.27 (compute peak shaving) |
| 168 | immune negative selection | *Central tolerance* (REST summary) | **PARTIAL** | "central tolerance is the process of eliminating any developing T or B lymphocytes that are autoreactive, i.e. reactive to the body itself" | the negative-selection rule in P16.12 and P16.32 |

### 14.4 Cross-project repositories (P16.35, WS17)

| # | repository | record | status | recorded evidence | what it actually implements |
| ---: | --- | --- | --- | --- | --- |
| 169 | **DSFB** | `github.com/infinityabundance/dsfb` + README via GitHub API | **VERIFIED** | README: "DSFB - Drift-Slew Fusion Bootstrap ... **DSFB creates human readable structure from usually discarded residuals**. Information from the signatures or fingerprints of noise. And it does it **without ML or LLM (AI)**, meaning it is white-box and can run on the edge ... DSFB doesn't replace existing, proven, tried and tested sensors and methods, it **augments** them ... as a **read-only observer**. ... it changes the status of residual exhaust from disposable diagnostic byproduct into replayable forensic evidence." Repository description: "Drift–Slew Fusion Bootstrap (DSFB)"; cited works are navigation, plasma state estimation and multi-sensor fusion (Zenodo DOIs) | a deterministic, read-only, provenance-carrying **residual diagnostics** framework for sensor fusion. It does **not** contain neural-cache code, an eviction controller, or the ten motifs of P16.2. The transfer is PROPOSED: what SELVEDGE borrows is the *stance* (deterministic observer, no mutation, replayable evidence), not code |
| 170 | **Phenom** | `github.com/infinityabundance/phenom` + README via GitHub API | **VERIFIED** | README: "**Phenomenological Workspace monorepo** ... the full phase-by-phase implementation of the **conversation operating system**"; crates `phenom-core` ("phenomenological system semantics, learning, empirical and alignment layers"), `phenom-storage` ("5-tier storage and lifecycle movement"), `phenom-graph`, `phenom-crdt`, `phenom-render`, `phenom-ui`; "Determinism first ... **Reason codes required:** policy decisions, degraded modes ..." | a Rust workspace/OS: storage tiers, event sourcing, graph, CRDT, rendering. The README contains **no** occurrence of "obligation" or "compartment". WS17's `phenom-obligation` boundary and the six compartments are therefore **SELVEDGE proposals about** Phenom, not descriptions of it |

## 15. Consequential findings from the fourth pass

1. **Three of the ten named WS15 works do not resolve to the mechanism the plan
   names.** DA-MoE (row 137) is a graph-learning mixture, not an LLM mixture;
   DynaMoE (row 138) resolves to solar dynamo physics; Dynamic Expert Sharing
   (row 143) has no located work, only a shared-basis *compression* paper. None
   may appear in the plan as a sparse-MoE alternative. They are removed from the
   comparison set and kept as `PROPOSED` vocabulary.
2. **Two identities were repaired, one of them load-bearing.** Product-Key
   Memory is 1907.05242, not the recalled 1611.01587 (row 141); the wrong id
   resolved to *A Joint Many-Task Model* and is retained in the receipt. PARROT
   is PMLR v119 `liu20f`, *An Imitation Learning Approach for Cache Replacement*
   — the plan's use of the name is correct, but the paper is a **cache-replacement**
   imitation learner, not a generic RL-labelled controller (row 145).
3. **Learned eviction is not a novelty surface.** Three independent published
   works learn eviction: PARROT with a Belady oracle (row 145), KV Policy with
   per-head RL agents (row 147), and LKV with end-to-end learned budgets and
   selection (row 148). P16.6, P16.23, P16.34 and P16.1 must therefore claim the
   *residual-conditioned, risk-constrained, non-mutating* part — and P16.34's
   learned arm is a reproduction of published baselines, not an invention.
4. **TextNCA is a negative result and is cited only as one.** The fetched abstract
   says the model "does not match a parameter-matched Transformer at this scale".
   No fetched record supports attractor-field computation as competitive, so
   P15.2's experiment is a falsification test, not a transfer of a known win.
5. **Every component of WS15 has a published neighbour.** Variable per-token
   compute (row 136), conditional depth (139), weight generation (140),
   product-key addressing (141), test-time memory (142), subtractive decoding
   (144). The WS15 novelty is the *obligation/constraint/evidence coupling and its
   measurement*, which is what P18.1–P18.5 test. Stated as a mechanism claim, none
   of P15.1–P15.8 would survive review.
6. **Four classic citations are analogues, not baselines.** Red-blue pebbling
   (153), successive refinement (155), CFAR (156) and Braess (158) are each
   supported only by citing works or secondary records; the Braess `Springer 2023`
   reference named in the plan was not identified at all. These items must say
   "the model of" and never "as measured by".
7. **The biological set cannot carry a mechanism claim.** Eight rows, all
   secondary (161–168), one of them (microglial pruning) weaker than the plan's
   wording: neither fetched extract attributes pruning to microglia. P16.32 is
   `HYPOTHESIS` and its text now says so.
8. **DSFB exists and is not what P16.2 needs it to be.** It is a deterministic,
   read-only residual diagnostics framework for sensor fusion with no ML and no
   neural-cache code (row 169). The ten motifs are SELVEDGE's, and the correct
   claim is "a DSFB-style observer", with the interface specified (P16.35) rather
   than assumed.
9. **Phenom exists and is a workspace OS, not an obligation computer.** Its README
   has no occurrence of "obligation" or "compartment" (row 170). WS17 is written
   as a *boundary proposal to* Phenom, and P17.2's compartment mapping is labelled
   a cross-project hypothesis, not a description.
10. **The most consequential `UNSUPPORTED` is the one the plan would have used to
    justify its measurement instrument.** No Springer 2026 LLM I/O bottleneck
    survey was located (row 152). The substitute is a primary measurement study
    (row 151) that says the measurement itself is fragile — which is a stronger
    reason to build P16.34's deterministic lab than the survey would have been.
11. **A 429 is not a verdict, and this pass marks it as such.** The Semantic
    Scholar rate limit and two Wikipedia 429s are recorded in the receipts; the
    affected rows were re-asked elsewhere or left `PARTIAL`/`UNSUPPORTED` with the
    queries named. No row in §14 rests on a response the script did not record.

## 16. Verification table — WS19 frontier-program works (fifth pass, 2026-11-16)

The proposed frontier program (`docs/PROGRAM-PLAN.md` §24) names 27 works. This
pass fetched them in four rounds, all recorded:
`results/prior-art/prior-art-fetch-20261116-round25-ws19.jsonl` (first pass),
`...-round26-ws19-arxiv.jsonl` (documented arXiv fallbacks),
`...-round27-ws19-names.jsonl` (name-shape probes) and
`...-round28-ws19-venues.jsonl` (venue probes).

Round 25 asked Semantic Scholar first; **21 of its 30 queries returned HTTP 429**
and are recorded as 429, not as absence. Rounds 26–28 re-asked each name against
arXiv's Atom API, which is the procedure §9 already prescribes.

**Counts for this pass.** 22 new rows below, plus 5 carried from earlier passes
(`Engram` row 4 `PARTIAL`, `Memory Grafting` row 59 `VERIFIED`, `MoSE` row 115,
`MoRE` row 116, `GMoE` row 117, both `VERIFIED`).

- `VERIFIED`: 17 new (rows 171, 172, 173, 177, 178, 179, 180, 183, 185, 186, 187,
  188, 189, 190, 191, 192, 193) — **three of these are verified in mechanism but
  not in venue**: 172 (NeurIPS 2025), 173 (NeurIPS 2025), 191 (ICLR 2026). The
  arXiv/ACL records do not state those venues.
- `PARTIAL`: 2 new (176, 184).
- `UNSUPPORTED`: 3 new (175, 181, 182).
- Merged identity: the plan lists `SCONE` and "Scaling Embedding Layers in
  Language Models" as two works; they are **one** (row 173).

| # | name in the plan | record | status | recorded evidence | what it actually reports, and the residual |
| ---: | --- | --- | --- | --- | --- |
| 171 | **Ouro / Scaling Latent Reasoning via Looped Language Models** | arXiv **2510.25741v6** ("Scaling Latent Reasoning via Looped Language Models") | **VERIFIED** | abstract: "We present and open-source **Ouro**, named after the recursive Ouroboros, a family of pre-trained **Looped Language Models (LoopLM)**" with "(i) iterative computation in latent space, (ii) an **entropy-regularized objective for learned depth allocation**, and (iii) scaling to **7.7T tokens**"; "Ouro **1.4B** and **2.6B** models". The family is independently named in row 179's neighbour (`Cross-Loop Query Handoff`, arXiv 2610.34639: "Ouro-1.4B and Ouro-2.6B") | a pretrained looped LM family with learned depth allocation. It adds **no token-addressed memory**; its loops reuse the model's own weights, and the analyses of it study loop-to-loop attention handoff and KV-cache error |
| 172 | **Retrofitted Recurrence (NeurIPS 2025)** | arXiv **2608.11233v1**, "Retrofitting Recurrent Depth into a Pretrained Language Model: Installation, Extrapolation, Transfer, and Retention at Two Parameter Budgets" | **VERIFIED** in mechanism, **PARTIAL** in venue | abstract: "Qwen2.5-0.5B-Instruct is split into a **Prelude**, a **weight-tied Recurrent Block**, and a **Coda**, with an **identity-preserving one-loop path** and a **re-entry bridge** on later loops. At loop 1 the retrofit remains **non-inferior to its base on a preregistered ARC battery**"; "installs at two budgets: **6M trained parameters over frozen base weights and 180M**". The fetched record does not state NeurIPS 2025 | the retrofit recipe and, critically, its **honest baseline**: a retrofit must first be *non-inferior at one loop*. P19.3 below is a reproduction of this recipe at our anchor, not an invention |
| 173 | **SCONE (NeurIPS 2025)** — and, separately in the plan, "**Scaling Embedding Layers in Language Models**" | arXiv **2502.01637v3**, "Scaling Embedding Layers in Language Models" | **VERIFIED** in mechanism, **PARTIAL** in venue; **one work, not two** | abstract: "We propose **SCONE** (**S**calable, **C**ontextualized, **O**ffloaded, **N**-gram **E**mbedding) ... retains the original vocabulary while introducing **embeddings for a set of frequent n-grams** ... learned with a separate model during training. After training, embeddings are **precomputed and stored in off-accelerator memory**; during inference, querying them has minimal impact on latency". NeurIPS 2025 is not stated in the fetched record and the Semantic Scholar venue query returned 429 | the exact mechanism the "grafted memory" stage proposes: n-gram-addressed embeddings stored off-device and read per token. The plan must cite **one** row for both names |
| 175 | **MoLE (ICML 2025)** | two distinct works, neither with the stated venue | **UNSUPPORTED** as named | arXiv `ti:"MoLE"` returns homonyms (moles the animal, *MOLE: Detecting Insider Threats in AI Agents*). The plausible mechanisms are separate papers: **Mixture of LoRA Experts** and **MoLE: Mixture of Latent Experts for Complementary Visual Reasoning**. An HTML grep of the **ICML 2025 PMLR v267 index** for either title matched **nothing**; the Semantic Scholar venue query returned 429 | there is no located "MoLE (ICML 2025)". If the plan means LoRA-expert mixtures, cite those by title; if it means layer-wise expert selection, say which paper |
| 176 | **N-gram Embedding / LongCat-Flash-Lite** | n-gram-embedding lineage verified; LongCat-Flash-Lite **not located as a paper** | **PARTIAL** | `ti:"N-gram Embedding"` returns *Low-Bit N-Gram Embeddings for Language Models*, *Modelling Intertextuality with N-gram Embeddings*, *Character n-gram Embeddings to Improve RNN Language Models*, plus SCONE (row 173). `all:"LongCat"` returns *LongCat-Flash Technical Report*, *LongCat-Next*, *LongCat-Video*, *LongCat-DeepResearch*; `all:"Flash-Lite"` returns no LongCat work | the n-gram-embedding claim is supported by several works; the **"LongCat-Flash-Lite" attribution is unsupported**. LongCat technical reports exist but none is retrieved as an n-gram-memory study |
| 177 | **Lngram** | arXiv **2605.24869v1**, "Lngram: N-gram Conditional Memory in Latent Space" (a v2, "Latent N-Gram Memory with Interpretable Discrete ...", also returned) | **VERIFIED** | abstract: "**Engram partially decouples retrieval from the backbone**, but its token-based keys remain tied to text tokenization and hash compression. We propose **Lngram**, a latent-space conditional memory module that **learns discrete symbols directly from hidden states** and performs N-gram lookup over these symbols. This design removes the dependence on tokenizer IDs" | a **token-addressed-in-latent-space** lookup memory, single pass. It is the closest *memory* prior art to our proposal and a direct competitor at the grafted-memory stage |
| 178 | **NGM** | arXiv **2605.16893v2**, "NGM: A Plug-and-Play Training-Free Memory Module for LLMs" | **VERIFIED** | abstract: "we propose **N-gram Memory (NGM)**, a **training-free, plug-and-play** module that **repurposes a backbone model's pretrained token embedding space as memory**. NGM consists of a **Causal N-Gram Encoder** and a **Cosine-Gated Memory**" | a training-free n-gram memory that needs no new parameters. It sets a hard baseline for P19.2: a trained table must beat this at equal union |
| 179 | **The Surprising Effectiveness of Shared Memory in Looped Transformers** | arXiv **2610.02383v1** | **VERIFIED** | abstract: "Each recursion ... writes its own key-value cache, so memory still grows with compute ... We pretrain looped language models to **share memory: only the first recursion writes a cache, and later recursions read it** while keeping a short window of their own. Surprisingly ... sharing memory **does not cost quality and instead improves it**. At **150M-1B parameters**, our Looped Prediction Transformer ..." | shared **KV cache** across passes — not a token-addressed lookup table. It is the nearest *structural* neighbour to "read one memory across passes" |
| 180 | **RecurTrace** | arXiv **2609.03379v3**, "RecurTrace: Adaptive Latent Reasoning with Loop-Time Memory" | **VERIFIED** | abstract: "Each iteration sees only the previous output and cannot directly access earlier computations ... **Loop Memory Attention**" lets the loop read its own **trajectory**; a **fixed loop count wastes depth** on easy inputs | memory over **loop time** (the trajectory), not over tokens. It is the second-nearest neighbour and the direct competitor for "per-pass memory" if the key is pass-indexed |
| 181 | **MeSH** | name search returns homonyms only | **UNSUPPORTED** | `ti:"MeSH"` returns mesh-geometry papers; `all:"MeSH" AND all:"expert"` returns *MME: Mixture of Mesh Experts*, *MESH: Memory-Efficient Sinkhorn Optimization for Mixture-of-...*, meshing algorithms. No shared-expert mechanism with that name | must not be cited. If the intended mechanism is shared experts, rows 115–117 and 193 cover the verified literature |
| 182 | **MeSA** | name search returns homonyms only | **UNSUPPORTED** | `ti:"MeSA"` returns stellar-evolution MESA models; `all:"MeSA" AND all:"mixture"` returns *MESA: Improving MoE Safety Alignment via Decentralized Expert ...*, *MESA: Effective Matching Redundancy Reduction ...* | must not be cited |
| 183 | **Repeated Shared Access Enables Grokking** | arXiv **2606.20737v2**, "Repeated Shared Access Enables Grokking, but Edit Propagation Depends on an Addressable Memory" | **VERIFIED** | abstract: "a **2x2 grid that crosses loop recurrence with shared-memory access**: a dense transformer (Dense), a looped transformer (Loop), a dense backbone with shared memory (Dense+Mem), and a looped backbone with shared memory (**loop-memory coupling, LMC**). The two factors dissociate. For learning, **both routes to repeated shared access — looped recomputation and repeated memory rereading — cross the out-of-distribution (OOD) grokking barrier that Dense fails** ... Edit propagation depends on an addressable memory" | the **closest published factorial** to our P19.4 design, and the strongest threat to its novelty. Its "shared memory" is a shared **state/read** route; the paper does not test a **token-addressed learned lookup table read inside each pass**. Its second finding (edit propagation needs an *addressable* memory) is evidence *for* the direction, not prior art |
| 184 | **On-Policy Distillation (Thinking Machines)** | page `https://thinkingmachines.ai/blog/on-policy-distillation/` | **PARTIAL** | fetched: title "*On-Policy Distillation - Thinking Machines Lab*", meta description "*On-policy, dense supervision is a useful tool for distillation*". The page serves **no abstract and no body text** to this fetcher | the recipe's existence is verified as a titled page; **no number reported in it may be quoted** from this ledger. If the plan needs its quantitative claims, someone must read the page and record the numbers with the access date |
| 185 | **SMELT** | arXiv **2609.01343v3**, "SMELT: Scaling Laws for Compute-Matched MoE Looped Transformers" | **VERIFIED** | abstract: "most evaluations compare at fixed model size, **conflating architectural advantage with extra FLOPs** ... while closely **matching per-token FLOPs, total non-embedding parameters, and KV cache** ... **SMELT** (Sparse MoE Transformer, middle layers **Loop Twice**), which loops the **middle half** of layers twice ... scales across four sizes up to **54B** ... a separate Chinchilla-style scaling law for each architecture. SMELT's loss drops faster with compute, **saving 6.8–18.0% of training FLOPs on the compute-optimal frontier** ... largest on **Code** ... the second visit **reduces the attention sink**" | the compute-matched loop win is **already published, with a number and a scaling law**. P19.3 is a reproduction of SMELT's accounting discipline on our anchor; the *claim* stage must compare against it, and cannot present looped recurrence as new |
| 186 | **When Are Experts Misrouted? (counterfactual routing)** | arXiv **2605.07260v1** | **VERIFIED** | abstract: "compare each **standard route against sampled equal-compute alternatives** for the same token and score each by **the next-token probability it assigns to the realized token in a verified reasoning trajectory** ... the standard router is **well-aligned with route utility on confident tokens but uninformative on the fragile tokens** that drive hard reasoning" | the diagnostic exists. The plan's WS20 stop list removes utility-trained routers; this paper is why the *diagnostic* is kept and the *router* is not |
| 187 | **Path-Constrained MoE** | arXiv **2603.18297v2** | **VERIFIED** | abstract: views MoE "through the lens of **expert paths** — the sequence of expert selections a token makes across all layers"; despite N^L possible paths, "tokens in practice **cluster into a small fraction of paths**"; "the vast majority of paths remain unexplored" | constraining paths across layers. Registered as stopped work in WS20: its own framing is a statistical-efficiency argument, not a capability claim |
| 188 | **ReMoE** | arXiv **2412.14711v2** | **VERIFIED** | abstract: "**ReLU** as the router instead", "fully differentiable", drop-in replacement for TopK+Softmax, with methods to regulate sparsity and balance load | a routing mechanism. Not needed at the anchor's scale (dense), registered as stopped |
| 189 | **Self-Distillation Policy Optimization (SDPO)** | PMLR **v306 hubotter26a**, "Reinforcement Learning via Self-Distillation" | **VERIFIED** | abstract: "**Self-Distillation Policy Optimization (SDPO)**, which converts tokenized feedback into a **dense learning signal without any external teacher or explicit reward model**. SDPO treats **the current model conditioned on feedback as a self-teacher** and distills its feedback-informed next-token predictions back into the policy ... Across scientific reasoning, tool use, and competitive programming on **LiveCodeBench v6**" | a self-teacher recipe that needs **no teacher logits**. It is the cheapest baseline for the claim stage and must be run before any teacher-cache spend is called justified |
| 190 | **SPEC-Distill (EMNLP main 814)** | ACL Anthology **2026.emnlp-main.814**, "SPEC-Distill: Towards Efficient Speculative Distillation for Reinforcement Learning with Verifiable Rewards" | **VERIFIED** (the year probe is recorded: **2025**.emnlp-main.814 is a different paper, "Do Slides Help? ...") | abstract: "On-policy distillation (OPD) ... asks a larger teacher to score every student trajectory token by token, but this **inserts large-teacher forward passes into every RL step** ... **SPEC-Distill** ... runs it **once offline to build a reusable pool of expert trajectories**. During RL, the student treats these trajectories as **speculative drafts**, accepts a prefix under a lenience-controlled criterion, completes the response under its own policy, and is updated only by verifiable rewards with standard GRPO" | a **published alternative that deletes most teacher forward passes**. It is the direct competitor to a large teacher-logit cache and must be in the comparison set, not in a footnote |
| 191 | **T1 tool-integrated verification (ICLR 2026)** | arXiv **2504.04718v2**, "T1: Tool-integrated Verification for Test-time Compute Scaling in Small Language Models" | **VERIFIED** in mechanism, **PARTIAL** in venue | abstract: small LMs "struggle with verification tasks requiring **memorization**, such as numerical calculations and fact-checking" even with distillation from larger verifiers; T1 is the proposed remedy | the mechanism exists. The ICLR 2026 venue is **not** in the fetched record; do not print the venue as verified |
| 192 | **MoE-to-dense distillation** | arXiv **2605.28207v3**, "Pruning and Distilling Mixture-of-Experts into Dense Language Models" | **VERIFIED** | abstract: "**350 configurations**" on **Qwen3-30B-A3B**, "7 scoring, 5 grouping, and 2 magnitude scaling methods"; "the choice of **scoring method** is the most impactful"; "at matched parameter count, MoE-to-dense **outperforms dense-to-dense pruning by +6.3 pp** in average downstream accuracy after ..." (also DeepSeek-V2-Lite, GPT-OSS-20B) | the distillation route that competes with our "graft onto a dense anchor" route. It is a **baseline**, not a method to re-derive |
| 193 | **Slicing and Dicing** | arXiv **2605.11689v1**, "Slicing and Dicing: Configuring Optimal Mixtures of Experts" | **VERIFIED** | abstract: "**over 2,000 pretraining runs** spanning models up to **6.6B total parameters**"; "performance consistently improves with total MoE parameters **even at extreme active expert parameter ratios like 128**"; "the optimal **expert size is nearly invariant** to total parameter count and **depends only on active parameter count**"; "shared experts, heterogeneous experts and load-balancing settings have **small effects**" | the systematic configuration study. Its "small effects" finding is one more reason WS20 stops specialist-bank scaling |

## 17. Consequential findings from the fifth pass

1. **The loop win is published, with a number.** SMELT (row 185) already loops a
   shared block under matched per-token FLOPs, parameters and KV cache, at four
   sizes up to 54B, and reports **6.8–18.0% training-FLOP savings on the
   compute-optimal frontier**. Retrofitted Recurrence (row 172) already installs
   recurrence into a pretrained model at two budgets and reports **non-inferiority
   at one loop**. P19.3 is therefore a **reproduction of published recipes on our
   anchor**, and the frontier claim may not include "looped recurrence" as a
   novel mechanism. What remains claimable is the *combination* in finding 6.
2. **The closest published factorial to the proposed novelty is row 183, and it
   is close.** 2606.20737 crosses *loop recurrence* with *shared-memory access* in
   a 2×2 and reports that **both looped recomputation and repeated memory
   rereading** cross an OOD grokking barrier. Any plan text that says "nobody has
   combined looping with memory" is **stronger than the record**.
3. **The precise residual gap, stated narrowly.** In the fetched records, the
   works that use a **token-addressed learned lookup table** (Engram row 4,
   Memory Grafting row 59, SCONE row 173, Lngram row 177, NGM row 178) read it in
   a **single forward pass**, and the works that **reuse one memory across
   recurrent passes** (row 179: shared KV cache; row 180: loop-time attention;
   row 183: shared-memory rereading) do **not** use a token-addressed lookup
   table. The residual claim is therefore exactly: *a token-addressed conditional
   memory table read inside every recurrent pass of a weight-tied loop, with the
   read counted in the per-token unique-active union.* It is narrow, it is
   falsifiable in a 2×2, and it is **not** "loop plus memory" in general. §8's
   caveat still applies: this is a statement about published mechanism
   descriptions that were fetched, not proof of absence.
4. **Two "new" names in the plan are one work, and three are not works.**
   `SCONE` is the method inside "Scaling Embedding Layers in Language Models"
   (row 173) — one citation, two entries in the plan. `MoLE (ICML 2025)` (row
   175), `MeSH` and `MeSA` (rows 181–182) resolve only to homonyms, and the ICML
   2025 index does not contain a MoLE of either plausible kind. They must be
   removed from any argument that depends on them.
5. **LongCat-Flash-Lite is not a paper.** The LongCat technical reports exist;
   none was fetched as an n-gram-memory study (row 176). The n-gram-embedding
   claim survives on SCONE and Lngram, not on LongCat.
6. **Three works already delete the teacher compute the plan budgets for.**
   SPEC-Distill (row 190) moves the teacher **offline into a trajectory pool**;
   SDPO (row 189) removes the external teacher entirely by self-distilling on
   feedback; MoE-to-dense (row 192) distils a *different* architecture into a
   dense one at matched parameters with a measured **+6.3 pp**. Before any
   teacher-logit cache is funded, these three are the comparison set — and they
   are cheaper than the cache.
7. **The plan's own stop list is now evidence-backed.** Rows 186–188 and 193
   support stopping utility-trained routers (186: the router is uninformative
   exactly on the tokens that matter), path-constrained routing (187) and
   specialist-bank scaling (193: shared experts and heterogeneous sizing have
   small effects; expert size tracks *active* parameters only).
8. **The On-Policy Distillation page yields no numbers.** Row 184 is `PARTIAL`:
   the page was fetched and its title and description recorded, but no abstract or
   body was served, so **no number from it may be cited**. Anyone who needs its
   numbers must record them in a new receipt with the access date.
9. **A 429 is still not a verdict.** 21 of round 25's 30 Semantic Scholar queries
   were rate-limited; every affected name was re-asked through arXiv (rounds
   26–28) and the queries are in the receipts. The three `UNSUPPORTED` rows are
   supported by *name-shape* probes returning homonyms, not by a rate limit.
