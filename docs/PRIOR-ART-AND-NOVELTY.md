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

## 2. Verification table — novelty-boundary list

| # | work | id / venue | status | recorded evidence | overlap with SELVEDGE |
| ---: | --- | --- | --- | --- | --- |
| 1 | Drop-Upcycling | arXiv 2502.19261 | VERIFIED | abstracts the upcycling slowdown and proposes "partial re-initialization" | dense→sparse upcycling, the P1.3 boundary |
| 2 | SIMoE (Sparse Interpolated MoE) | ACL 2025 long 816 | VERIFIED | "experts are structurally sparse subsets of the seed LLM's parameters"; router learns an input-dependent merging | expert upcycling + conditional merging |
| 3 | Memory Layers at Scale | ICML 2025 (PMLR v267) | VERIFIED | "trainable key-value lookup mechanism to add extra parameters ... without increasing FLOPs" | learned memory path (P3.1) |
| 4 | Engram | DeepSeek (primary not located) | PARTIAL | four 2026 works describe it: "Deepseek's Engram, a conditional memory module ... token-level N-gram hashing" (arXiv 2607.29065); "Engram conditional memory ... decoupling static knowledge lookup from dynamic computation" (arXiv 2603.10087) | hashed conditional memory (P3.1) |
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
