> Tracked copy of the local research input `research/SELVEDGE-Implementation-Mandate.md` (which is gitignored). Research cutoff: 2026-11-14.

# SELVEDGE — Frontier Research and Implementation Mandate

**Status:** Research engineering specification, not a demonstrated new architecture  
**Research cutoff:** 2026-11-14  
**Target:** Rust-first, reproducible, single-workstation research program

## ROLE AND OBJECTIVE

Act as a principal research engineer in sparse neural networks, mechanistic attribution, storage systems, GPU inference, and Rust. Implement **SELVEDGE**, a research system for measuring and improving the amount of useful neural computation obtainable under a strict **1,000,000,000 unique active learned-parameter ceiling per output token**. The larger learned substrate may live on NVMe/host RAM; its accessible size is independent of the active ceiling. Never confuse total parameters, unique active parameters, GPU-resident parameters, bytes transferred, or FLOPs.

The scientific question is:

> Can a causal, interaction-aware, physically costed shard selector and execution scheduler improve the task-quality / active-parameter / wall-clock / physical-I/O Pareto frontier over the strongest existing sparse inference and offloading baselines?

The desired progression is: (1) discover whether high-fidelity sparse computation is possible, (2) prove a useful causal-selection advantage, (3) realize that advantage in a correct Rust runtime, and only then (4) investigate more ambitious unified choices among computation, memory, exact recovery, and recurrent depth. Do not build a huge system on an unproven assumption. Do not claim victory or novelty without reproducible evidence.

If there is an existing SELVEDGE repository, inspect the current source, tests, README, CI, dependencies, and available model artifacts first; preserve working components. Otherwise initialize a new Apache-2.0 Rust workspace. Keep any future WeftFS, WeftCodec, or specialized storage integration behind stable interfaces; do not make it a prerequisite for proving the core ML hypothesis.

## NONNEGOTIABLE SCIENTIFIC AND ENGINEERING CONSTRAINTS

1. **Hard parameter rule.** For output token `t`, compute `active_unique_params(t) = |union of identities of all learned parameters actually used to calculate that token|`. Count embeddings/output projection (account for tied matrices only once), attention, norms, routers, expert fragments, adapters, learned memory vectors read, and verifier/controller weights. Reused recurrent weights count once as unique parameters but every repeated operation is counted in FLOPs. Assert `active_unique_params(t) <= 1_000_000_000` in compliance mode. Also report the **number of parameter reads**, **FLOPs**, **maximum simultaneous GPU-resident learned weights**, and **total accessible learned parameters** separately. A teacher model or dense fallback that violates the cap must be labeled **noncompliant** and must never appear in compliant results.
2. **Truthful source assumptions.** GGUF/safetensors store checkpoint weights and metadata, not original training data or prompt-independent KV. Runtime KV is tied to the exact causal prefix and compatible model configuration. Do not claim to infer original training examples from weight inspection.
3. **Two independent modes.** `exact_execution`: original selected operations and weights are not changed by caching/prefetch; transfer prediction errors may alter latency, never model semantics. `approximate_execution`: selection/omission may alter outputs and must be separately evaluated and clearly labeled. Never conflate them.
4. **No imaginary speedups.** Mathematical transfer-time lower bounds are not benchmarks. Realize throughput claims with physical reads, CUDA events and synchronizations, cold/warm control, data checksums, and wall-clock measurement. Report peak memory and p50/p95/p99 latency. All comparisons must include controller overhead and misses.
5. **No fake causal oracle.** Calibration may use interventions and gradients. The deployed router may only see information actually available before it makes its decision. Never leak future layer activations, teacher logits, gold answers, future tokens, or evaluation labels into online selectors.
6. **No false novelty.** A technique is not novel because we renamed it or combined two acronyms. Maintain a dated prior-art table, explicitly contrasting SELVEDGE mechanisms with antecedents, overlapping features, and ablation evidence. Distinguish verified research from hypotheses.
7. **Hardware-aware and portable.** Probe available CPU, RAM, NVMe, driver, CUDA, GPU, and VRAM. Target a single consumer NVIDIA card such as an RTX 4080 with 16 GiB VRAM when available, but require CPU-executable tests. Cap physical memory in configuration; avoid silently offloading datasets to swap. CUDA support must be feature-gated.
8. **Research hygiene.** Pin source commit/revision, tokenizer, model license, dataset revision, seeds, compiler/toolkit, kernels, precision, prompts, decoding, and machine environment. Keep calibration, validation, and held-out tests disjoint, including near-duplicate/contamination checks. Do not use benchmark results to train the controller.
9. **Safety/correctness of artifact loading.** Validate offsets/lengths/shapes/strides/dtypes/checksums and bound all allocations; reject malformed or truncated shards, index corruption, unsupported GGUF quantization types, model digest mismatches, and stale KV. No silent mutation of upstream model artifacts.
10. **Honest completion.** Implement real runnable increments, not empty scaffold files and futuristic diagrams. If blocked, preserve what works and publish an explicit blocker with evidence. Never rewrite reported observations to make a failure look successful.

## REQUIRED LITERATURE FORENSICS — READ FIRST

Study primary sources and, where feasible, actual implementation code. Record each claim with a URL, access date, model, hardware, comparison conditions, and one reproducible test. At minimum:

- [Deja Vu: Contextual Sparsity for Efficient LLMs](https://proceedings.mlr.press/v202/liu23am.html); [PowerInfer](https://doi.org/10.1145/3694715.3695964): activation prediction and heterogeneous sparse inference.
- [Attribution-based Sparse Activation in Large Language Models (2026)](https://pmc.ncbi.nlm.nih.gov/articles/PMC13570667/): corrected gradient-attribution and group-dependence caveats. **Direct novelty threat.**
- [SEAP (AAAI 2026)](https://ojs.aaai.org/index.php/AAAI/article/view/40463); [SparseInfer (ICML 2026)](https://proceedings.mlr.press/v306/wang26hn.html): task-domain routing and semantic core-neuron prediction. **Direct novelty threat.**
- [SIMoE (ACL 2025)](https://aclanthology.org/2025.acl-long.816/); [UpIT (ACL 2025)](https://aclanthology.org/2025.acl-long.637/): dense-to-sparse upcycling with additional training.
- [PEER: Mixture of A Million Experts](https://arxiv.org/abs/2407.04153); [Monet (ICLR 2025)](https://proceedings.iclr.cc/paper_files/paper/2025/hash/382c35d1a57c07055dfcba58dd39e012-Abstract-Conference.html): fine-grained indexing and very many experts.
- [MoSE (ICML 2026)](https://proceedings.mlr.press/v306/tastan26a.html); [DAG-MoE (ICML 2026)](https://proceedings.mlr.press/v306/feng26v.html); [ProbMoE (ICML 2026)](https://proceedings.mlr.press/v306/zhao26ap.html): variable width, expert-output composition, and dynamic routing. **Do not treat these as unsolved simply because the implementation is novel to this repository.**
- [Memory Layers at Scale (ICML 2025)](https://proceedings.mlr.press/v267/berges25a.html): learnable memory lookup distinct from procedural computation.
- [MoTE: Mixture of Ternary Experts (2025)](https://arxiv.org/abs/2506.14435); [PuzzleMoE (ICML 2026)](https://proceedings.mlr.press/v306/zhao26ay.html): low-bit experts and redundancy-aware expert storage.
- [MoE-Infinity](https://github.com/EfficientMoE/MoE-Infinity), [SpecMD (2026)](https://arxiv.org/abs/2602.03921), [SpecPrefetch (2026)](https://arxiv.org/abs/2607.24787), [SeqMoE (2026)](https://arxiv.org/abs/2609.12978): SSD/RAM offloading, cache policy, predictive transfer, deadline scheduling. **Must compare real I/O and exact-routing execution.**
- [DoMoE (IJCAI 2026)](https://ijcai.org/proceedings/2026/657), [Task-Aware Coactivation Grouping (2026)](https://www.alphaxiv.org/abs/2606.01007): domain-level routing tables and task-conditioned placement. **Do not call a task-index or co-access graph new.**
- [Prefetch-Steered Routing (2026)](https://arxiv.org/abs/2610.34721); [MoE-CORE (2026)](https://arxiv.org/abs/2610.01950): quality/latency implications of changing routing and coordinated residency.
- [OLMoE 1B-7B](https://arxiv.org/abs/2409.02060): open sparse architecture for routing/benchmark research; its commonly cited active parameter count is roughly **1.3B**, not certified under our strict `<=1B` rule.
- [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B): a practical small open model candidate; inspect exact revision, architecture, actual unique parameter count, and license before selection.
- [Huginn recurrent depth](https://arxiv.org/abs/2502.05171) and [critical recurrent analysis](https://arxiv.org/abs/2507.02199): repeated inference is not automatically improved reasoning.
- [vLLM prefix-cache design](https://docs.vllm.ai/en/v0.13.0/design/prefix_caching/): identity and exact-prefix constraints for KV reuse.
- [GGUF specification](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md), [Candle Rust](https://github.com/huggingface/candle), [EleutherAI lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness): file layout, available inference primitives, and independent evaluation.

Add any closer 2026 work found during implementation. **Research deliverable:** `docs/PRIOR-ART-AND-NOVELTY.md` with a feature-overlap matrix and `VERIFIED / PARTIAL / UNSUPPORTED / PROPOSED` status for every claim. Our target claim is a *joint operating point and verified causal-cost advantage*, not "first to use causal attribution", "first to prefetch experts", or "first ternary MoE".

## HYPOTHESES AND EXPERIMENTS

### H0: Exact decomposition of the dense FFN, without learned routing

For SwiGLU (adapt math for each architecture's actual tensor orientation):

`FFN(x) = W_down [ SiLU(W_gate x) ⊙ (W_up x) ]`.

Partition the intermediate-channel indices into disjoint groups `G_1..G_m`. Then in ideal arithmetic:

`FFN(x) = Σ_g W_down[:, G_g] (SiLU(W_gate[G_g,:] x) ⊙ (W_up[G_g,:] x))`.

This identity is exact when all groups are included; numerical reductions may differ. It is **not** proof that skipping a group preserves model quality. Implement a deterministic CPU toy court that checks reconstruction across groups, different group sizes, mixed input shapes, and transposed storage. Then repeat with one real supported model and tolerance calibrated by dtype. Repack actual contiguous group tensors rather than relying on gather-heavy microfragments during production inference.

**Failure gate:** If tensor conversion/reconstruction is not correct, do not proceed to learned sparsification.

### H1: Causal group utility is more useful than activation frequency

For state/context `h`, selected group set `S`, candidate group `g`, and a measured quality loss `L`, estimate conditional marginal gain:

`Δ(g | S, h) = L(S, h) - L(S ∪ {g}, h)`.

Here `L` can include teacher divergence and task-specific correctness. No single universal measure is adequate; log each. Measure **pairwise non-additivity**:

`I(g,j | S,h) = Δ(g | S ∪ {j},h) - Δ(g | S,h)`.

`I` may be positive or negative. The selection objective is generally **not submodular**; never invoke greedy guarantees without proving assumptions. Use finite interventions only where computationally feasible, stratified pair sampling, uncertainty intervals, and a low-rank/graph surrogate. Train the deployment selector using **pre-decision** representations only. Do not materialize the exponential powerset of groups.

**Mandatory selectors:** random, activation-magnitude, frequency, gradient×activation, interaction-corrected attribution from the 2026 work if reproducible, task-static, learned one-step, learned interaction-aware, and an expensive offline oracle (explicitly labeled non-deployable).

**Gate:** At matched executed unique parameter counts, demonstrate a replicated held-out improvement over the strongest implementable baseline (e.g. teacher KL and accuracy) with confidence intervals. If the improvement disappears under distribution shift or predictor overhead, reject the claim.

### H2: Joint causal-cost selection is better than independent route + cache

Define empirical cost `C(g | S, device_state)` using measured **marginal exposed stall**, bytes fetched, deadline, residency, alignment, transfer overlap, and co-fetched neighbors. Do not approximate cost as `bytes / advertised_bandwidth` except for lower-bound sanity checks.

Investigate selection of `S` under a global unique-parameter cap and latency/memory constraints:

`maximize estimated_quality_gain(S,h) - λ_io * exposed_io_cost(S) - λ_f * flops(S)`

subject to `active_unique_params(S) <= 1e9`, `peak_vram <= configured_limit`, and an explicit tail-latency objective. Account for parameter sharing and non-additive cache costs. Use a small exact offline integer program or bounded enumerator as an oracle on toy cases; compare fast online heuristics and learned dual prices. Never quietly spend the parameter budget twice in different layers.

**Mandatory comparison:** (a) causal routing + LRU, (b) ordinary routing + learned prefetch, (c) independent causal routing and prefetch, (d) joint selector and scheduler, (e) unmodified exact-routing with optimized cache. The joint policy wins only if it improves the measured end-to-end frontier.

### H3: Execution envelopes and causal neighborhoods can reduce physical traffic

Measure phase persistence in inference traces: does a short prefix or task-phase state predict a stable **superset** of needed fragments, and for how long? Build invalidatable request/phase "execution envelopes" holding likely shard IDs, confidence, deadlines, and expected live ranges. They are **advisory prefetch predictions** in exact mode, not a license to change the native router.

Use an interaction-weighted/trace-weighted graph or hypergraph to propose physically contiguous placements of shards frequently needed together. Compare with existing task-aware placement and predicted-cache techniques, not just LRU. Include storage amplification and unused-prefetch bytes. Let the algorithm decide between single shards, compact bundles, and replicated small hot fragments; evaluate re-layout and cache pollution costs.

**Gate:** Show net p95 token-latency or physical-I/O improvement on replayed and live traces, with no exact-mode output change. Avoid "95% prefetch hit rate" as a substitute for throughput.

### H4: The 1B ceiling can be used more effectively with modular learned capability

Only after H0–H3 are characterized, compare (i) expert-only external capacity, (ii) a learned key/value memory layer, (iii) a few trained specialist LoRA/residual overlays, and (iv) combinations, all matched for active parameters, FLOPs, VRAM, storage footprint, and token budget. Source retrieval is **nonparametric**, separately metered in bytes/latency; learned keys/values and adapters count as accessed learned parameters. Do not pretend that inserting untrained weight shards gives a model new abilities. Train or distill modules with explicit matched data/compute and independent tests.

**Gate:** Demonstrate meaningful specialization, held-out transfer, and no unacceptable interference or quality collapse.

### H5: Recurrent depth and exact memory recovery can be costed as alternative actions

Research-only after earlier gates. Additional latent loops, specialist execution, exact source lookup, and KV reconstruction are alternative costed actions. Train a small value-of-computation estimator from counterfactual outcomes. Compare against fixed depth, confidence-based halting, and independent policies. More recurrence has **no guaranteed accuracy benefit**. Exact source restoration is not the same as lossless KV compression; preserve data or recompute with compatible model configuration. Do not deploy a learned controller without regret and calibration tests.

## IMPLEMENTATION PLAN AND REQUIRED SOURCE TREE

Use a Rust Cargo workspace. Allow a **narrow Python research oracle** for Hugging Face/PyTorch activation instrumentation and fine-tuning; the production file format, scheduler, simulator, accounting, and CLI must remain Rust. Do not waste the project reimplementing an entire Transformer or tokenizer before the central hypothesis is tested. Evaluate Candle for the minimal Rust backend and compare with available existing inference engines; record reasons if another backend is chosen.

Suggested layout (adapt to an existing repository rather than mechanically creating everything):

```text
selvedge/
  Cargo.toml
  crates/
    selvedge-core/        # deterministic IDs, budgets, errors, config, counters
    selvedge-format/      # versioned shard manifests, validator, checksums
    selvedge-convert/     # tensor-aligned FFN decomposition and repacking
    selvedge-atlas/       # causal utility observations, graph, selector interface
    selvedge-store/       # NVMe/host/GPU tier contracts, cache, async transfers
    selvedge-sim/         # trace-driven execution and deadline simulation
    selvedge-runtime/     # exact and approximate execution paths
    selvedge-bench/       # workload replay, hardware telemetry, ablations
    selvedge-cli/         # `selvedge` end-user commands
  oracle/                 # pinned Python teacher probes and training utilities
  tests/                  # integration, malformed artifact, crash courts
  fixtures/               # tiny generated non-proprietary test weights
  configs/                # CPU, single GPU, 1B compliance
  docs/                   # scientific receipts and architecture
  results/                # structured outputs; never check in giant artifacts
```

Start with working implementations of `core`, `format`, `convert`, `atlas`, `sim`, and `cli`; create others only when needed to run a real experiment. Design concrete interfaces for `ShardId`, `ModelDigest`, `TensorSlice`, `ParameterLedger`, `SelectionDecision`, `ResidencySnapshot`, `TransferRequest`, `TransferCompletion`, `ExecutionTrace`, `QualityObservation`, `CostModel` and `Backend`. Explicitly version data contracts. Use `Result`, no silent panics in library/public input paths; avoid unsafe outside isolated, justified modules.

### FORMAT / STORAGE CONTRACT

Versioned manifest includes: source model revision and cryptographic digest; tokenizer digest; operation/tensor and layer identities; tensor orientation and shapes; group channel lists; dtype/quantization type, block scales, strides, repacking transform, quantization calibration; shard offset and length; checksum; compatibility identifiers; dependency/co-access links; training/calibration provenance. Validate before use. A chosen default alignment must be benchmarked, not assumed optimal (4 KiB is a baseline, not a prescription for every shard). In exact mode, every required group must be retrievable even after a prefetch miss. Keep durable storage immutable and commit manifests atomically.

Provide both in-memory toy and on-disk implementations. Make data movement observable: actual bytes requested, completed, used, wasted, page-fault counts where supported, queue and transfer timing, backpressure, cache eviction, GPU transfer time, and exposed stall. Never attribute all async bytes to fully overlapped execution without verifying event completion.

### ACCOUNTING CONTRACT

Track learned parameter identity at tensor-slice granularity; detect shared/tied weights, fragments that alias the same underlying parameter, and repetition in recurrence. Emit per-token `active_unique_params`, `resident_unique_params`, total accessible, bytes read from each tier, quant/dequant work, FLOPs, and routing/controller FLOPs. Enforce the 1B active ceiling *before* executing a compliant token. The runtime may reject an infeasible decision or choose a **within-budget** fallback; never execute over-budget and then conceal it. Preserve an optional explicit `research_noncompliant_teacher` mode for oracle measurements.

### ORACLE / MODEL PATH

1. Implement deterministic synthetic SwiGLU and tiny transformer fixtures to prove exact conversion and controlled causal experiments.
2. Support a suitable open, permissively licensed, small safetensors model for real activation hooks (candidate: Qwen3-0.6B); verify architecture and checkpoint before coding against it. Keep Hugging Face execution as the teacher oracle initially.
3. Probe a fully open sparse model such as OLMoE-1B-7B as a separate reference for expert traces, **not** as proof of strict <=1B compliance.
4. Add GGUF **inspection/validation** after safetensors path works. Support only named/tested versions and tensor types. Correctly handle quantization block boundaries; never slice packed GGUF payload bytes as though they were unquantized floats. Train/attribute from the high-fidelity tensor source when possible.
5. Record the residual backbone's unavoidable parameter count before promising a compliant dense-to-sparse conversion. If always-executed weights alone exceed 1B, that conversion is impossible under our rules without architecture compression or distillation.

## TEST AND VERIFICATION COURTS

**Court A — Algebra and tensor reconstruction.** Exact full-group sum vs original FFN in F32 within stated numerical tolerances; group permutations; zero/full subsets; unusual shapes and tensor axis conventions; gradient comparisons in the teacher oracle. Do not claim bitwise equality when reduction order or precision changes.

**Court B — Attribution counterfactuals.** Small exhaustive group-subset oracle; correct signs and interactions; synthetic synergy and antagonism cases; uncertainty estimation; predictor sees no future state; contaminated prompt detection; OOD held-out tasks; independent seeds.

**Court C — Budget accounting.** Tied weights, shared embeddings, overlapping groups, adapters, learned memory, recurrent loops, quantization variants, model switching, attempted over-budget decisions. Prove the cap for every executed output token in compliance mode.

**Court D — Storage integrity and crash behavior.** Corrupt hashes, malformed lengths, endianness/unsupported tensor type, truncated bundles, interrupted atomic publication, conflicting versions, stale index, cancelled I/O, cache invalidation, concurrent readers, hot reload rejection. Use property-based tests and fuzz targets on parser/manifest boundary.

**Court E — Exact-mode semantic parity.** Same model, same tokens, deterministic decoding and precision, different cache capacities, prefetch predictors, transfer delays, eviction policies and scheduling interleavings. Exact-mode outputs must be numerically equivalent within established backend tolerance; caching must not change selected operators. If routing itself changes, reclassify as approximate mode.

**Court F — Real latency and performance.** Cold and warm; batch-1 then small batch; short and long prompts; repeated task, alternating task, adversarial rapid task-switch; memory pressure; synthetic and recorded I/O traces; pageable vs pinned buffers when available. Report TTFT, decode TPS, p50/p95/p99 inter-token latency, GPU memory, RAM, power if measurable, read bytes/op, prefetch waste, cache miss cost, controller overhead. Use paired interleaved A/B schedules and confidence intervals, not a single favorable run.

**Court G — Quality.** Perplexity / teacher KL on held-out texts, plus task benchmarks selected before tuning (e.g., code, math, general QA, factual recall, long context). Include both task-conditioned and task-mixed workloads; report per-domain regressions, not only pooled average. Use independent harness integration where possible. Track evaluation dataset licenses and contamination risks.

**Court H — Research integrity.** Each benchmark row must carry git SHA, config hash, source model revision, dataset revision, sample count, hardware, driver, precision, warm/cold state, RNG seed, timestamp, and exact command. Missing evidence means "unverified", not "pass". All benchmark artifacts should have an explicit schema.

## MILESTONES, IN ORDER, WITH HARD GATES

**M0 — Research and executable specification:** inspect repo/hardware; produce literature overlap matrix, measurable hypotheses, baseline inventory, budget math, minimal model decision, commands to reproduce, and synthetic fixtures. Deliver a working `selvedge doctor`, `selvedge format validate`, and CPU test court. No empty architecture theater.

**M1 — Exact FFN decomposition + offline sparsity oracle:** instrument a teacher, export/reload group shards, prove full-reconstruction parity, compute teacher divergence when subsets are skipped, map the achievable quality-vs-parameter curve, and identify backbone minimum. If the teacher cannot remain acceptable under 1B, report this and pursue a smaller backbone or retraining branch—do not fake success with caching tricks.

**M2 — Attribution selectors:** implement baseline selectors and intervention-labeled interaction-aware predictor, including an offline optimal/near-optimal selector on small problems. Reproduce matched-budget results with held-out and OOD evaluation. Stop if the method lacks a meaningful, reproducible quality advantage.

**M3 — Trace simulator and physical shard engine:** consume exact execution traces, compare LRU/LFU/Least-Stale-style policies where reproducible, deadline-aware prefetch, request/phase envelopes, and cost-aware placement. Validate cost-model prediction against a real NVMe and GPU transfer experiment. A simulation-only speedup is insufficient.

**M4 — Single-GPU integration:** implement at least one real end-to-end inference configuration with actual selected fragments, actual transfers, and audited accounting. Include native/unmodified inference and established offloading runtimes as comparisons when reproducible. Exact-mode parity and measured physical cost are release gates.

**M5 — Joint router/scheduler:** implement the interaction-aware causal benefit and exposed I/O cost objective, with budgeted online optimization. Compare independently tuned variants at fixed quality, memory, and compute. Publish Pareto curves and confidence intervals; no automatic "frontier beaten" claim.

**M6 — Optional extensions:** learned memory versus specialist experts; verified adapters; exact-recovery memory; adaptive recurrent work. One new mechanism per ablation. Require a fresh scientific hypothesis, baseline, kill criterion, and model/data cost budget before enabling each.

## ABLATION MATRIX — THE CORE SCIENTIFIC PRODUCT

Execute where applicable and budget permits:

1. Dense unmodified teacher / small native base model.
2. Random and magnitude-sparse; activation-frequency and task-static sparse.
3. Gradient×activation and published interaction-corrected attribution baseline.
4. Causal group predictor without pairwise interactions.
5. Causal interaction-aware predictor without hardware costs.
6. Native routing + LRU; native routing + best reproducible predictive prefetch.
7. Causal predictor + independent LRU/prefetch.
8. Interaction-aware causal predictor + joint exposed-I/O optimization.
9. (8) minus envelope; (8) minus co-placement; (8) minus uncertainty calibration; (8) with different shard granularity; (8) with cold vs hot tiers.
10. Exact native semantics vs approximate constrained execution; never pool their quality curves.
11. Equal active params; equal peak VRAM; equal total FLOPs; equal output quality; equal latency — each is a different comparison, report each explicitly.

Produce Pareto plots, raw machine-readable logs, sample-level paired deltas, confidence intervals, and effect sizes. If a proposed component fails, keep the negative result in the report; do not delete it from the storyline.

## GENUINELY RISKY COMBINATIONS TO INVESTIGATE (NOT CLAIM AS PRIOR-ART-FREE)

**1. Conditional causal-cost hypergraph.** Edges encode *intervention-measured complementarity or interference*, not just coincident activation; node/edge weights incorporate measured marginal physical I/O and future deadlines. Test whether learning both kinds of structure beats causal-only and placement-only baselines. Check related 2026 attribution, DAG-MoE and task-aware coactivation studies explicitly.

**2. Budget shadow-price controller.** Approximate a global (across-layer) mixed decision problem offline and distill its resource dual prices into an online controller. The policy allocates the 1B budget across layers based on measured counterfactual regret instead of fixed per-layer quotas. Compare against static layer budgets, MoSE-like dynamic width, and exact small-case optimizers.

**3. Two-view Atlas.** Keep independent heads for `value_if_executed` and `cost_to_materialize_now`. The former is trained on interventions; the latter is trained on real scheduler traces. Permit uncertainty-aware decisions, recalibration, and fallback within the cap. Prevent accidental training shortcuts from cache state into the value model.

**4. Causal execution envelopes.** A task-phase plan stores a conservative candidate superset, optional high-interaction neighborhoods, and invalidation triggers. Its utility must be measured by **bytes not transferred and exposed time saved**, not by merely predicting the top task label. Exact mode does not change actual route.

**5. Residual/patch shard representation.** Explore shared base expert fragments plus small specialized deltas as an alternative to independent cold expert blobs. Compare against PuzzleMoE/low-bit quantization and account for decoding, base dependency, and extra random reads. Do not claim it is an original compression primitive.

**6. Value-of-computation across representations.** Once a real expert execution path works, compare spending the remaining budget on an expert, learned memory lookup, exact source retrieval, or more recurrent computation. This is a research extension requiring separately verified costs and quality; do not start here.

All six require pre-registered expected value, a plausible nearest prior-art method, an experimental control, and a stop condition before implementation. Search the latest relevant literature again when advancing a milestone.

## DEVELOPMENT STANDARDS

- Rust stable, edition chosen from actual toolchain; `cargo fmt --all --check`, `cargo clippy --workspace --all-targets -- -D warnings`, `cargo test --workspace` on CPU; CUDA tests behind explicit features and detect missing hardware cleanly.
- Prefer explicit state machines and cancellation-safe transfer ownership. No `unwrap` in file/benchmark public paths. Ensure backend and index lifetimes cannot create use-after-free or stale borrowed GPU buffers. Clear ownership of pinned staging buffers and completion events.
- Record dependencies in `Cargo.lock`; pin Python environment; audit licenses and model-card constraints. Separate optional model downloads from normal CI. Synthetic CPU tests must work without network.
- Do not write custom CUDA kernels until profiling identifies an actual bottleneck and an existing implementation is insufficient. Provide reference CPU and tested CUDA parity when optimized kernels arrive.
- Use typed CLI configs with `selvedge doctor`, `selvedge inspect`, `selvedge convert`, `selvedge court`, `selvedge trace`, `selvedge simulate`, `selvedge bench`, `selvedge report`; document actual implemented commands, avoid fictional examples.
- Structured JSONL/Parquet metrics with documented schema; machine-readable baseline-vs-experiment report and human-readable summary. Persist failed runs, timeout/cancellation reasons, and OOMs.
- Keep README concise and truthful: current result, demonstration command, known losses, next falsifiable test. No inflated adjectives and no benchmark headline detached from conditions.
- Document threat model for untrusted model artifacts; avoid automatic executable remote code on model import. If an upstream model requires `trust_remote_code`, isolate the oracle and state the risk explicitly.

## REQUIRED DOCUMENTS AND DEMO

Maintain:

- `README.md`: concrete current capabilities and a truly runnable quickstart.
- `docs/ARCHITECTURE.md`: boundaries, dataflow, invariants, exact/approx distinction, diagrams.
- `docs/PRIOR-ART-AND-NOVELTY.md`: live literature ledger and closest-overlap matrix.
- `docs/EXPERIMENT-CONTRACT.md`: hypotheses, metrics, leakage prevention, statistical method, kill conditions.
- `docs/SHARD-FORMAT.md`: versioned layout, compatibility IDs, integrity rules.
- `docs/REPRODUCE.md`: exact commands, offline toy route and optional real-model route.
- `docs/LIMITATIONS.md`: incomplete parts, losses, known hazards, external validity.
- `results/REPORT.md` and `results/results.json`: measured outcomes, including failed comparisons and negative findings.

One end-to-end, offline, no-download demo must: generate tiny synthetic weights; partition and serialize groups; validate and reconstruct the original result; produce held-out causal observations; train or fit at least a simple selector; replay candidate requests through the scheduler; enforce the parameter cap; and generate a reproducible report. The real-model demo is a separate, optional-download path, but must eventually process actual model activations and perform real inference before claiming a deployable neural runtime.

## AGENT OPERATING PROTOCOL

1. Begin by inspecting repository, installed tools, hardware, and verified baselines. Report a short initial plan in `docs/WORK-LOG.md`, then **execute immediately**.
2. Work in coherent increments. After each increment, run and record the relevant tests; include commands and hashes. Prefer a small complete result to half-implemented features.
3. Keep a live table of `hypothesis -> baseline -> intervention -> result -> interpretation -> next decision`.
4. Correct false premises, even if they appear in this mandate. Never let speculative architecture outrank experimental evidence.
5. Never hide unfavorable evidence or invent literature findings, benchmark numbers, source paths, files, or prior test runs.
6. Do not ask for decisions already resolved here. Make sensible defaults and explain them. If external resources are missing, deliver a fully working CPU court and mark the real-model step as blocked, rather than claiming it succeeded.
7. End each engineering session with implemented changes, commands executed, passing/failing tests, measured results, current limits, and the single next experiment with the highest expected information gain.

## SUCCESS CONDITION

SELVEDGE is successful **only if** a reproducible experiment demonstrates a real Pareto improvement over credible existing baselines. The first publishable win could be either (a) demonstrably better quality at the same active parameter budget, (b) demonstrably lower cold/warm physical I/O and decode latency at identical output semantics, or (c) a joint improvement that neither mechanism achieves alone. Do not require all three to accept a valuable partial result. Do not claim to outperform frontier intelligence without independently verified matched-quality and matched-compute comparisons.

The guiding principle is simple:

> Store the large learned substrate wherever it is economical, but execute only the computation whose *measured marginal contribution* justifies its *measured total cost*. Prove each benefit with counterfactuals and real hardware before scaling the design.
