# PROGRAM PLAN — next phase, exhaustive and auditable

Status: **plan of record, pre-registration not yet written.** This document is
the complete TODO for the next program phase. Every item is numbered, and no
item from the strategy review is omitted. An item is only "done" when the
acceptance evidence named in it exists in the repository; a prose claim is not
evidence.

**Third extension, 2026-11-16 (S16C).** The plan is now **tiered** (§0.1),
an ordering inversion in the §16 queue is corrected and recorded (§0.2), and two
new workstreams are appended: **WS13** (published mechanisms to reproduce and
sixteen inventions, 21 items) and **WS14** (runtime and evaluation corrections,
7 items). No existing P-ID is renumbered, retitled or removed.

Companion documents: `docs/MANDATE.md` (requirements),
`docs/EXPERIMENT-CONTRACT.md` (registered thresholds; T1 is the current
section), `docs/PRIOR-ART-AND-NOVELTY.md` (what is actually prior art),
`docs/LIMITATIONS.md` (what the measured results do not establish),
`docs/TRAINING.md` (the training instrument as it exists).

## 0. How to read this plan

**Fields.** Each item carries:

- **What** — the deliverable, in checkable terms.
- **Why** — the reason it exists; what fails if it is skipped.
- **Depends** — item IDs that must be done first, or "none".
- **Acceptance evidence** — the exact receipt, court, test or document that
  makes the item done. Where an item changes a measurement, the receipt must
  name revision, command, environment and fixture.
- **Status** — one of `done`, `partial`, `open`, `blocked`, `deferred`.
- **Effort** — engineering sessions (S/M/L/XL) and, where a run is required,
  measured-resource estimate. One session is a focused working session on this
  workstation. Estimates are honest guesses and are not evidence.

**Status at the time of writing (revision at HEAD of `staging`):**

| area | state |
| --- | --- |
| M1 static whole-group omission | measured, **negative** (`results/M1-FRONTIER.md`) |
| M2 per-token post-hoc selection | measured, **negative**; oracle bound outside the usable tier (`results/M2-SELECTORS.md`, `docs/M2-CORRECTION.md`) |
| T1 trained/upcycled arms | pre-registered; infrastructure built; **no training run** |
| Rust crates | `core`, `format`, `convert`, `fixtures`, `atlas`, `store`, `cli` exist; `sim`, `runtime`, `bench` do not |
| Storage baselines | measured cold/warm NVMe and host-RAM baselines (`docs/STORAGE.md`) |
| Memory path, error-directed router, prefetch predictor | **do not exist** |
| Prior-art ledger | exists (`docs/PRIOR-ART-AND-NOVELTY.md`), created in this phase from recorded fetches |

**Number rule.** Measured performance, storage, latency, resource and
verification-count numbers are never written here by hand. Where a measured
quantity matters, this plan names the generated block in `docs/PERF-NUMBERS.md`
or the receipt under `results/`. Derived design counts (parameter arithmetic
over the registered architecture constants) are marked as arithmetic, are
exact integers, and carry a reproduce command; they are not measurements.

**Extension, 2026-11-15.** Sections 10–17 were appended (WS7–WS12, the
dependency-ordered execution queue and an updated NOW) without renumbering or
deleting any existing item. WS1–WS6 and their IDs are unchanged. WS7–WS12 are
written against the state of the T2 pilot at that time (running: `D` and `M`
evaluated, `X` evaluating, `H` and `H*` pending) and against the extended
prior-art ledger; every mechanism item whose novelty is contested carries the
ledger entry that contests it. Section 7's sequencing diagram remains the order
for WS1–WS6; section 16 supersedes it only for the new items.

---

## 0.0 Critical paths (2026-11-16, S20)

The program has **two independent critical paths**. They share instruments but not
blockers, and conflating them has repeatedly made the physical-execution work
look like a gate on the capability work (it is not).

**(a) Neural capability path — the decisive path now.**
1. Finish **H\*** (running) and record its verdict.
2. Certify the historical **X** and **H** checkpoints by provenance amendment
   (`results/t2/PROVENANCE-AMENDMENT-X-and-H.json`), or leave them unscoreable
   with the reason recorded.
3. Run the fresh Rust suite through the ordinary `cargo test` suite and the
   3-item trio that the capability court already invokes.
This path needs the **PyTorch adapter**, not the Rust runtime.

**(b) Physical execution path — real but not on the capability critical path.**
1. Runtime optimisation (P14.1–P14.3: decoded residency, async prefetch,
   batched prefill).
2. Integration of trained modules into the runtime.
3. Real transfer and latency measurement on the executed path.

**Dependency rule.** **P14.1–P14.3 do not gate the T2 capability run.** The T2
capability court executes through the PyTorch adapter; WS14 changes what the
*Rust runtime* measures (bytes/token, cold-byte p95, latency), not whether a
capability verdict can be produced. Any text hat implies otherwise is corrected
where it appears. The two paths converge only when a *runtime* claim is made. The
capability path may report before any WS14 item lands, and must say which
runtime revision it measured on.

---

## 0.1 Tier structure and status legend (2026-11-16)

A tier states **how much the program is committed to an item and how much
evidence it has**, not how interesting it is. Tiers are orthogonal to the
`Status` field: an item can be `open` in any tier.

| tier | meaning | what an item in this tier must carry |
| --- | --- | --- |
| **ACTIVE** | the current decisive path plus the instruments and constraints that bind it now | a scheduled next action; acceptance evidence that exists or is being produced in this phase; for an experiment, a registered contract |
| **CANDIDATE** | rigorously defined, **not yet justified** | a cheap decisive experiment, an explicit falsifying control, and a kill criterion that says what result retires it |
| **HYPOTHESIS** | retained for the long horizon, **no implementation commitment** | a statement of the question and why the program keeps it; no schedule, no budget, no claim |

Rules:

1. An item moves **up** a tier only when the acceptance evidence of the tier
   below exists. It moves **down** when its kill criterion fires; the negative
   result is kept (P6.5), never deleted.
2. A tier is revisable at a checkpoint and must be revised then. It is a record
   of commitment, not a verdict on the item's merit.
3. Tiers do **not** renumber anything. WS1–WS12 keep their IDs and text; WS13 and
   WS14 continue the numbering with P13.x and P14.x.
4. The tier index below is the authoritative statement. The table lists ranges
   only where the range is contiguous and every item inside it has the same tier.

### Tier index — 112 existing items (WS1–WS12)

| tier | items by workstream section | count |
| --- | --- | ---: |
| **ACTIVE** | §1 (WS1): P1.1–P1.4 · §2 (WS2): P2.1, P2.3–P2.4 · §3 (WS3): P3.4 · §6 (WS6): P6.1–P6.5 · §10 (WS7): P7.1, P7.3, P7.5, P7.7 · §15 (WS12): P12.1, P12.1a, P12.2, P12.9–P12.11, P12.15–P12.17 | 26 |
| **CANDIDATE** | §2: P2.2 · §3: P3.1–P3.3, P3.5–P3.7 · §4 (WS4): P4.1–P4.5 · §5 (WS5): P5.1–P5.10 · §10 (WS7): P7.2, P7.4, P7.6 · §11 (WS8): P8.1–P8.5 · §12 (WS9): P9.1–P9.7 · §13 (WS10): P10.1–P10.13, P10.15, P10.17–P10.18 · §14 (WS11): P11.1–P11.21 · §15 (WS12): P12.3–P12.6 | 78 |
| **HYPOTHESIS** | §5: P5.11 · §13: P10.14, P10.16 · §15: P12.7–P12.8, P12.12–P12.14 | 8 |

Tier index — new items: WS13 (§18) has **21** items — P13.1–P13.4 and
P13.6–P13.21 `CANDIDATE` (20), P13.5 `HYPOTHESIS` (1). WS14 (§19) has **7**
items — P14.1–P14.3 `ACTIVE` as runtime-optimisation work (they **do not** gate
the T2 capability run; see §0.0) and P14.4–P14.7 `CANDIDATE`.

**Assignments that need a reason, not just a row.**

- **P12.10 and P12.11 are ACTIVE although their content is "infrastructure".**
  They are *decision criteria* — the row set every comparison is reported against
  and the thresholds a frontier run is scored against. §0.2 records why they may
  not be scheduled after the experiments.
- **P3.4 is ACTIVE** because it is running as the registered T2 pilot (P12.1);
  the five-arm matrix is the T2 matrix, not a second experiment.
- **P12.3–P12.6 (T3–T6) are CANDIDATE, not HYPOTHESIS**, even though they are the
  next courts: each has a registration and a kill condition, but none may start
  before the pilot's verdict (P12.1) and the two blockers (P12.15).
- **P12.7, P12.8, P12.12–P12.14 are HYPOTHESIS**: they commit the program to a
  *shape* of claim (a joint court, three levels, a transfer test) with no
  implementation commitment in this phase. P12.11's thresholds still bind them.
- **P5.11, P10.14, P10.16 are HYPOTHESIS** because each is explicitly deferred in
  its own text (recurrent latent computation; shortcut-connected specialists;
  learned reasoning length).
- **P1.x, P6.x and four of the seven WS7 doctrine items are ACTIVE** as standing
  constraints: they bind how the current experiments are registered, measured and
  reported, which is what makes them active rather than background.


### The three tiers, item by item

The row above is the summary; this is the same assignment spelled out, so no item
can hide inside a range. **ACTIVE** opens with the decisive path — P12.1 (finish
T2 and preserve its verdict), P12.1a (the corrected-dispatch sensitivity run),
P12.2 (the `H*`-target diagnostics), P12.17 (the successor target, which is the
one next decisive test) — and then the criteria, instruments and constraints that
bind it. **CANDIDATE** is the defined mechanism space: every item carries a cheap
experiment, a falsifying control and a kill criterion, and none is scheduled.
**HYPOTHESIS** is retained with no implementation commitment at all.

#### ACTIVE — 29 items

**§1 — WS1 — Strategy and novelty boundaries**

| id | item |
| --- | --- |
| P1.1 | Compete on the joint design, not on sparse routing alone |
| P1.2 | Separate "1B active with a dormant expert bank" from the actual question |
| P1.3 | Maintain the novelty boundary ledger |
| P1.4 | Rank the three directions and defer the third |

**§2 — WS2 — Correct the registered architecture accounting**

| id | item |
| --- | --- |
| P2.1 | Record the Arm B sizing defect: routing is a compute saver, not capacity extraction |
| P2.3 | Correct Arm A adapter counting and re-verify the active union with the Rust ledger |
| P2.4 | Verify the teacher-target approximation against a full-distribution teacher, and add a hard-label control |

**§3 — WS3 — Error-directed capacity architecture (first experimental matrix)**

| id | item |
| --- | --- |
| P3.4 | First matrix: five arms, one backbone, one data order, one compute budget |

**§6 — WS6 — Success criteria and honest evaluation**

| id | item |
| --- | --- |
| P6.1 | Neural-capacity advantage |
| P6.2 | Physical-execution advantage |
| P6.3 | Every comparison reports quality, bytes/token and wall-clock |
| P6.4 | Keep the Rust core/Atlas/converter/format/store; add a trainable one-pass model with an independently measured active union |
| P6.5 | Pre-register before each run; keep negative results; no post-hoc thresholds |

**§10 — WS7 — Design doctrine derived from the selvedge metaphor**

| id | item |
| --- | --- |
| P7.1 | Weave selectivity in from the start; no post-hoc repair of a dense weave |
| P7.3 | The boundary may have a different structure from the body |
| P7.5 | Self-finished execution: the plan travels with its own evidence |
| P7.7 | The formal objective, with the three quantities kept apart |

**§15 — WS12 — Staged program and frontier claim**

| id | item |
| --- | --- |
| P12.1 | Finish T2 and preserve its verdict; audit the routed-overflow validity mask |
| P12.1a | The corrected-dispatch sensitivity run (successor to P12.1, registered separately) |
| P12.2 | Diagnose H*'s target: leave-one-out, per-expert normalisation, deployed-budget utility |
| P12.9 | Fresh Rust task suite |
| P12.10 | The control matrix |
| P12.11 | Frontier thresholds, fixed before any frontier run |
| P12.15 | Two pre-scaling technical problems, tracked as blockers |
| P12.16 | Anti-pattern guard: no new implementation program before the first three steps report |
| P12.17 | The successor utility target: interaction-aware, deployed-budget, absolute utility retained |

**§19 — WS14 — Runtime and evaluation corrections (physical-execution path; does not gate the T2 capability run — see §0.0)**

| id | item |
| --- | --- |
| P14.1 | Decoded-weight residency: stop re-decoding on every read |
| P14.2 | Prefetch is synchronous: separate fetch and compute timelines |
| P14.3 | Token-at-a-time prefill: keep the scalar oracle, add a batched path |


#### CANDIDATE — 102 items

**§2 — WS2 — Correct the registered architecture accounting**

| id | item |
| --- | --- |
| P2.2 | Capacity-scaling ladder with a genuinely oversized bank |

**§3 — WS3 — Error-directed capacity architecture (first experimental matrix)**

| id | item |
| --- | --- |
| P3.1 | Memory path with deterministic hashed addressing |
| P3.2 | Expert path: conditional residual specialists that do not duplicate memory or backbone |
| P3.3 | Error-directed training: utility target and pre-execution prediction |
| P3.5 | Training-recipe phases A–D |
| P3.6 | Cold-start handling: warm-up/diversity initialisation, measured specialisation |
| P3.7 | Predictability without forcing reuse: a separate prefetch predictor |

**§4 — WS4 — Physical bottleneck and placement**

| id | item |
| --- | --- |
| P4.1 | Record the bytes-per-token budget as a design constraint |
| P4.2 | Three-tier progression: VRAM → host RAM → selective NVMe |
| P4.3 | Three-way per-group decision: full precision / compressed approximation / omit |
| P4.4 | Weight-traffic equivalence reporting |
| P4.5 | Do not use sparse fragmentation when a resident contiguous approximation plus sparse correction is cheaper |

**§5 — WS5 — Inverted and combined mechanisms**

| id | item |
| --- | --- |
| P5.1 | Resident dense low-rank approximation + sparse exact delta |
| P5.2 | Chunk-tiered tiering (chunk union decides fetch) |
| P5.3 | Rotated shared-base delta experts |
| P5.4 | Adaptive dense correction triggered by delta mass |
| P5.5 | WINA / quantisation-error tiering |
| P5.6 | Native gate as router under the union budget, with low-parameter gating |
| P5.7 | Sequence-level budget allocation with per-output-token union accounting |
| P5.8 | Sequence-level masked-expert bank plus adaptive depth |
| P5.9 | Cheap interaction signals |
| P5.10 | Oracle-in-the-loop training |

**§10 — WS7 — Design doctrine derived from the selvedge metaphor**

| id | item |
| --- | --- |
| P7.2 | Continuity across the boundary: reusable task working sets, never forced router stickiness |
| P7.4 | The boundary must not distort what it protects |
| P7.6 | Composable capability protocol, with composability measured |

**§11 — WS8 — Complementarity and two-timescale organization**

| id | item |
| --- | --- |
| P8.1 | Task-level D/M selection on fresh held-out tasks |
| P8.2 | Compositional residuals with a bounded learned gate |
| P8.3 | Two-timescale selector: task-level working set, token-level inside it |
| P8.4 | Sublinear cold bytes per task and p95 latency as accessible capacity grows |
| P8.5 | Task-coherent specialization versus vocabulary specialization |

**§12 — WS9 — Deficit-directed capability construction**

| id | item |
| --- | --- |
| P9.1 | Counterfactual capability compiler |
| P9.2 | Three representations compete under one error-driven objective |
| P9.3 | Combination utility and an interaction-aware controller |
| P9.4 | Student-deficit-conditioned addressing |
| P9.5 | Teacher-derived memory is immutable and versioned at inference |
| P9.6 | Verifier-compiled procedural capabilities |
| P9.7 | Memory-selected recurrent reasoning |

**§13 — WS10 — Eighteen candidate mechanisms, each with its own cheap decisive experiment**

| id | item |
| --- | --- |
| P10.1 | Hypernetwork-generated experts |
| P10.2 | LatentMoE geometry: near-square experts at fixed active cost |
| P10.3 | Writable fast-weight memory split into stable / episodic / working |
| P10.4 | Supervised latent reasoning |
| P10.5 | Multi-view routing and expert modulation |
| P10.6 | Intrinsic-dimension-guided capacity allocation across layers |
| P10.7 | Query-selective test-time updates on multi-stage tasks |
| P10.8 | Verifier-compiled symbolic solvers for recurring reasoning classes |
| P10.9 | Mixture-of-Recursions adaptive depth with shared weights |
| P10.10 | Student-aligned reasoning distillation |
| P10.11 | One-token internal correctness probes |
| P10.12 | Conflict-directed repair |
| P10.13 | Task-conditioned layer-wise expert budgets under one global budget |
| P10.15 | Information-density-aware expert compression |
| P10.17 | Fast-weight product key memory as a writable episodic store |
| P10.18 | Optimizer court: FOAM vs Muon vs tuned AdamW |

**§14 — WS11 — Twenty-one original mechanism proposals**

| id | item |
| --- | --- |
| P11.1 | Causal Bottleneck Repair |
| P11.2 | Capability Operator Algebra |
| P11.3 | Functional Innovation Storage |
| P11.4 | Counterfactual State Transplants |
| P11.5 | Representation Impedance Matching |
| P11.6 | Error-Syndrome Addressing |
| P11.7 | Residual Safety Escrow |
| P11.8 | Capability Birth and Fusion |
| P11.9 | Counterexample-Constrained Routing |
| P11.10 | Inter-Layer Shortcut Capsules |
| P11.11 | Temporal Capability Leasing |
| P11.12 | Failure-Directed Latent Repair |
| P11.13 | Predictive Capability Checkpoints |
| P11.14 | Reasoning State Compression |
| P11.15 | Precision-Selective Intelligence |
| P11.16 | Adversarial Capability Crystallization |
| P11.17 | Causal Computation Auctions |
| P11.18 | Adaptive Verification Depth |
| P11.19 | Neural State Transactionality |
| P11.20 | Capability Conservation Training |
| P11.21 | Anti-superficial-specialization suite |

**§15 — WS12 — Staged program and frontier claim**

| id | item |
| --- | --- |
| P12.3 | T3: latent-geometry court (P10.2) |
| P12.4 | T4: generated-weight versus stored-weight court (P10.1) |
| P12.5 | T5: fast-weight adaptation court (P10.3 / P10.7) |
| P12.6 | T6: latent-reasoning court (P10.4 / P11.14) |

**§18 — WS13 — Published mechanisms and sixteen inventions, run as competing alternatives**

| id | item |
| --- | --- |
| P13.1 | Adaptive expert widths: nested slimmable experts under one budget |
| P13.2 | Cross-layer expert reuse: shared pools over 2 / 4 / 7 layers |
| P13.3 | Global experts plus a recurrent routing state |
| P13.4 | Typed latent reasoning with a learned allocation budget |
| P13.6 | Capability Seed Compilation |
| P13.7 | Constraint-Debt Reasoning |
| P13.8 | Metamorphic Quotient Routing |
| P13.9 | Semantic Refinement Ladders |
| P13.10 | Verified Constraint Distillation |
| P13.11 | Certified Read Elision |
| P13.12 | Interaction Compilation |
| P13.13 | Conditional Representation Expansion |
| P13.14 | Omission-Conditioned Recovery |
| P13.15 | Counterexample-Preserving Compression |
| P13.16 | Latent Branch-and-Reconcile |
| P13.17 | Resident-Side-Information Coding |
| P13.18 | Phase-Transition Capability Switching |
| P13.19 | Executable Capability Contracts |
| P13.20 | Training for Future Locality |
| P13.21 | Verifier-Calibrated Internal Stopping |

**§19 — WS14 — Runtime and evaluation corrections (physical-execution path; does not gate the T2 capability run — see §0.0)**

| id | item |
| --- | --- |
| P14.4 | Expand the evaluation suite beyond 24 tasks |
| P14.5 | Strengthen checker coverage and scale for Rust cases (execution already exists) |
| P14.6 | Keep correctness separate from reference-continuation likelihood |
| P14.7 | A sealed evaluation set for external comparison |


#### HYPOTHESIS — 9 items

**§5 — WS5 — Inverted and combined mechanisms**

| id | item |
| --- | --- |
| P5.11 | Recurrent latent computation as a separate later axis |

**§13 — WS10 — Eighteen candidate mechanisms, each with its own cheap decisive experiment**

| id | item |
| --- | --- |
| P10.14 | Shortcut-connected specialist execution |
| P10.16 | Learned reasoning-length control |

**§15 — WS12 — Staged program and frontier claim**

| id | item |
| --- | --- |
| P12.7 | T7: the joint court, then capacity scaling |
| P12.8 | Capability court, three levels |
| P12.12 | Three distinguishable achievement claims |
| P12.13 | Transfer test: equivalent capability libraries on a small and a larger backbone |
| P12.14 | Wildcards: mHC/xHC multi-stream residual capacity, low-rank logits basis |

**§18 — WS13 — Published mechanisms and sixteen inventions, run as competing alternatives**

| id | item |
| --- | --- |
| P13.5 | Integration controller (only after the individual winners report) |

## 0.2 P-correction — the ordering inversion, recorded and fixed

**The inversion.** The §16 queue as written on 2026-11-15 ended with

```
V[P12.8 levels 1-3] --> W[P12.10 control matrix P12.11 thresholds]
```

and its hard-order list put step 7 "**T7 and P12.8** — only winners, then the
three levels" before step 8 "**P12.10–P12.12** — the control matrix and the
claims". The diagram therefore scheduled the **control matrix** (P12.10) and the
**frontier thresholds** (P12.11) *after* the experiments they govern.

**Why that is a defect and not a preference.** P12.10 fixes the row set every
comparison is reported in, including the two rows the item itself says kill most
results ("Memory Grafting-style latent baseline" and "matching library without
routing optimisation"). P12.11 fixes the ≥10% threshold, the capacity-scaling
criterion, the replay court, the Pareto requirement and the branch split. If both
are built after T7, then T3–T6 are designed, run and read without the controls
that would have falsified them, and the frontier run is scored against
thresholds that were written after its numbers existed. That is the exact
post-hoc-threshold failure P6.5 and P12.11 exist to prevent, appearing inside the
plan that forbids it.

The dependency text was already correct and only the diagram was wrong: P12.10's
`Depends` is "P12.9, P7.7" and P12.11's is "P12.10" — neither names T7 or P12.8.
The edge was drawn from a "experiments, then claims" narrative rather than from
the `Depends` fields.

**The correction.** *Decision criteria and the measurement instrument move before
the experiments they govern; infrastructure may still be finalised later.*

| established before the first governed experiment | may be finalised later |
| --- | --- |
| the fresh suite (P12.9) — exists | residency tiers, GPU staging, kernels (P14.1–P14.3, WS4) |
| the control matrix's schema **and row set** (P12.10) — schema exists; each run fills its row | the Level-2 capacity ladder and Level-3 assembly (P12.8, P2.2) |
| the frontier thresholds and gating checks (P12.11) — pre-registered | the joint court's ablation matrix (P12.7) |
| the two pre-scaling blockers closed (P12.15) | the transfer test (P12.13) and the wildcards (P12.14) |
| the successor utility target registered (P12.17) | the third achievement claim (P12.12, Level 3 wording) |
| the anti-pattern guard in force (P12.16) | |

The rule this leaves behind, for every future edit to this plan: **a threshold, a
row set, a control, or a kill criterion may never be written after the numbers it
judges; an engineer, a kernel, a residency tier or an additional claim may be.**
§16 is redrawn accordingly.

---

## 1. WS1 — Strategy and novelty boundaries

### P1.1 — Compete on the joint design, not on sparse routing alone

**What.** State, in the contract and in the README, that the program's
contribution candidate is the *joint* design of (a) learned conditional
capacity, (b) a learned retrieval/memory path, and (c) physical execution under
a strict per-token union budget — and that sparse routing by itself is neither
new nor the claim.

**Why.** Every component in isolation has prior art (see
`docs/PRIOR-ART-AND-NOVELTY.md`). A claim framed around routing would be
refuted by Deja Vu, PowerInfer, SEAP, SparseInfer, MoE-CORE and Prefetch-Steered
Routing on day one.

**Depends.** none.
**Acceptance evidence.** `docs/EXPERIMENT-CONTRACT.md` amendment adding the claim
sentence and the "not a claim" list; `README.md` current-result paragraph that
matches it; a `perf_numbers` court run proving no number was hand-written.
**Status.** open.
**Effort.** S.

### P1.2 — Separate "1B active with a dormant expert bank" from the actual question

**What.** Record the distinction explicitly: a 1B-active model with a large
dormant expert bank is established practice (OLMoE-1B-7B reports roughly 1.3B
active; the "1B" in its name is not a strict-union count). SELVEDGE's question
is different: whether *additional* capacity can be organised so that its
retrieval is predictable under a parameter budget **and** an I/O budget, at a
strictly counted union.

**Why.** Without this distinction the program will be read as re-running
well-known MoE inference, and the review will be right.

**Depends.** P1.3.
**Acceptance evidence.** a paragraph in the contract's "what this is not"
section, with the OLMoE citation and its verified active-parameter number
(`docs/PRIOR-ART-AND-NOVELTY.md`, entry `olmoe`).
**Status.** open.
**Effort.** S.

### P1.3 — Maintain the novelty boundary ledger

**What.** Keep `docs/PRIOR-ART-AND-NOVELTY.md` live. It must name, for each
mechanism the program intends to use, the closest antecedent, the overlapping
features, and what is genuinely uncombined. The boundary list is fixed:

- **Upcycling prior art:** Drop-Upcycling, SIMoE, UpIT, ToOL/ToMoE-style
  conversion, cluster-aware upcycling.
- **Learned/conditional memory prior art:** Memory Layers at Scale, Engram
  (and its derived works), PEER, Monet.
- **Predictive loading / cache-aware routing prior art:** SeqMoE, SpecMD,
  SpecPrefetch, Prefetch-Steered Routing, DoMoE, MoE-CORE, NeuroPrefetcher,
  MoE-Infinity.
- **Activation-sparsity prior art:** Deja Vu, PowerInfer, attribution-based
  sparse activation (2026), SEAP, SparseInfer, R-Sparse, La RoSA, Sirius,
  WINA, SpAx, BlockFFN, SPON.

The only claim permitted is: a **joint trained error-directed system with strict
union accounting** that combines capacity, retrieval and physical execution, at
a measured operating point.

**Why.** The mandate makes this a deliverable, and unverified citations are the
fastest route to an invalid result.

**Depends.** none.
**Acceptance evidence.** the ledger exists, every claim carries a status
(`VERIFIED`/`PARTIAL`/`UNSUPPORTED`/`PROPOSED`), every URL carries a retrieval
date, and `scripts/prior_art_fetch.py` reproduces the fetched bytes into
`results/prior-art/`. Re-run at every milestone advance.
**Status.** partial (ledger created this phase; must be refreshed when WS3/WS4
experiments are pre-registered).
**Effort.** S per refresh, M for a full re-verification.

### P1.4 — Rank the three directions and defer the third

**What.** Adopt this ranking for the phase:

1. **Error-directed memory + expert specialization** (WS3) — first.
2. **Utility-weighted predictive execution** (WS4/WS5.10) — second.
3. **Hierarchical shared-base + delta specialists** (WS5.1, WS5.3) — deferred
   until a trained bank shows a quality win.

**Why.** Direction 3 is a compression/parameter-efficiency play. Its benefit is
conditional on there being a specialist bank whose quality is worth preserving.
Building it first would optimise a bank that may not exist.

**Depends.** P3.4 (the pilot that decides whether a bank is worth preserving).
**Acceptance evidence.** the ranking recorded in the contract with the explicit
deferral condition ("deferred until P3.4 shows a quality win for the trained
bank over the dense control").
**Status.** open.
**Effort.** S.

---

## 2. WS2 — Correct the registered architecture accounting

### P2.1 — Record the Arm B sizing defect: routing is a compute saver, not capacity extraction

**What.** Amend T1.2 to state Arm B's **total accessible** count alongside its
active count, and to draw the consequence: with a 596,049,920-element core and
8 specialists per layer at 1,572,864 elements each, the whole bank is
352,321,536 and the whole model is 948,600,832 — already under the 1e9 ceiling.
Arm B therefore does not extract capacity from beyond the ceiling at all; it
trades *compute* for quality at equal total size. That is a legitimate
experiment, but it is not the capacity-extraction experiment the program
eventually needs.

**Why.** Without this, a reader will read Arm B as evidence about dormant
capacity, which it structurally cannot be. The arithmetic is already encoded in
`oracle/train/arms.py` (`DESIGN_TOTAL_ACCESSIBLE["B"]`), but the contract's
T1.2 never states it.

**Depends.** none (arithmetic verified in §2.6 below).
**Acceptance evidence.** contract amendment (next free letter) containing the
total-accessible formula and the "compute saver" sentence; `scripts/train-admission.sh`
receipt for Arm B whose emitted record carries `total_accessible` below the
ceiling; the `rust` court at that revision.
**Status.** open (arithmetic verified; amendment not written).
**Effort.** S.

### P2.2 — Capacity-scaling ladder with a genuinely oversized bank

**What.** Register a ladder of specialist counts per layer at width 512 and
`k = 4` selected, at 8 / 16 / 64 / 256 specialists, and record for each the
total accessible parameters, the active union and the BF16 weight footprint.
Verify every figure from the actual architecture constants and router sizes
(exact values in §2.6). State the BF16 footprint at the top rung.

**Why.** The ladder is the experiment that distinguishes "routing saves
compute" from "capacity beyond the ceiling helps". At 8 specialists the whole
model is under the ceiling and the bank alone is only a third of it; the total
first exceeds the ceiling at 16, and the bank alone first exceeds it at 64.
Without the ladder, a null result at 8 specialists would be misread as a null
result about dormant capacity.

**Depends.** P2.1.
**Acceptance evidence.** contract amendment with the ladder table; a CPU test
that recomputes each rung from `results/accounting/*-qwen3-0.6b.json` and
`arms.py` constants and fails closed on any mismatch; an admission record per
rung produced by `selvedge admit` (the ledger is the source of truth, not
Python).
**Status.** open (arithmetic verified; test and admission records not written).
**Effort.** M.

### P2.3 — Correct Arm A adapter counting and re-verify the active union with the Rust ledger

**What.** Re-verify Arm A end to end. The adapter is a low-rank factorisation
over the 384-wide group activation, `r*(384+384)` elements per selected group —
not `r*(384+2048)`. The first committed text had the latter. Recompute A's
active union with the Rust `ParameterLedger` and record the admission receipt.

**Why.** The difference is load-bearing, not cosmetic: the wrong form adds
7,454,720 unique elements and puts A at 1,005,724,672, i.e. **over** the
ceiling. A review flagged the inconsistency; the contract's A5 corrects it, but
the corrected form has not been re-admitted by the Rust ledger at the current
revision.

**Depends.** none.
**Acceptance evidence.** `selvedge admit` record for A and A0 at HEAD with
`active_unique_params` equal to the design count; a guard test that the
`r*(384+2048)` variant is **rejected** by the ledger (proving the guard works,
not merely that the good case passes); the `rust` court receipt.
**Status.** partial (contract corrected in A5; ledger re-admission not recorded).
**Effort.** S.

### P2.4 — Verify the teacher-target approximation against a full-distribution teacher, and add a hard-label control

**What.** Measure whether the T1 objective (top-16 support + one tail bucket,
`docs/TRAINING.md` §2) ranks arms the same way as the full-vocabulary
cross-entropy from a live teacher. Then register a **hard-label control arm**
(teacher argmax, or the true next token) trained identically, to separate "the
compact target is adequate" from "distillation itself is doing the work".

**Why.** Every T1 comparison uses the compact target. If the ranking it induces
differs from the full target's, the T1 conclusion is about the instrument, not
the architecture. The tail bucket is lossy for the teacher by construction; the
question is whether the loss matters for the comparison.

**Depends.** P2.2 (the arms being ranked), P3.5 phase A/B (the trained models
whose ranking is compared).
**Acceptance evidence.** a receipt comparing, on a held-out sample, the compact
target loss and the full-distribution CE for the same student checkpoints, with
rank correlation and paired differences; contract amendment adding the
hard-label arm with its own pre-registered threshold; the `oracle` court that
runs the comparison twice (two seeds) so a single run cannot decide it.
**Status.** open.
**Effort.** L (requires trained checkpoints).

### 2.6 Derived design counts (arithmetic, not measurement)

Constants are the registered receipts' unique counts (T1.1): Qwen3-0.6B core
`596,049,920`; hidden `1024`; 28 layers; specialist width `512`; `k = 4`.
One specialist is `3 × 512 × 1024 = 1,572,864` elements. One router is
`1024 × s` elements per layer.

| specialists/layer `s` | bank elements (28·s·1,572,864) | router elements (28·1024·s) | total accessible | active (core + 28·4·per + routers) | BF16 bytes (2 × total) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 352,321,536 | 229,376 | 948,600,832 | 772,440,064 | 1,897,201,664 |
| 16 | 704,643,072 | 458,752 | 1,301,151,744 | 772,669,440 | 2,602,303,488 |
| 64 | 2,818,572,288 | 1,835,008 | 3,416,457,216 | 774,045,696 | 6,832,914,432 |
| 256 | 11,274,289,152 | 7,340,032 | 11,877,679,104 | 779,550,720 | 23,755,358,208 |

Reading:

- At 8 specialists the **total** is under the ceiling, so the bank is not
  capacity beyond the ceiling. This is P2.1.
- From 16 upward the total exceeds the ceiling while the active union stays
  essentially flat (only the router term moves), which is the capacity question
  the program actually wants to ask.
- The top rung's BF16 footprint is `23,755,358,208` bytes = 22.12 GiB. The
  strategy note's "~23.7 GB" and "11.872B" are rounded differently; the exact
  values here include the routers and should replace them.
- Arm A's arithmetic is `663,610,368 + 28·5·2,359,296 + 28·5·2·32·384 +
  28·2048·16 = 998,269,952` (adapters `3,440,640`, routers `917,504`). The
  rejected `r*(384+2048)` form gives `1,005,724,672`, over the ceiling. This is
  P2.3.

Reproduce (pure integer arithmetic, no model access):

```bash
python3 - <<'PY'
core, h, layers, w, k = 596_049_920, 1024, 28, 512, 4
per = 3 * w * h
for s in (8, 16, 64, 256):
    bank, routers = layers * s * per, layers * h * s
    total = core + bank + routers
    active = core + layers * k * per + routers
    print(s, bank, routers, total, active, 2 * total)
PY
```

---

## 3. WS3 — Error-directed capacity architecture (first experimental matrix)

### P3.1 — Memory path with deterministic hashed addressing

**What.** Implement a learned memory path that:

- addresses deterministically by hashing the **token prefix** (n-gram keys);
  there is no scan of a large key table per token;
- stores 256-dimensional values;
- has `2^18` to `2^20` rows;
- is inserted at a small number of layers, not all 28;
- performs 2–4 reads per token;
- charges the active union as **only the retrieved value rows plus the index
  parameters actually evaluated** — never the whole table;
- records resident bytes even when the table is resident, and bytes read when
  it is not.

**Why.** The memory path is the cheapest test of "capacity that is not
procedural computation". It must be physically honest: a table scan would blow
the union budget while looking cheap in a logical count (the M2 defect class).

**Depends.** P1.3 (Engram/Memory Layers/PEER/Monet boundaries), P2.3 (ledger
admission path).
**Acceptance evidence.** a Rust ledger test proving the union for a memory read
is `rows_read × 256 + evaluated index elements`; a leakage test proving the key
is a function of the prefix only; a receipt recording resident bytes and read
bytes separately; `rust` + `oracle` courts.
**Status.** open (no memory module exists).
**Effort.** L.

### P3.2 — Expert path: conditional residual specialists that do not duplicate memory or backbone

**What.** Implement the specialist bank as conditional residual
transformations selected by a small router. Add an explicit **non-duplication
audit**: (a) no specialist may subsume the dense backbone FFN's function at
init (zero-init parity), (b) no specialist may be a copy of a memory row set,
(c) each specialist must show measurable residual-correction behaviour on held
out data rather than reproducing the backbone output.

**Why.** A bank that learns to imitate the backbone or the memory is capacity
that does nothing. The audit turns "experts specialised" into a falsifiable
measurement.

**Depends.** P3.1 (to measure memory/expert complementarity), P3.6.
**Acceptance evidence.** a complementarity report: per-specialist residual
correlation with the backbone output, cosine similarity of specialist updates
to zero, and a memory-vs-expert ablation at matched budget; the audit runs on a
trained checkpoint, not at init.
**Status.** open (the `SpecialistBank` module exists; the audit does not).
**Effort.** M after P3.6.

### P3.3 — Error-directed training: utility target and pre-execution prediction

**What.** Define the utility target

`u_e(x) = L(without e) − L(with e)`

measured **on calibration text only**, with real labels and real interventions.
Train the router to predict the **pre-execution incremental benefit** of each
candidate. Then run the decisive comparison against magnitude/probability
routing at the same budget.

**Why.** This is the program's central scientific bet (P1.4 direction 1). M2
showed a post-hoc magnitude statistic fails for *omission*; that does not answer
whether a trained router can predict the *incremental benefit* of *added*
capacity. If it cannot beat magnitude, the error-directed idea is dead and the
program should say so.

**Depends.** P3.2, P3.5 phase B.
**Acceptance evidence.** a pre-registered contract for the comparison; a receipt
with paired bootstrap CIs; a **leakage test** that the router's features are
computable before the expert executes (no future layer state, no teacher logit,
no gold token); the `oracle` court.
**Status.** open.
**Effort.** XL (training) + M (instrument).

### P3.4 — First matrix: five arms, one backbone, one data order, one compute budget

**What.** Run: **dense control**, **memory-only**, **expert-only**, **naive
hybrid**, **error-directed hybrid**. Same backbone, same data order, same
training compute. Scale the bank **only if** the error-directed hybrid wins.

**Why.** The matrix is the minimum that can attribute a win to *error
direction* rather than to *more parameters* or *more modules*. The gating rule
prevents the program from scaling a design that has not earned it.

**Depends.** P3.1, P3.2, P3.3, P2.4 (instrument check), P6.5 (pre-registration).
**Acceptance evidence.** a receipt per arm with token count, step count, loss
curve, held-out metrics, active union and bytes/token; a pre-registration
commit before the first step; K-style kill record if the hybrid does not win.
**Status.** **this is the NOW experiment** at pilot scale (§8).
**Effort.** XL.

### P3.5 — Training-recipe phases A–D

**What.** Register and run four phases in order:

- **A. Memory residual.** Memory path versus a parameter-matched dense adapter
  versus a learned-memory baseline (Memory Layers-style).
- **B. Specialists on residual errors.** Real labels + distillation + sampled
  counterfactual utility.
- **C. Complementarity.** memory-only vs expert-only vs both vs dense control at
  matched training compute.
- **D. Dormant capacity beyond GPU memory.** Quality, utilisation long tail,
  cold bytes/token, p95 latency.

**Why.** Each phase isolates one claim. Phase D is the only one that tests the
program's physical premise, and it is worthless before C shows a quality reason
to keep the bank.

**Depends.** A: P3.1. B: P3.2. C: A+B. D: C + WS4.
**Acceptance evidence.** one pre-registration per phase; per-phase receipts;
phase C's matched-compute table with CIs; phase D's long-tail utilisation
histogram and cold-bytes distribution.
**Status.** open.
**Effort.** XL total.

### P3.6 — Cold-start handling: warm-up/diversity initialisation, measured specialisation

**What.** Zero-init parity remains required. Add and compare specialist warm-up
initialisations: (i) random, (ii) dense-weight-derived, (iii) semantic-cluster
initialisation (cluster centroids as router init, SVD subspaces as expert init —
the mechanism Cluster-Aware Upcycling reports). Measure specialisation
explicitly (routing entropy, expert-usage Gini, per-expert residual error).
**Do not assume that more tokens fixes the cold start.**

**Why.** All-identical expert initialisation plus a random router produces
expert symmetry; the prior art says so and proposes a fix. The program must
test that fix rather than rediscover the failure.

**Depends.** P3.2.
**Acceptance evidence.** a receipt comparing the three initialisations at
matched compute with (a) the specialisation metrics above, (b) held-out quality,
(c) routing entropy over training; a kill note if more tokens does not close a
persistent symmetry gap.
**Status.** open.
**Effort.** M + training.

### P3.7 — Predictability without forcing reuse: a separate prefetch predictor

**What.** Add an auxiliary prefetch predictor that is **separate from the
router**. The router decides what executes; the predictor only decides what to
load. A prefetch miss causes a **stall**, never a different result. There is no
router-stickiness penalty: do not bias routing toward resident experts.

Record the 2026 negative result as prior evidence: a stickiness-style scheme cut
misses by roughly 60% but failed the 1%-PPL preservation test, so the trade was
rejected.

**Why.** Cache-aware routing changes the deployed model's semantics
(documented by Prefetch-Steered Routing) and the separate-predictor design is
already published (SpecPrefetch: a shared lightweight adapter predicts
next-layer candidates while the frozen native router decides execution). The
honest claim is therefore *not* "predictive loading"; it is that the predictor
is scored inside the same union/bytes budget as the selector, and that a miss is
provably semantics-preserving.

**Depends.** P4.2 (a tier to prefetch from), P3.4.
**Acceptance evidence.** an exact-mode parity court (Court E) proving output
equality under prefetch on/off and under injected transfer delay; a receipt with
miss rate, exposed stall p50/p95, and prefetch-wasted bytes; the negative
result recorded with its receipt.
**Status.** open.
**Effort.** L.

---

## 4. WS4 — Physical bottleneck and placement

### P4.1 — Record the bytes-per-token budget as a design constraint

**What.** Compute the ceiling on bytes/token from **measured** cold throughput
and a stated decode rate, and record it as a constraint: at the cold no-cache
device rate in `docs/STORAGE.md` (`storage-17b-none-cold`) and a target of 20
tokens/s, the ceiling is `rate / 20` MiB/token. Arm B's four selected experts
per layer are 336 MiB in BF16 per token. Naive fetch is therefore impossible,
not merely slow.

**Why.** This is the number that decides the architecture. If the active set
cannot be fetched in the decode interval, the design must be about residency,
compression or recomputation — not about routing cleverness.

**Depends.** none (measured storage baselines already exist).
**Acceptance evidence.** a section in the contract that references the generated
block, states 20 tok/s as a **design target supplied by the operator, not a
measurement**, and derives the MiB/token ceiling; a test that fails if the Arm B
per-token expert bytes exceed the derived ceiling (so the constraint cannot
silently stop binding).
**Status.** open.
**Effort.** S.

### P4.2 — Three-tier progression: VRAM → host RAM → selective NVMe

**What.** Register and execute the progression: (1) bank fits in VRAM; (2) bank
exceeds VRAM but fits host RAM; (3) bank requires selective NVMe. **Isolate
quality, then placement, then offload** — do not confound them.

**Why.** Tier 1 measures the neural question with no I/O noise. Tier 3 measures
the systems question. Comparing a tier-3 system against a tier-1 baseline
measures the tier, not the architecture.

**Depends.** P3.4 (a bank worth placing).
**Acceptance evidence.** three receipts with the same checkpoint, one per tier,
reporting quality and physical cost; a placement-only ablation (same quality,
different tier); the `io` court for the NVMe tier.
**Status.** open.
**Effort.** XL.

### P4.3 — Three-way per-group decision: full precision / compressed approximation / omit

**What.** For each candidate group, choose among (a) full-precision execution,
(b) a compressed approximation, (c) omission. Assign the tier by **predicted
error contribution**, not by magnitude alone.

**Why.** The three-way decision is a known mechanism: SpAx (arXiv 2610.02598)
replaces the binary read/skip choice with exactly "fully retain, approximate,
omit" for offloaded weights. The program can still test *predicted-error-based
tier assignment* against magnitude, but it must not present the three-way choice
as new.

**Depends.** P4.2, P3.3 (an error predictor).
**Acceptance evidence.** a receipt comparing magnitude-assigned tiers against
predicted-error-assigned tiers at matched bytes; a calibration curve for the
error predictor; an explicit overlap statement in the ledger.
**Status.** open.
**Effort.** L.

### P4.4 — Weight-traffic equivalence reporting

**What.** Every quality comparison must report **unique parameters**, **bytes
per token** (including selector, router and prefetch bytes) and an **equivalent
bit-width** alongside accuracy.

**Why.** A quality win at 4× the traffic is not a win. Equivalent bit-width
makes a dense BF16 baseline and a sparse mixed-precision system comparable on
one axis.

**Depends.** P2.3 (the union ledger), P3.7 (prefetch bytes).
**Acceptance evidence.** a receipt schema requiring the three fields; a
generated table in the plan/results with the fields for every arm; a test that
refuses a comparison row missing any of them.
**Status.** partial (M1/M2 report active counts and bytes; equivalent bit-width
is not in the schema).
**Effort.** M.

### P4.5 — Do not use sparse fragmentation when a resident contiguous approximation plus sparse correction is cheaper

**What.** Test the inversion: compare (a) fetching sparse fragmented exact
groups against (b) keeping a resident contiguous low-rank approximation and
applying a sparse correction. If (b) is cheaper in bytes or latency at equal
quality, adopt it.

**Why.** Fragmented reads pay alignment and queue-depth costs that a contiguous
resident approximation does not (the measured over-read amplification in
`docs/STORAGE.md` is the evidence that this matters). The strategy note asserts
the inversion; it must be measured, not assumed.

**Depends.** P4.2, P5.1 (the approximation).
**Acceptance evidence.** a measured bytes/latency comparison at matched quality,
with the `io` court and p50/p95 reported.
**Status.** open.
**Effort.** L.

---

## 5. WS5 — Inverted and combined mechanisms

Each item below is literature-derived and **requires verification before
adoption**. Statuses reference `docs/PRIOR-ART-AND-NOVELTY.md`; where the named
work could not be verified, the item says so and the mechanism is treated as
`PROPOSED`, not prior art.

### P5.1 — Resident dense low-rank approximation + sparse exact delta

**What.** Represent `W = W_lowrank + Δ_sparse`, score columns by
`|x| · ||W_col − Q(W_col)||`, and measure bytes saved at equal PPL. **Kill if**
bytes are reduced by less than 15% at equal PPL on 2 of 3 models.

**Why.** This is the cheapest way to keep a contiguous resident base while
sending only corrections over the bus.

**Depends.** P4.5, P1.4 direction 3 (deferred until a trained bank wins).
**Acceptance evidence.** a receipt with per-model bytes and PPL at the fixed
threshold; the kill decision recorded even if negative.
**Status.** proposed / deferred.
**Effort.** L.

### P5.2 — Chunk-tiered tiering (chunk union decides fetch)

**What.** Make the fetch unit a **chunk union** over consecutive tokens rather
than a per-token set, and measure whether chunk-level PPL degrades faster than
per-token PPL. **Kill if** it does.

**Why.** BlockFFN shows that token-level sparsity has poor chunk-level sparsity,
i.e. the union over consecutive tokens activates a large fraction. That is a
direct threat to per-token fetch plans and a possible argument for chunk
tiering. Whether the quality cost is acceptable is the open question.

**Depends.** P4.2.
**Acceptance evidence.** a receipt with chunk sizes {2,4,8,16}, chunk-PPL vs
per-token-PPL, and bytes/token per chunk size; the `io` court for the fetch
side.
**Status.** proposed.
**Effort.** L.

### P5.3 — Rotated shared-base delta experts

**What.** Combine three verified mechanisms: an orthogonal rotation of
activations (La RoSA), cluster-aware initialisation (Cluster-Aware Upcycling),
and shared-base-plus-delta experts (DeRS). **Kill if** the rotation *raises*
inter-expert similarity (measured by pairwise delta cosine), since the rotation
would then be destroying specialisation rather than enabling it.

**Why.** Each component exists; the combination is the candidate. The kill
criterion is the one that detects the failure mode the combination invites.

**Depends.** P3.6 (initialisation), P5.1 (delta representation).
**Acceptance evidence.** an inter-expert similarity matrix before/after
rotation, plus quality and bytes at matched budget.
**Status.** proposed.
**Effort.** L.

### P5.4 — Adaptive dense correction triggered by delta mass

**What.** Trigger a dense correction pass when the sparse delta's mass exceeds a
threshold. Build it **only if** delta mass correlates with sparse error at
Spearman `≥ 0.5` on held-out data; **kill below 0.3**.

**Why.** Without the correlation, the trigger is an uninformative heuristic and
the correction pass is wasted compute.

**Depends.** P5.1.
**Acceptance evidence.** a held-out correlation receipt with CI; the build/kill
decision recorded.
**Status.** proposed.
**Effort.** M.

### P5.5 — WINA / quantisation-error tiering

**What.** Test whether activation-magnitude-based selection already proxies for
quantisation error (WINA argues magnitude alone is a poor criterion). If
magnitude does proxy it, tier assignment can stay cheap; if not, use the
measured quantisation error.

**Why.** Cheapest possible version of P4.3's "predicted error, not magnitude"
requirement, and it can falsify the requirement.

**Depends.** P4.3.
**Acceptance evidence.** a receipt correlating magnitude with measured
quantisation error per group, and the resulting tier-assignment quality at
matched bytes.
**Status.** proposed.
**Effort.** M.

### P5.6 — Native gate as router under the union budget, with low-parameter gating

**What.** Train the native SwiGLU gate as the top-k router (SGA-Sparse) and
compare against a learned low-parameter gate (Model Casting / LoPA Gating), all
under the strict union budget. Measure whether this **removes selector cost**
entirely (no separate router parameters or reads).

**Why.** A selector that costs nothing is worth more than a better selector that
costs 5% of the budget. SGA-Sparse also shows the gate/up ordering *reverses*
after sparsity-aware training, which invalidates training-free routing
comparisons.

**Depends.** P2.3 (the ledger must be able to express "gate is both selector and
execution").
**Acceptance evidence.** a receipt with selector elements = 0 for the
native-gate arm, quality at matched active budget, and a leakage/ordering test.
**Status.** proposed.
**Effort.** L.

### P5.7 — Sequence-level budget allocation with per-output-token union accounting

**What.** Allocate the expert budget across a sequence (SeqTopK-style) instead
of per token, but account the union **per output token** as the contract
requires. Compare against per-token top-k at the same sequence budget.

**Why.** Sequence-level allocation is verified prior art (SeqTopK reports under
1% overhead and consistent gains). The open question for SELVEDGE is whether it
survives the strict per-token union rule, which sequence-level schemes can
violate at the worst token.

**Depends.** P3.4.
**Acceptance evidence.** a receipt reporting both the sequence-average and the
**worst per-token** union, with a compliance verdict per token (not just per
sequence).
**Status.** proposed.
**Effort.** M.

### P5.8 — Sequence-level masked-expert bank plus adaptive depth

**What.** Combine a masked-expert bank over frozen weights (MASKerade-style
per-token masks) whose mask *set* is predicted at sequence level
(SparseInfer-style), with adaptive recursive depth (Mixture-of-Recursions-style)
and recursion-wise KV caching. Use a staged curriculum.

**Why.** Both halves are verified prior art. The combination is attractive
because masks are cheap to store and depth is a second, orthogonal budget knob.
The risk is that masked subnetworks still read the full frozen tensor unless the
runtime slices them — so the union accounting must be checked, not assumed.

**Depends.** P4.2, P3.2.
**Acceptance evidence.** a receipt proving the masked execution reads only
selected elements (profiler cross-check, as in `docs/M2-CORRECTION.md` §5), plus
quality at matched depth/budget; a curriculum table.
**Status.** proposed.
**Effort.** XL.

### P5.9 — Cheap interaction signals

**What.** Implement and compare four low-cost interaction mechanisms:
(a) low-rank **pairwise** interaction scores (the "HOPE-style" idea — note:
`HOPE` could not be verified as a paper with this mechanism, so treat it as
`PROPOSED`); (b) online geometric centroids; (c) tree-structured routing;
(d) spontaneous/resting anchor vectors (SPON, verified: input-independent
learnable vectors trained by distribution matching).

**Why.** M2 found the pairwise interaction term is context-dependent and not
predicted by the tested graph features, using a linear bilinear surrogate. The
cheap mechanisms here are the plausible next step — but only SPON has a verified
source.

**Depends.** P3.3.
**Acceptance evidence.** a head-to-head receipt at matched selector cost, with
the M2 interaction finding cited as the baseline to beat.
**Status.** proposed.
**Effort.** L.

### P5.10 — Oracle-in-the-loop training

**What.** Use the offline budgeted oracle (the M2 `offline oracle` instrument,
labelled causally unavailable and non-deployable) to label optimal group sets on
training text, then distill a cheap **causal** selector from those labels.
Compare the distilled selector against the oracle's own bound to measure the
distillation gap.

**Why.** M2 established the oracle bound is outside the usable tier for
*omission*. The open question is whether an oracle over *added capacity* has a
bound worth distilling. If the bound is poor, no selector can help and the
program should stop early.

**Depends.** P3.3, P3.4.
**Acceptance evidence.** a receipt with the oracle bound, the distilled selector,
the gap and its CI; leakage test proving the student sees only pre-decision
features.
**Status.** proposed.
**Effort.** L.

### P5.11 — Recurrent latent computation as a separate later axis

**What.** Keep recurrent latent computation as a distinct axis with **no
guaranteed benefit**. If pursued, it needs its own hypothesis, baseline and kill
criterion, plus the critical recurrent analysis as prior evidence.

**Why.** Huginn reports gains from latent recurrence; the critical analysis
finds limited evidence of interpretable latent chain-of-thought and significant
problems. The program must not present recurrence as a free win.

**Depends.** nothing; explicitly after WS3/WS4.
**Acceptance evidence.** its own pre-registration if and when started; until
then, this item is a placeholder that prevents the idea from being smuggled in
as an assumption.
**Status.** deferred.
**Effort.** unknown by design.

---

## 6. WS6 — Success criteria and honest evaluation

### P6.1 — Neural-capacity advantage

**What.** Pre-register a primary capability composite and require a **3–5%
relative** improvement over the strongest matched baseline, with paired CIs and
no major domain regression. The baseline set is: dense control, dense distilled
control, conventional MoE, memory-only, memory+MoE.

**Why.** A win against a weak baseline is not a win. The dense distilled control
is the one that separates "conditional computation" from "distillation".

**Depends.** P3.4, P3.5.
**Acceptance evidence.** the pre-registration (thresholds fixed before
measurement); the evaluation receipt with per-domain results and paired CIs;
`lm-evaluation-harness` integration or a documented reason it cannot be used.
**Status.** open.
**Effort.** L.

### P6.2 — Physical-execution advantage

**What.** Report the end-to-end frontier against strong offloading baselines at
the **same GPU memory**: quality/latency Pareto, TPS, TTFT, p50/p95/p99, cold
bytes/token, PCIe bytes, residency, and energy if measurable.

**Why.** The mandate's success condition includes a physical win; a quality-only
result is a different (weaker) claim.

**Depends.** P4.2, P3.7.
**Acceptance evidence.** an `io`/bench receipt per configuration with all fields
above; paired interleaved A/B schedule, not a single run.
**Status.** open.
**Effort.** XL.

### P6.3 — Every comparison reports quality, bytes/token and wall-clock

**What.** Enforce the reporting triple in the receipt schema. **FLOPs alone is
not a result.**

**Why.** FLOPs are an arithmetic estimate; bytes and wall-clock are what the
deployment pays.

**Depends.** P4.4.
**Acceptance evidence.** schema validation that rejects a row missing any of the
three; a regenerated table for all existing arms showing the triple.
**Status.** partial (M1/M2 report active counts, FLOPs and some timings; the
schema does not enforce the triple).
**Effort.** M.

### P6.4 — Keep the Rust core/Atlas/converter/format/store; add a trainable one-pass model with an independently measured active union

**What.** Do not replace the existing Rust components. Add the trainable
one-pass model alongside them, and measure its active union **independently**
(the Rust ledger plus a profiler cross-check), not from the model's own
bookkeeping.

**Why.** The M1/M2 correction exists because a model's own count was wrong. The
independence is the safeguard.

**Depends.** P2.3.
**Acceptance evidence.** a profiler cross-check receipt in the style of
`results/selectors/sparse-audit-*.json` for the trained model, plus the ledger
admission record per arm.
**Status.** partial (the crates exist and are retained; the trained model does
not exist).
**Effort.** L.

### P6.5 — Pre-register before each run; keep negative results; no post-hoc thresholds

**What.** One pre-registration commit per training/eval run, before the first
step. Failed and negative runs are retained with their receipts. Thresholds are
never moved after seeing results; a change requires a dated amendment that says
it was made without knowledge of the outcome, or the result is reclassified as
exploratory.

**Why.** This is the only defence against the failure mode the M2 correction
documents: a number that looked fine until the instrument was audited.

**Depends.** none.
**Acceptance evidence.** the pre-registration commit ids referenced in every
result doc; a negative-results register; the `perf_numbers` court failing on a
hand-written measured number.
**Status.** partial (M1/M2/T1 followed this; it must continue).
**Effort.** S per run.

---

## 7. Sequencing

```mermaid
flowchart TD
    A[P2.1 P2.3 accounting corrections] --> B[P2.2 capacity ladder]
    B --> C[P3.1 memory path]
    B --> D[P3.2 expert path]
    C --> E[P3.6 cold start]
    D --> E
    E --> F[P3.3 error-directed router]
    F --> G[P3.4 five-arm pilot — NOW]
    G --> H{hybrid wins}
    H -->|no| I[record negative and stop WS3 scaling]
    H -->|yes| J[P3.5 phases C and D, P4.1 P4.2]
    J --> K[P3.7 prefetch predictor]
    K --> L[P4.3 P4.4 P4.5]
    L --> M[WS5 mechanisms]
    M --> N[P6.1 P6.2]
```

Hard order:

1. **P2.x** — accounting must be right before anything is measured.
2. **P3.1/P3.2/P3.6** — the capacity modules and their initialisation.
3. **P3.3/P3.4** — the pilot. Nothing in WS5 starts until this reports.
4. **WS4** — placement, only for a bank that P3.4 justified.
5. **WS5** — each mechanism gated by its own kill criterion.
6. **P6.x** — the evaluation regime runs throughout; P6.5 gates every run.

## 8. NOW — the single next experiment

**Run the five-arm pilot of P3.4 (B-like experts vs parameter-matched dense
adapter, with memory-only, naive hybrid and error-directed hybrid), at the
smallest pre-registered token budget that can distinguish them, immediately
after the P2.4 teacher-target instrument check on a held-out sample.**

Why this one, and not something cheaper:

- The M2 negative result already closed the *post-hoc inference-only* branch.
  The remaining decisive question is whether **trained conditional capacity**
  beats **dense capacity at the same active parameters, data order and compute**.
  That is a single comparison with a pre-registered decision rule, and every
  later workstream (placement, prefetch, delta compression, recurrent depth)
  spends real resources on the assumption that it can succeed.
- It is the only experiment whose outcome changes the program's direction. A
  null result retires WS3, WS4's tier-2/3 work and most of WS5; a positive
  result justifies the ladder (P2.2), the tiers (P4.2) and the mechanism matrix
  (WS5).
- The instrument check (P2.4) is folded in because a target-function artefact
  would invalidate the pilot. It costs hours, not a training campaign, and it
  must pass before the pilot's first step.

Minimum pilot requirements (otherwise the result is not interpretable): one
backbone, one data order, matched tokens and matched FLOPs, an admission record
per arm, a pre-registration commit, and paired CIs on the primary metric. If the
error-directed hybrid does not beat the dense control, record the negative
result and do not scale the bank.

## 9. Explicitly out of scope for this phase

- No claim of a Pareto win over any external system until P6.2 has receipts.
- No capacity-ladder run above 16 specialists until the pilot justifies it.
- No new storage format, no new quantisation codec, no custom CUDA kernel
  (the mandate's rule: profile first).
- No recurrent-depth work (P5.11) without its own pre-registration.
- No revision of any M1/M2 measured number. They stand as recorded, including
  the corrections of record.

---

## 10. WS7 — Design doctrine derived from the selvedge metaphor

The selvedge is a self-finished edge: the fabric's own threads, woven tighter at
the boundary so it neither unravels nor needs a hem sewn on afterwards. The
workstreams below treat that as seven *constraints on every later experiment*,
not as decoration. Each is testable, and each can fail.

These items carry no experiment of their own. Their acceptance evidence is a
contract amendment plus a checklist that later pre-registrations must satisfy.
An experiment whose pre-registration does not cite the doctrine items it obeys
is not registered.

### P7.1 — Weave selectivity in from the start; no post-hoc repair of a dense weave

**What.** Record as a standing constraint that a claim about a selection
mechanism may only be made from a model that was *trained* under that selection.
Post-hoc application of a selector to a densely trained model is a closed branch,
not an open one.

**Why.** The program measured this twice. M1 showed static whole-group omission
is far outside the registered usable tier (`results/M1-FRONTIER.md`). M2 showed a
causally *unavailable* per-token oracle at the compliant budget is still outside
the tier, and that the corrected accounting makes every strong per-token selector
physically non-compliant (`docs/M2-CORRECTION.md`). A dense weave cannot be
repaired at the boundary; it has to be woven selectively.

**Depends.** none.
**Acceptance evidence.** a contract amendment stating the rule, plus a
`doctrine:` block in the pre-registration template requiring each new
registration to name the items it satisfies; `perf_numbers` court at that
revision.
**Status.** open.
**Effort.** S.

### P7.2 — Continuity across the boundary: reusable task working sets, never forced router stickiness

**What.** Separate two things that are routinely conflated:

- **working-set residency** — a *cache policy*: which capability bytes are kept
  warm across a task's tokens;
- **per-token routing freedom** — a *model property*: whether the router is
  still allowed to change its selection inside a task.

P7.2 permits the first and forbids the second. Reuse is achieved by predicting and
preloading the same *set* of units, not by freezing the route.

**Why.** Forcing stickiness buys bytes at an unknown quality cost, and the
counter-evidence must be answered rather than ignored: Temporally Extended MoE
(arXiv 2604.20156) reports switch rates falling from over 50% to below 5% while
retaining up to 90% of base-model accuracy
(`docs/PRIOR-ART-AND-NOVELTY.md`, entry 54). Meanwhile SpecMD (2602.03921) finds
access is *not* temporally local, so residency must be predicted, not assumed.

**Depends.** none.
**Acceptance evidence.** contract amendment defining both registers; a runtime
receipt with a `working_set_residency` field distinct from per-token routing; a
court test in which a router that refuses to change selection inside a task fails
closed.
**Status.** open.
**Effort.** M.

### P7.3 — The boundary may have a different structure from the body

**What.** Permit — and by default prefer — a controller that is a different kind
of object from the substrate: deterministic hash, exact index, tiny linear head or
fixed rule, driving a large learned bank. Report controller parameters and
controller latency as separate quantities from the substrate's.

**Why.** SpecPrefetch (2607.24787) keeps a frozen native router and adds a light
separate predictor; SeqTopK (2511.06494) shifts budget with under-1% overhead. In
this program's own negative results, the only *physically compliant* M2 arms were
static masks executed through the sliced sparse path. Determinism may be a
feature of the boundary rather than a compromise.

**Depends.** none.
**Acceptance evidence.** P8.1's three-rule ladder implemented with controller
parameters and latency as separate receipt fields; a comparison showing the
controller's own cost is bounded.
**Status.** open.
**Effort.** S.

### P7.4 — The boundary must not distort what it protects

**What.** Define **puckering** as the measured degradation of the substrate caused
by the budget or the selection itself — a slice that did not fit and was
substituted, a chunk whose union was forced, a selector whose scoring read the
weights it was supposed to avoid. Every experiment must report (a) the overhead
parameters and bytes of routing and storage and (b) a **no-distortion control**:
the same substrate with the budget relaxed, so the distortion has a size.

**Why.** The M2 correction is exactly a puckering measurement: the magnitude,
frequency and learned selectors all read every gate/up channel to score groups,
which is why they could not be executed under the ceiling. Without the control,
a quality drop gets attributed to the architecture instead of the boundary.

**Depends.** none.
**Acceptance evidence.** a puckering metric definition in the contract; a receipt
pair (budgeted / relaxed) for at least one T3 arm; the routing-and-storage
overhead block present in every arm receipt.
**Status.** open.
**Effort.** M.

### P7.5 — Self-finished execution: the plan travels with its own evidence

**What.** Define a **plan** record that carries: selected unit identities with
content hashes, the capability version, the causal inputs that determined the
selection, the budget accounting (union, bytes, latency), and an integrity
witness over the whole record. A plan must replay with the teacher process and
the network absent; any hash mismatch fails closed.

**Why.** "No hidden teacher at inference" is currently a promise. Made a replay
test, it is evidence. The program already fails closed on hash mismatch elsewhere
(operator directive 9), and `selvedge-format`/`selvedge-store` already carry the
hashes; this binds them into one artifact.

**Depends.** P2.3, P6.4.
**Acceptance evidence.** a `Plan` type in `selvedge-format` with a replay test in
the `rust` court; an offline replay receipt with the network court proving
`network=none`; a documented failure case for a corrupted hash.
**Status.** open.
**Effort.** L.

### P7.6 — Composable capability protocol, with composability measured

**What.** Every capability unit declares: representation space (dimension, dtype,
layernorm convention), version, precondition on inputs, cost model (params, bytes
per token, latency), and its compatibility set. Composition is then a *measured*
claim: attach two capabilities and test whether the composition works, rather
than asserting that adapters are interchangeable.

**Why.** "Capability library" language is empty without an ABI. LoRA-style
adapters are routinely assumed composable and routinely are not; the program must
be able to say which pairs compose and to show the matrix.

**Depends.** P7.5.
**Acceptance evidence.** the protocol as a Rust type; a compatibility matrix over
at least six pairs with a pre-registered threshold fixed before the matrix is run;
a receipt naming every pair's outcome, including failures.
**Status.** open.
**Effort.** M.

### P7.7 — The formal objective, with the three quantities kept apart

**What.** Write the optimisation as:

```
maximise   capability(x)
subject to |union of active weights| <= 1e9
           latency(x)                <= L
           resident memory(x)        <= M
```

and require every arm to report **active**, **transferred** and **latency** as
three distinct quantities. No scalar "effective parameters" is permitted in a
comparison table.

**Why.** The single most effective way to make a capacity claim unfalsifiable is
to fold residency and transfer into one number. The program's own M1/M2
correction exists because a collapsed count hid a real cost.

**Depends.** P4.4, P6.3.
**Acceptance evidence.** contract amendment with the objective written out; schema
validation that rejects a row in which any of the three is missing; regenerated
tables for the T2 arms showing the triple.
**Status.** open.
**Effort.** S.

---

## 11. WS8 — Complementarity and two-timescale organization

T2 asks whether conditional capacity beats dense capacity *per token*. WS8 asks a
different question: whether two capabilities that each help can be **combined**,
and whether the combination is organised at the task level rather than improvised
per token. The T2 `H`/`H*` arms answer neither, because they select per token with
a single flat router.

### P8.1 — Task-level D/M selection on fresh held-out tasks

**What.** Select between the dense-adapter capability (`D`) and the memory
capability (`M`) **per task**, in three pre-registered regimes, in this order:

1. a **fixed rule** (no learned parameters: e.g. longest-match / OOD score);
2. a **trained small router** over task-level features;
3. a **non-deployable per-problem oracle** as the bound.

The task set must be fresh: a new split, a new contamination receipt, and no
reuse of any T2 evaluation item.

**Why.** Task-level selection is the cheapest form of the boundary doctrine
(P7.2/P7.3) and it is measurable without a runtime. The oracle separates "the
selector is bad" from "the capabilities are not complementary".

**Depends.** P12.1 (T2's verdict, so the capabilities are the ones that survived),
P12.2.
**Acceptance evidence.** a fresh data-order receipt and contamination receipt with
thresholds fixed before measurement; three arm receipts (rule / router / oracle);
a decision rule in the contract stating what each ordering means.
**Status.** open.
**Effort.** L.

### P8.2 — Compositional residuals with a bounded learned gate

**What.** Instead of loading both *paths* (T2's `H`), load both learned
**residuals** and combine them with a gate bounded in both magnitude and effect:
`y = S + a·r_D + b·r_M`, with `(a,b)` produced by a small gate and constrained
(e.g. a per-token L2 budget on the perturbation). Report: fresh union accounting,
an **interference test** (does adding `M` change the output that `D` alone would
have produced, beyond the gate's declared budget?), and four references — `D`
alone, `M` alone, gated composition, per-problem oracle.

**Why.** T2's hybrid may fail through interference rather than routing. A bounded
gate makes the interference a quantity instead of a story, and the four references
are what make a composition result interpretable.

**Depends.** P8.1.
**Acceptance evidence.** four receipts on the same fresh tasks; an interference
metric with a pre-registered bound; the gate's own parameters counted in the union.
**Status.** implemented and pre-registered (`docs/EXPERIMENT-CONTRACT.md` P8.2);
CPU tests `oracle/tests/test_p82_composition_gate.py` pass; the GPU runs are
blocked only on the pilot releasing the card. The entry points are
`oracle/train/composition.py --run-composition` and `--score-composition`;
launch commands are `results/T2-PLAN.md` §6.1. No measurement exists yet.
**Effort.** L.

### P8.3 — Two-timescale selector: task-level working set, token-level inside it

**What.** Hierarchical selection: at task start a working set is chosen (and its
cold bytes charged); inside the task, token-level selection is free to move
*within* the working set, with excursions outside it charged as cold.

**Why.** This is the honest version of reuse (P7.2) and the only version whose
cost is bookkeepable: every cold byte has an event that caused it. It also gives
P8.4 something to measure.

**Depends.** P7.2, P8.1, P8.2.
**Acceptance evidence.** a runtime/log receipt with per-task cold-byte events and
their causes; a pre-registered excursion policy; a comparison against a flat
token-level selector at matched cold bytes.
**Status.** open.
**Effort.** L.

### P8.4 — Sublinear cold bytes per task and p95 latency as accessible capacity grows

**What.** Pre-register a scaling test over the accessible-capacity rungs (P2.2 /
P12.8): cold bytes per task and p95 latency must grow **sublinearly** in accessible
capacity, i.e. a fitted slope below one with a confidence interval that excludes
one.

**Why.** This is the actual product claim of a selvedge: more capacity at
sublinear residency cost. It is also falsifiable in one plot, which makes it a
good early gate.

**Depends.** P8.3, P2.2.
**Acceptance evidence.** a pre-registered estimator (slope, CI method, rungs); a
receipt per rung; the fitted curve with CI, generated — not typed.
**Status.** open.
**Effort.** L.

### P8.5 — Task-coherent specialization versus vocabulary specialization

**What.** Train two expert banks under the same budget: one specialised by
**domain/task**, one specialised by **vocabulary/token statistics**. Compare
routing consistency (does the same task stay on the same working set?), cold bytes,
and verified quality.

**Why.** Coherence is the property the boundary doctrine assumes and nobody has
measured at this scale in this program. If vocabulary specialisation wins, the
task-level story in P8.1–P8.4 is built on sand and must be revised.

**Depends.** P8.3.
**Acceptance evidence.** two bank receipts with matched active counts; the routing
consistency metric computed from logged routes; a pre-registered winner rule.
**Status.** open.
**Effort.** M.

---

## 12. WS9 — Deficit-directed capability construction

The doctrine says the boundary must be *self-finished*. That implies the program
must be able to **construct** capabilities from verified deficits rather than
hoping a bank of trainable modules happens to contain one. WS9 is that
construction pipeline, and it is where the teacher is allowed — offline.

### P9.1 — Counterfactual capability compiler

**What.** An offline pipeline that, given a student and a task suite with
executable verifiers:

1. enumerates failures (counterfactually: same item, with and without candidate
   capacity);
2. compiles a teacher correction for each failure into a capability candidate;
3. **retains** a candidate only if it transfers to *new* tasks in the same
   capability family.

**Why.** Without the retention test the compiler produces memorisation of the
discovery set, and "capability construction" becomes a synonym for a bigger
training set. The transfer requirement is the whole item.

**Depends.** P12.9 (a verifiable domain with fresh items); a synthetic verifiable
family may be used first.
**Acceptance evidence.** a compiler receipt with discovered / compiled / retained
counts taken from the compiler's own log, a retention rate with a paired CI, and
the retained set's transfer receipt on unseen tasks.
**Status.** open.
**Effort.** XL.

### P9.2 — Three representations compete under one error-driven objective

**What.** One objective, three representations of the same correction:

- **latent correction memory** — a stored activation-space correction keyed by
  context;
- **residual neural capability** — a trained low-rank/residual module;
- **verified procedural capability** — an executable, checked artefact.

Same deficits, same verifier, same budget accounting; the comparison is which
representation retains the most verified capability per stored byte and per second.

**Why.** The program currently assumes the neural representation is the right one.
It has never tested that against the two cheaper alternatives.

**Depends.** P9.1.
**Acceptance evidence.** three arms on the same retained deficit set, with verified
success, stored bytes, and solve latency; a pre-registered selection rule.
**Status.** open.
**Effort.** XL.

### P9.3 — Combination utility and an interaction-aware controller

**What.** Define the interaction utility of a pair of capabilities for an item:

```
I(m, e; x) = R(S + m + e) - R(S + m) - R(S + e) + R(S)
```

Generate **four-condition counterfactual** training data per item (none, m only, e
only, both), train a controller to predict conditional marginal utility from
**causal inputs only**, and compare against: `H*`'s independent-utility weighting,
load-balanced routing, independent gates, and a small combinatorial oracle.

**Why.** Independent utility labels cannot represent redundancy or synergy; the
T2 `H*` labels are exactly independent utilities measured once (T2.5/T2.6 risks).
The four-condition design is the cheapest way to measure the cross term, and it is
what makes "composition" more than a weighted sum.

**Depends.** P8.2, P12.2.
**Acceptance evidence.** the four-condition dataset manifest with hashes; a
controller receipt naming its inputs and proving no label leakage; five arm
results at matched union budget; the oracle bound reported separately.
**Status.** implemented and pre-registered (`docs/EXPERIMENT-CONTRACT.md` P9.3);
CPU tests `oracle/tests/test_p9_combination_utility.py` pass on synthetic ground
truth (sign recovery, CI coverage). The GPU run needs the pilot's `H*` (or `H`)
checkpoint; launch command `results/T2-PLAN.md` §6.4. No measurement exists yet.
**Effort.** XL.

### P9.4 — Student-deficit-conditioned addressing

**What.** Address capabilities from the student's **own early hidden state**: a
bounded coarse index returns a candidate set of at most `k_c` capabilities, then a
cheap re-scorer ranks them. No scan of a large learned key table is permitted, and
every inspected learned key is counted in the union/bytes ledger.

**Why.** This is where "memory" designs usually smuggle in an uncounted cost: a
long key scan. Counting inspected keys makes the addressing cost comparable to a
router's.

**Depends.** P9.1, P9.2.
**Acceptance evidence.** a receipt with inspected-key counts per token, worst-token
union compliance, and an ablation against a flat scan showing the scan is both
slower and not better.
**Status.** open.
**Effort.** L.

### P9.5 — Teacher-derived memory is immutable and versioned at inference

**What.** Everything the teacher produced is frozen at inference, carries a
version and a content hash, is referenced by plans (P7.5), and can be rolled back.
The inference courts run with the teacher model absent and the network off.

**Why.** A capability library whose entries can drift is not auditable, and
"distilled capability" claims are otherwise untestable. This also stops a slow
leak whereby inference quietly calls a teacher for hard items.

**Depends.** P7.5.
**Acceptance evidence.** immutable-store types/tests in `selvedge-store`; an
inference receipt produced in the `network=none` court with teacher artifacts
removed; a rollback test.
**Status.** open.
**Effort.** M.

### P9.6 — Verifier-compiled procedural capabilities

**What.** Compile recurring reasoning classes into **checked executables** (a
solver plus its verifier), charge compile cost, verify cost and solve cost
separately, and compare against text retrieval of worked examples and against
latent correction memory at matched stored bytes.

**Why.** For a large class of tasks an executable is cheaper and more reliable
than added weights, and the program's own doctrine (P7.3) says the boundary may be
deterministic. This is that claim, made measurable.

**Depends.** P9.2.
**Acceptance evidence.** per-class receipts with the three cost components; a
soundness statement for each verifier (what it does and does not check); the
retrieval and latent baselines at matched bytes.
**Status.** open.
**Effort.** XL.

### P9.7 — Memory-selected recurrent reasoning

**What.** Let the memory/controller choose *how much recurrence* a problem gets,
with the recurrent block shared and the selection charged.

**Why.** It is the natural join of WS9 and P10.9/P11.14, and it is the most
expensive item in WS9.

**Depends.** P9.3 (must report first), P10.9.
**Acceptance evidence.** would be a pre-registration with matched compute, KV
handling for exited tokens, and a per-problem recurrence ledger.
**Status.** blocked — deliberately deferred until P9.3 reports; running it first
would confound "composition" with "depth".
**Effort.** L.

---

## 13. WS10 — Eighteen candidate mechanisms, each with its own cheap decisive experiment

Every item here is a *mechanism*, not a phase. The rule for the whole workstream:
no mechanism earns a second experiment until its own cheap decisive experiment has
run, and each carries a **kill criterion** fixed before the run. Items whose
novelty is already published are labelled with the ledger entry that publishes
them, and the SELVEDGE increment is stated as the accounting/measurement that is
not published.

### P10.1 — Hypernetwork-generated experts

**What.** Generate expert weights instead of storing them: shared low-rank factors
plus a compact per-capability code,

```
W_c = W_0 + U diag(z_c) V^T
```

Cheap decisive experiment: at matched **stored** parameters and matched **active**
FLOPs, compare a generated bank against a stored bank of the same size, sweeping
codebook size. Report stored params, effective executed computation and generator
overhead separately.

**Why.** It is the only way capacity can grow without growing residency, so it is
the natural continuation of the selvedge idea. Related published work: D²-MoE
(known through RFID-MoE), PuzzleMoE, DeRS.
**Depends.** P12.1.
**Acceptance evidence.** one receipt per bank size with the three cost columns and
a quality/variance test.
**Kill criterion.** kill if generated experts are **less expressive at matched
stored parameters and compute**, measured as no verified-task improvement with CIs
excluding zero, or as higher output variance at equal quality.
**Status.** open.
**Effort.** L.

### P10.2 — LatentMoE geometry: near-square experts at fixed active cost

**What.** Sweep latent width 128 / 256 / 384 / 512 with near-square expert widths,
at fixed active FLOPs and active parameters; measure local Jacobian rank, utility
diversity across experts, verified task success, VRAM, and decode overhead.

**Why.** The geometry question is **published**: LatentMoE (2601.18089) reports
accuracy per FLOP/parameter, Shaping Latent MoE (2610.14644) shows local Jacobian
rank is jointly bounded by expert input and intermediate width and that near-square
experts win at fixed per-expert parameters, and Multi-View Latent MoE (2610.40577)
shows single-router rankings disagree with expert-utility rankings. SELVEDGE's
increment is the union/bytes ledger at fixed *active* cost, and the utility-diversity
measurement as a behavioural quantity rather than a rank bound.
**Depends.** P12.1.
**Acceptance evidence.** a sweep receipt with all five measurements per point, and
the accounting triple (P7.7).
**Kill criterion.** kill the geometry axis if no latent width improves verified
success at fixed active FLOPs with a CI excluding zero.
**Status.** implemented and pre-registered as the T3 section of
`docs/EXPERIMENT-CONTRACT.md`; CPU tests `oracle/tests/test_t3_latent_geometry.py`
pass; entry points `oracle/train/latent_geometry.py --train/--report` and
`oracle/eval/baseline.py --profile t3 --latent-spec`. GPU runs blocked on the
pilot; commands in `results/T2-PLAN.md` §6.3. No measurement exists yet.
**Effort.** L.

### P10.3 — Writable fast-weight memory split into stable / episodic / working

**What.** Three tiers of writable state — **stable** (protected knowledge),
**episodic** (this session), **working** (this context) — with query-selective
writes, retention, interference tests, rollback and versioning. Target task: learn
an unseen API in one interaction and use it correctly after the document is removed.

**Why.** Test-time writing is published in several forms (Titans 2501.00663;
TTT-E2E 2512.23675; TTT-NTP 2606.21803; Select-TTT 2610.36794, which shows uniform
write schedules dilute query-relevant content). KV-Skill (2608.05475) already
covers "skill as a loadable/removable operator". The uncombined part is the
**three-tier split with rollback and versioning, and a target task whose success
requires removal of the source document**.
**Depends.** P7.5, P12.1.
**Acceptance evidence.** a before/after-document-removal evaluation, an interference
matrix across the three tiers, and a rollback receipt.
**Kill criterion.** kill if after document removal the written state performs no
better than a text-retrieval baseline at equal bytes.
**Status.** open.
**Effort.** XL.

### P10.4 — Supervised latent reasoning

**What.** A looped transformer with supervision on intermediate thinking states;
compare against explicit CoT, best-of-N, and verifier-guided search **at matched
wall-clock**.

**Why.** LOTUS (2606.31779) already reports bridging the latent/explicit gap at 3B
with parallel supervision on latents, and Thinking States (2602.08332) supervises
thinking tokens during processing. The open increment is the matched-wall-clock
comparison against search, and the cost accounting.
**Depends.** P12.1.
**Acceptance evidence.** four arms, matched wall-clock, with tokens emitted (for
CoT) counted in the cost.
**Kill criterion.** kill if latent reasoning loses to explicit CoT at matched
wall-clock on verified tasks.
**Status.** open.
**Effort.** XL.

### P10.5 — Multi-view routing and expert modulation

**What.** Route from several views (task type, uncertainty, hidden state, memory
relevance) with a modulation of expert output; count **all** router parameters;
compare against a single linear router at matched total parameters.

**Why.** Multi-view selection is being published (Multi-View Latent MoE 2610.40577;
SkillMoV; MARS 2610.23583 combines memory construction and model routing). The
uncombined part is counting every router parameter inside the same union budget as
the experts, which none of those do.
**Depends.** P8.1.
**Acceptance evidence.** a receipt with the router parameter block itemised and the
single-linear control at matched total parameters.
**Kill criterion.** kill if the multi-view router's gain disappears once its own
parameters are inside the budget.
**Status.** open.
**Effort.** M.

### P10.6 — Intrinsic-dimension-guided capacity allocation across layers

**What.** Allocate expert capacity per layer by measured intrinsic dimension (or
Jacobian rank), at matched total accessible and active parameters and FLOPs.

**Why.** Layerwise allocation from expert geometry is published (2610.14644); the
intrinsic-dimension *input* to the allocation, at matched active FLOPs, with the
union ledger, is not.
**Depends.** P10.2.
**Acceptance evidence.** a per-layer allocation receipt plus a uniform-allocation
control at matched totals.
**Kill criterion.** kill if the allocation does not beat uniform at matched totals
with a CI excluding zero.
**Status.** open.
**Effort.** M.

### P10.7 — Query-selective test-time updates on multi-stage tasks

**What.** On tasks whose relevance changes mid-episode, compare uniform writes,
query-conditioned writes, and write-then-consolidate; measure retention of the
final stage after the intermediate stages are removed.

**Why.** Select-TTT (2610.36794) establishes query-conditioned write strength with
no new parameters. The uncombined part is multi-stage relevance change and
consolidation, i.e. the working→episodic transition of P10.3.
**Depends.** P10.3.
**Acceptance evidence.** a three-arm receipt on a multi-stage suite with the
final-stage retention measured.
**Kill criterion.** kill if consolidation never beats raw selective writes.
**Status.** open.
**Effort.** M.

### P10.8 — Verifier-compiled symbolic solvers for recurring reasoning classes

**What.** Compile recurring reasoning classes into solvers with verifiers; charge
the tools; compare against a larger model solving directly at matched resources.

**Why.** Same argument as P9.6, different unit (solver vs capability). No published
work located that reports the three-way cost split at matched resources.
**Depends.** P9.6.
**Acceptance evidence.** per-class receipts and the matched-resource baseline.
**Kill criterion.** kill a class if direct solving is cheaper at equal verified
accuracy.
**Status.** open.
**Effort.** L.

### P10.9 — Mixture-of-Recursions adaptive depth with shared weights

**What.** Per-token recursion depth over a shared stack; matched compute against a
fixed-depth control; explicit KV handling for tokens that exit early.

**Why.** Published: MoR (2507.10524) assigns per-token depths and caches only the
selected tokens' KV; Huginn (2502.05171) established arbitrary test-time depth, and
the critical analysis (2507.02199) found limited evidence of interpretable latent
CoT. SELVEDGE's increment is the union/bytes accounting of the retained KV.
**Depends.** P12.1.
**Acceptance evidence.** a matched-compute receipt with KV bytes per token and a
correctness check of the early-exit path.
**Kill criterion.** kill if adaptive depth does not beat fixed depth at matched
compute.
**Status.** open.
**Effort.** L.

### P10.10 — Student-aligned reasoning distillation

**What.** Select teacher traces by *learnability* and information for this student
(hard-but-learnable), not by teacher confidence; no runtime change.

**Why.** Agent Memory Distillation (ACL 2026 findings 2407) transfers hierarchical
teacher memory into a small agent training-free; no work located that selects
traces by student-side learnability for reasoning with a measured selection effect.
**Depends.** P9.1.
**Acceptance evidence.** a selection receipt plus a random-selection control at
matched tokens.
**Kill criterion.** kill if learnability selection does not beat random at matched
training tokens.
**Status.** open.
**Effort.** M.

### P10.11 — One-token internal correctness probes

**What.** Predict correctness from a single internal read-out, calibrate it on OOD
data, gate extra computation on it, and measure **both** token savings and harmful
early stopping.

**Why.** No work matching the strategy note's "One-Token Verification" was located
(`docs/PRIOR-ART-AND-NOVELTY.md` round-15 searches); it is PROPOSED here. The
harmful-early-stopping measurement is the part that makes it decidable.
**Depends.** P12.1.
**Acceptance evidence.** an AUROC/calibration receipt on OOD splits, plus a net
token-saving count that charges the probes themselves.
**Kill criterion.** kill if the probe's net saving is not positive after charging
probe cost and harmful stops.
**Status.** open.
**Effort.** M.

### P10.12 — Conflict-directed repair

**What.** From a compiler/test diagnostic, derive a **certified failure
explanation**, backjump to the implicated decision, keep the successful parts, and
verify the soundness of the diagnostic itself.

**Why.** Classical conflict-directed backjumping plus learned repair is not
published for LLM capability selection with a verified diagnostic. Related:
MemCredit (2611.15600) uses counterfactual interventions to attribute memory
utility — the attribution machinery, not the repair loop.
**Depends.** P12.9.
**Acceptance evidence.** a repair receipt with the diagnostic's soundness check
(false-explanation rate) and repaired-success rate.
**Kill criterion.** kill if the false-explanation rate exceeds the pre-registered
bound, or if repair does not beat rerun-from-scratch at equal tokens.
**Status.** open.
**Effort.** L.

### P10.13 — Task-conditioned layer-wise expert budgets under one global budget

**What.** Allocate per-layer, per-task expert budgets under a single global budget;
compare against fixed top-k.

**Why.** Sequence-level budgets are published (SeqTopK 2511.06494); layer×task
allocation under one global budget with the union ledger is not.
**Depends.** P10.6.
**Acceptance evidence.** an allocation receipt with worst-token union compliance and
the fixed-top-k control.
**Kill criterion.** kill if fixed top-k matches the allocation at equal global
budget.
**Status.** open.
**Effort.** M.

### P10.14 — Shortcut-connected specialist execution

**What.** Short-circuit connections between specialist units so a later unit can
consume an earlier unit's state without the full path, with gates.

**Why.** ScMoE (2404.05019) publishes shortcut-connected expert parallelism for
communication overlap; the capability-side use (shortcut to skip work, not to hide
latency) is different, but it is close enough that the claim must be narrow.
**Depends.** a profitable specialist bank (P12.1 verdict) **and** a runtime
(exists only after P4.2/P6.2 work).
**Acceptance evidence.** would be a receipt with changed dependencies explicitly
re-tested: any shortcut that alters the computation graph invalidates earlier
parity tests, and those tests must be re-run and named.
**Kill criterion.** kill if shortcut skipping does not reduce bytes/token at equal
verified quality.
**Status.** blocked — deferred until a profitable bank and a runtime exist.
**Effort.** L.

### P10.15 — Information-density-aware expert compression

**What.** Compress experts by routing frequency × effective rank plus sparse
corrections; allocate bits by **verified capability loss and access cost**, not by
reconstruction error.

**Why.** RFID-MoE (2602.09316) already compresses by routing frequency and
information density with SVD. The uncombined part is the allocation objective:
verified capability loss per byte, and the access cost of the correction path.
**Depends.** P2.3.
**Acceptance evidence.** a bit-allocation receipt plus an equal-bytes reconstruction-
error-optimal control.
**Kill criterion.** kill if the capability-loss allocation does not beat the
reconstruction-error allocation at equal bytes.
**Status.** open.
**Effort.** M.

### P10.16 — Learned reasoning-length control

**What.** Learn to stop or extend reasoning per item under a budget.

**Why.** Needs a reasoning process whose extra depth is worth controlling; the
program does not yet have one that beats its own baselines.
**Depends.** P10.4 or P10.9 must show depth helps.
**Acceptance evidence.** would be a receipt with matched total compute and a
harmful-stop count.
**Kill criterion.** kill if the controller's net verified success is not above fixed
length at matched compute.
**Status.** blocked — deferred until useful deeper reasoning exists.
**Effort.** M.

### P10.17 — Fast-weight product key memory as a writable episodic store

**What.** Use a product-key memory as a *writable* episodic store, against a static
hashed memory as the control, protecting durable knowledge with rollback and
versioning.

**Why.** PEER (2407.04153) and Monet (2412.04139) publish product-key retrieval
over very many tiny experts; Titans/MIRAS publish learned test-time memory. A
writable product-key store with three-tier protection is not located; the closest
named work (CNN with large memory layers, 2101.11685) is static.
**Depends.** P10.3.
**Acceptance evidence.** a two-arm receipt (static vs writable) with interference and
rollback tests.
**Kill criterion.** kill if the writable store does not beat the static hashed store
on the retention task at equal bytes.
**Status.** open.
**Effort.** L.

### P10.18 — Optimizer court: FOAM vs Muon vs tuned AdamW

**What.** On the residual-training workload only, compare optimisers by **loss per
GPU-second and per joule**; the result must never be used to explain an architecture
difference.

**Why.** The program's architecture comparisons are only as good as the optimiser
being equally tuned. FOAM could not be located by the recorded searches as an
optimiser (`docs/PRIOR-ART-AND-NOVELTY.md` §10), so the FOAM arm is conditional:
if it remains unfindable, the court runs Muon vs tuned AdamW and records FOAM as
PROPOSED.
**Depends.** none (CPU-scheduled; GPU court when the pilot is idle).
**Acceptance evidence.** a receipt with the same workload, three seeds, per-step
loss curves, GPU-seconds and joules, and a tuning protocol fixed before the run.
**Kill criterion.** no kill; the court's output is the tuning used by every later
comparison, and it is reported as a control, not as a result.
**Status.** open (gated on GPU availability — the T2 pilot currently owns the GPU).
**Effort.** M.

---

## 14. WS11 — Twenty-one original mechanism proposals

These are the program's own hypotheses. For each, the ledger search did not locate
the mechanism (`PROPOSED`), and the item states the **minimal decisive
experiment** — the smallest run that can falsify it. A proposal without a decisive
experiment is not listed here.

### P11.1 — Causal Bottleneck Repair

**What.** Find, offline and layer-wise, a **bounded intervention** that repairs a
verified failure (a layer-wise oracle over bounded edits), then learn a predictor
that proposes the same intervention at inference. The control is an equal-budget
distributed LoRA that is *not* causal.

**Why.** Causal attribution exists (attribution-based sparse activation, PMC
13570667); repair-by-intervention with a bounded budget and an equal-budget
non-causal control is not published. The control is what makes "causal" mean
something.
**Decisive experiment.** 200 verified failures, one repair budget, three arms
(oracle intervention, learned predictor, distributed LoRA control).
**Kill criterion.** kill if the learned predictor does not beat the equal-budget
non-causal control on repaired successes.
**Depends.** P12.9. **Status.** implemented and pre-registered
(`docs/EXPERIMENT-CONTRACT.md` P11.1); CPU fixture tests
`oracle/tests/test_p11_causal_repair.py` pass. The 0.6B run is blocked on the
failed-item artifact and on the GPU; command in `results/T2-PLAN.md` §6.2. No
measurement exists yet. **Effort.** XL.

### P11.2 — Capability Operator Algebra

**What.** Represent capabilities as shared low-rank **nonlinear operators** plus
bounded programs; measure non-commutativity `[A,B]` and solve held-out operator
compositions. Compare additive MoE, sequential composition, and an equal-budget
dense adapter.

**Why.** DAG-MoE (PMLR v306) composes expert outputs structurally; an algebra with
measured non-commutativity and a held-out composition test is not located.
**Decisive experiment.** 8 operators, all pairs measured, then all length-2
compositions tested on unseen item types.
**Kill criterion.** kill if sequential composition is not better than additive
aggregation on held-out compositions.
**Depends.** P9.2. **Status.** open. **Effort.** L.

### P11.3 — Functional Innovation Storage

**What.** Admit capacity only for **verified functional novelty** relative to the
existing bank; report marginal verified success per stored byte. Compare against
redundancy-penalty baselines.

**Why.** Deduplication is studied (PuzzleMoE bit-packing, MoTE); a *verified novelty
admission test* as the storage gate is not, and it directly expresses the selvedge
idea.
**Decisive experiment.** 100 candidate capabilities, admitted vs rejected, with the
bank's verified success curve per byte.
**Kill criterion.** kill if the novelty gate does not beat a redundancy penalty at
equal stored bytes.
**Depends.** P9.1. **Status.** open. **Effort.** L.

### P11.4 — Counterfactual State Transplants

**What.** Learn a transformation that maps a student hidden state **toward a
causally validated state** (one whose downstream answer is verifiably correct),
not toward raw teacher features. Compare against token-logit KD and feature
matching.
**Why.** Feature-matching KD is standard; the *causal validation of the target
state* is the difference, and it is cheap to test: transplant and verify.
**Decisive experiment.** one layer, three arms, 500 verified items.
**Kill criterion.** kill if causal-state training does not beat feature matching at
equal tokens.
**Depends.** P9.1. **Status.** open. **Effort.** L.

### P11.5 — Representation Impedance Matching

**What.** Bridge fixed stored capability into the student through four ascending
bridges (linear → nonlinear → causal-state → outcome-trained); penalise bridges
that reconstruct teacher features but fail tasks.
**Why.** "Impedance matching" as a named object with an explicit
reconstruct-but-fail penalty is not located. Frozen Memory Is Not Enough
(2608.17050) shows the frozen memory is not the whole story when moved across
backbones, which is the same gap from the memory side.
**Decisive experiment.** four bridges, one stored capability, task success vs
feature reconstruction reported together.
**Kill criterion.** kill a bridge level if its task success does not exceed the
level below at equal stored bytes.
**Depends.** P11.4. **Status.** open. **Effort.** M.

### P11.6 — Error-Syndrome Addressing

**What.** Diagnose the failure **type** from internal state (lost dependency,
quantifier confusion, broken invariant) and route repair operators by syndrome;
then test whether the same syndrome transfers across domains (Rust lifetime vs
math dependency vs quantifier).

**Why.** Task-level routing exists (SEAP); syndrome-level diagnosis with a
cross-domain transfer test is not, and it is the only version that can be called a
*mechanism* rather than a domain classifier.
**Decisive experiment.** three domains × three syndromes, held-out recombination.
**Kill criterion.** kill if the syndrome transfer matrix is not better than
chance off the diagonal.
**Depends.** P12.9, P11.21. **Status.** open. **Effort.** L.

### P11.7 — Residual Safety Escrow

**What.** A conservative lower-confidence gate on interventions that minimises
**harmful** interventions rather than maximising average gain; measure calibration
under distribution shift; compare against a sigmoid gate.

**Why.** Safety gating exists in serving systems; a gate whose objective is the
harm tail with a shift-calibration test is not established for capability
interventions.
**Decisive experiment.** one intervention type, two gates, an OOD split with a
pre-registered harm budget.
**Kill criterion.** kill if the escrow gate does not reduce harmful interventions
at equal accepted interventions.
**Depends.** P8.2. **Status.** open. **Effort.** S.

### P11.8 — Capability Birth and Fusion

**What.** Decide learn / reuse / compose / create for a new deficit; fuse duplicated
modules; admit by marginal verified utility, storage, training cost and
transferability.

**Why.** Module fusion and growth exist separately (net2net-style growth, expert
merging); the four-way birth decision with admission by verified marginal utility is
not located.
**Decisive experiment.** 50 deficits, four-way decision, measure decision quality
against a hindsight oracle.
**Kill criterion.** kill if the policy's regret against the hindsight oracle is not
below the best fixed policy.
**Depends.** P11.3. **Status.** open. **Effort.** L.

### P11.9 — Counterexample-Constrained Routing

**What.** Learn **incompatibility constraints** between capabilities from observed
harmful combinations, then select under those constraints; compare against
unconstrained top-k.

**Why.** Load balancing and capacity factors exist; learned incompatibility
constraints from measured harm do not.
**Decisive experiment.** one bank, induce harm pairs, then evaluate constrained vs
unconstrained at matched active set size.
**Kill criterion.** kill if constrained selection does not reduce harmful
combinations without losing verified success.
**Depends.** P11.2. **Status.** open. **Effort.** M.

### P11.10 — Inter-Layer Shortcut Capsules

**What.** Sparse cross-layer low-rank arcs with gates, so a later layer can read an
earlier layer's capability state directly; compare against layer-local adapters at
matched parameters and FLOPs.

**Why.** Residual-stream widening (mHC, 2609.05309 / 2608.05549) and multi-stream
attention residuals (2610.13630) are published; a *sparse, gated, cross-layer
low-rank capsule with matched-parameter layer-local controls* is the narrower
uncombined form, and it must be claimed narrowly.
**Decisive experiment.** one model, matched parameters, three arc densities.
**Kill criterion.** kill if arcs do not beat layer-local adapters at matched
parameters and FLOPs.
**Depends.** P12.1. **Status.** open. **Effort.** M.

### P11.11 — Temporal Capability Leasing

**What.** Ephemeral task-scoped low-rank workspaces with bounded lifetime, rollback
and versioning; compare against permanent specialists.

**Why.** Test-time adaptation exists (TTT family); *leasing* — a workspace whose
expiry is part of the contract, with rollback and versioning — is not, and it is
the natural unit of a selvedge boundary.
**Decisive experiment.** same task stream with leased vs permanent specialists, at
matched total training and storage.
**Kill criterion.** kill if leasing loses verified success at equal total cost.
**Depends.** P10.3. **Status.** open. **Effort.** M.

### P11.12 — Failure-Directed Latent Repair

**What.** Map `(failed hidden state, verifier signal)` to a **repaired state**,
decode and verify; compare against text-feedback repair by measured GPU-seconds.

**Why.** Self-repair with textual feedback is common; repairing the *state* with a
verifier-trained map and charging GPU-seconds against text feedback is not located.
**Decisive experiment.** 300 verified failures, two repair arms, GPU-seconds and
repaired-success.
**Kill criterion.** kill if latent repair is not cheaper per repaired success than
text-feedback repair.
**Depends.** P12.9. **Status.** open. **Effort.** L.

### P11.13 — Predictive Capability Checkpoints

**What.** Reuse verified subproblem states with a cheap applicability test and
recomputation fallback; explicitly distinguish from KV caching and semantic caching.

**Why.** Semantic caches exist (vCache; "From Exact Hits to Close Enough"; no
SemPIC located). A checkpoint of a *verified subproblem state* with an applicability
test and a fallback is a different object, and the distinction must be argued with a
non-cache baseline.
**Decisive experiment.** one task family with frequent subproblems; hit rate, verified
correctness on hits, and recomputation cost.
**Kill criterion.** kill if hits are not verifiably correct at a rate above the
pre-registered bound.
**Depends.** P12.9. **Status.** open. **Effort.** M.

### P11.14 — Reasoning State Compression

**What.** Compile an 8-iteration internal trajectory into a single operator with
fallback to the trajectory; measure latency at matched success.

**Why.** Loop distillation exists in several forms; the fallback-with-verification
contract and the latency-at-matched-success comparison are the measurable part.
SkillCache (2611.10022) compiles *orchestration* from one trace, not a reasoning
trajectory into an operator.
**Decisive experiment.** one looped model, 8 iterations, three arms (full, compiled,
compiled+fallback).
**Kill criterion.** kill if compiled execution does not match success at lower
latency.
**Depends.** P10.4. **Status.** open. **Effort.** M.

### P11.15 — Precision-Selective Intelligence

**What.** Allocate numeric precision by measured causal effect on verified solving;
report verified success per physical byte.

**Why.** Mixed-precision and per-tensor quantisation are routine; allocating
precision by *causal effect on verified solving* rather than by sensitivity proxies
is not, and the reporting unit (success per physical byte) is the honest one.
**Decisive experiment.** one model, 3 precision profiles, verified success and bytes.
**Kill criterion.** kill if causal allocation does not beat sensitivity-based
allocation at equal bytes.
**Depends.** P2.3. **Status.** open. **Effort.** M.

### P11.16 — Adversarial Capability Crystallization

**What.** Build minimal pairs that share surface form but require different
operations; test unseen **recombinations**.

**Why.** Contrastive and counterfactual data exist; the recombination test is the
only way to distinguish "learned the distinction" from "learned the template", and
it is the anti-overfitting instrument for every other item here.
**Decisive experiment.** 500 minimal pairs, 200 held-out recombinations.
**Kill criterion.** kill a mechanism if its gain vanishes on recombinations.
**Depends.** P12.9. **Status.** open. **Effort.** M.

### P11.17 — Causal Computation Auctions

**What.** Bid for computation by predicted marginal reward per second and per byte;
compare against fixed top-k and cost-aware routing.

**Why.** Cost-aware routing exists; an auction whose currency is predicted *marginal
verified reward* per second and per byte is not located, and it makes the objective
of P7.7 the literal selection rule.
**Decisive experiment.** one bank, three arms, verified success vs wall-clock Pareto.
**Kill criterion.** kill if the auction is not on the Pareto frontier of the other
two.
**Depends.** P9.3. **Status.** open. **Effort.** M.

### P11.18 — Adaptive Verification Depth

**What.** Escalate verification only when expected value exceeds cost; measure
correct answers at matched total time.

**Why.** Verification is usually fixed-depth; making depth a decision with an
expected-value rule, and reporting correct-at-matched-time, is the measurable
version.
**Decisive experiment.** one task suite, fixed vs adaptive depth, matched wall clock.
**Kill criterion.** kill if adaptive depth does not beat fixed at matched time.
**Depends.** P9.6. **Status.** open. **Effort.** M.

### P11.19 — Neural State Transactionality

**What.** Overlays with commit / rollback / expiry / compatibility checks for
test-time updates.

**Why.** Interface-level definitions of agents are everywhere; *transactional*
semantics for test-time weight or memory updates, with compatibility checks against
in-flight plans (P7.5), are not established in this literature.
**Decisive experiment.** two concurrent tasks updating one store, with commit/rollback
injected at pre-registered points.
**Kill criterion.** kill if transactional updates cannot be implemented without
losing measured throughput on the uncontended path.
**Depends.** P10.3, P7.5. **Status.** open. **Effort.** M.

### P11.20 — Capability Conservation Training

**What.** Add capacity under explicit **regression constraints on frozen behaviour**,
so competence is retained across the board instead of trading domains.

**Why.** Regularisation toward the base model exists (KL anchoring); a breadth-of-
competence constraint measured as a *conservation* property with per-skill retention
floors is not, and it is the only defence against the classic specialist trade-off.
**Decisive experiment.** one bank, with and without conservation, on a breadth matrix
with pre-registered retention floors.
**Kill criterion.** kill if conservation costs more new-capability gain than it saves
in retention.
**Depends.** P9.1. **Status.** open. **Effort.** M.

### P11.21 — Anti-superficial-specialization suite

**What.** Rust lifetime errors, mathematical dependency problems and logical-quantifier
problems engineered so the **same underlying operation** is required. A win requires
the same operator to generalise across all three.

**Why.** This is the program's anti-hindsight instrument: it makes "domain
specialisation" insufficient, and it is the acceptance test for P11.6, P10.10 and any
capability built by P9.1. No such suite is published, and it must be built and
validated (each item checked by hand to require the shared operation) before use.
**Decisive experiment.** the suite itself: 3 domains, ≥ 40 items each, with an
operation annotation and an inter-annotator agreement check.
**Kill criterion.** the suite is rejected if a competent solver cannot be shown to use
the shared operation on ≥ 90% of items in a documented sample.
**Depends.** P12.9. **Status.** the trio is implemented and operation-annotated
(`docs/CAPABILITY-COURT.md`); P11.21's per-domain item minimum and
inter-annotator agreement remain open. **Effort.** L.

---

## 15. WS12 — Staged program and frontier claim

WS12 is the staging, the control matrix, the thresholds and the claims. It exists
because a frontier claim assembled after the fact is not a claim.

### P12.1 — Finish T2 and preserve its verdict; audit the routed-overflow validity mask

**What.** Let the five-arm pilot finish, apply K15–K19 mechanically, publish
`results/T2-PILOT.md`, and preserve the verdict whatever it is. In parallel, audit
one specific implementation hazard: the **routed-overflow validity mask applied
after restoring token order**. Then reproduce the affected arm under an independent
reference implementation and document any correction **separately** from the
registered comparison.

**Why.** A capacity factor that silently drops or re-routes tokens produces a
result that is internally consistent and wrong. The audit is cheap; the
post-mortem is not.
**Depends.** nothing (the pilot is running).
**Acceptance evidence.** the pilot's decision receipt and published report; an audit
note naming the mask's semantics and a test that the mask is applied to the right
tokens; a reference-implementation receipt; a dated correction entry in
`docs/WORK-LOG.md` if anything is wrong, explicitly not folding into T2's registered
numbers.
**Status.** running (pilot `running`: `D` and `M` evaluated, `X` evaluated, `H` and
`H*` pending). **Audit outcome (2026-11-15, revision `642a32b`): the defect is
real and reproduced.** `valid` is built in capacity-sorted pair order and applied
after the restore, so pair `p` is gated by another pair's validity whenever an
expert overflowed; an overflowing pair is retained and reads a clamped buffer row
that can hold another token's activation. The finished routed arm's own log shows
the overflow is routine. The correction is `mask_order="pair"`, default **off**,
recorded in every run receipt; the registered arms and thresholds are untouched.
See `docs/P12.1-DISPATCH-AUDIT.md` and
`results/t2/dispatch-audit-20261115T234953Z.json`.
**Effort.** S (audit) + M (reference implementation).

### P12.1a — The corrected-dispatch sensitivity run (successor to P12.1, registered separately)

**What.** Two steps, both GPU, neither replacing a registered number:

1. **evaluation only** — evaluate the trained `X` checkpoint with
   `--mask-order pair` and compare the receipt to `results/eval/t2-X.json`;
2. **training** — the T2.8 replication tier (`H*`, `H`, `X` at one quarter of the
   budget, three seeds) with `--mask-order pair`, published as **V2**.

**Why.** The audit measures that the overflow happened and how many pairs it
mis-assigned, not whether the registered comparison moved. Only a corrected run
can answer that, and the decisive pair is cheap.
**Depends.** P12.1, T2's published verdict. **Acceptance evidence.** two
receipts (evaluation, V2 training) with `dispatch_mask_order: pair`, and a
decision note that states whether the registered verdict survives, in the
registered words, without editing it. **Status.** open (blocked on the GPU the
pilot owns). **Effort.** M.

### P12.2 — Diagnose H*'s target: leave-one-out, per-expert normalisation, deployed-budget utility

**What.** Three diagnostics on the existing `H*` labels, none requiring a new run:

1. **leave-one-out** utility with the full bank, versus the deployed top-4 read;
2. **per-expert normalisation** that removes absolute utility, leaving only relative
   preference;
3. an **interaction-aware deployed-budget utility target** for the successor
   experiment.

**Why.** `u_e` is measured once in the full-bank reference set and the router is
frozen (T2.5/T2.6 risks). If the full-bank utility and the deployed-budget utility
rank experts differently, `H*` is optimising the wrong target and its verdict is
about that target, not about utility routing.
**Depends.** P12.1.
**Acceptance evidence.** one diagnostic receipt with the rank correlation between
the full-bank and deployed-budget utilities per layer, and the target definition
written into the successor pre-registration.
**Status.** tooling complete, measurement blocked on `H*`'s Phase U.
`oracle/eval/t2_diagnostics.py` implements all three diagnostics and is tested on
synthetic ground truth; the label artifact does not exist yet, so no gap number
is claimed. The corrected target is registered as **P12.17**.
**Effort.** M.

### P12.3 — T3: latent-geometry court (P10.2)

**What.** The cheapest architecture test: sweep latent width at fixed active FLOPs
with the near-square expert configuration, on the T2 data order and budget.
**Why.** It changes the model's shape, not the program's machinery: no runtime, no
new storage, no new data.
**Depends.** P12.1, P12.2. **Acceptance evidence.** the sweep receipt and the decision
rule fixed beforehand. **Status.** implemented and pre-registered as the T3
section of `docs/EXPERIMENT-CONTRACT.md`; GPU runs blocked on the pilot.
**Effort.** L.

### P12.4 — T4: generated-weight versus stored-weight court (P10.1)

**What.** Generated bank against stored bank at matched stored parameters and active
FLOPs.
**Depends.** P12.3 (it reuses the geometry that T3 selects). **Status.** open.
**Acceptance evidence.** the matched-storage receipt and the expressivity test.
**Effort.** L.

### P12.5 — T5: fast-weight adaptation court (P10.3 / P10.7)

**What.** The one-interaction API task with document removal, and the multi-stage
relevance-change variant.
**Depends.** P12.1, P7.5. **Status.** open. **Acceptance evidence.** the removal test
and the interference matrix. **Effort.** XL.

### P12.6 — T6: latent-reasoning court (P10.4 / P11.14)

**What.** Latent reasoning against CoT and search at matched wall-clock, then
reasoning-state compression with fallback.
**Depends.** P12.1. **Status.** open. **Acceptance evidence.** four matched-clock arms
and the compression receipt. **Effort.** XL.

### P12.7 — T7: the joint court, then capacity scaling

**What.** Combine **only winners** of T3–T6; then measure capacity scaling and cold
residency behaviour, including the sublinearity test (P8.4).
**Why.** Combining mechanisms before each has an individual verdict is how
uninterpretable systems get built.
**Depends.** T3–T6 each reporting. **Status.** blocked — waits on T3–T6.
**Acceptance evidence.** a joint receipt with the individual-attribution ablation
(each mechanism removed in turn). **Effort.** XL.

### P12.8 — Capability court, three levels

**What.**

- **Level 1 — neural mechanism:** does a mechanism improve verified solving at matched
  cost?
- **Level 2 — physically executable capacity scaling** at roughly 1B → 4B → 8B → 16B
  accessible capacity with a fixed active budget;
- **Level 3 — a useful offline local assistant**: does the assembled system do work a
  strong local baseline cannot, measured on the fresh suite (P12.9)?

**Why.** Each level is a different claim and they fail independently. Level 2 without
Level 1 is a storage demo; Level 3 without Level 2 is a product demo with no capacity
story.
**Depends.** P12.7 (L1), P2.2 (L2), P12.9 (L3). **Status.** harness implemented and CPU-tested (`docs/CAPABILITY-COURT.md`); no level run yet.
**Acceptance evidence.** one receipt per level with the accounting triple (P7.7) and
the controls of P12.10. **Effort.** XL.

### P12.9 — Fresh Rust task suite

**What.** A verifiable suite of compiler diagnostics, failing tests, API migrations,
dependency-compatibility problems, lifetime errors, concurrency bugs and repo-scale
repairs; each item with an executable verifier, and each public-benchmark item with
its **contamination** and **verifier-quality** caveat recorded.

**Why.** Every capability item in WS9–WS11 needs a verifiable, uncontaminated domain.
Public code benchmarks are contaminated and their tests are imperfect; saying so per
item is the difference between a measurement and a leaderboard number.
**Depends.** none. **Acceptance evidence.** the suite's manifest with per-item
verifier and provenance fields; a contamination check receipt; a documented
verifier-false-positive rate on a hand-audited sample. **Status.** suite authored, hashed and contamination-checked; checkers self-tested in `docs/CAPABILITY-COURT.md`; no arm evaluated. **Effort.** L.

### P12.10 — The control matrix

**What.** One fixed matrix in which every row reports the same ten fields:

| # | row |
| ---: | --- |
| 1 | frozen student |
| 2 | equally trained dense adapter |
| 3 | plain text retrieval |
| 4 | Memory Grafting-style latent baseline (2605.20948) |
| 5 | conventional MoE |
| 6 | naive hybrid |
| 7 | utility-trained joint hybrid |
| 8 | matching library without routing optimisation |

Reported per row: training tokens, optimiser FLOPs, offline teacher construction
cost, total accessible capacity, active unique parameters, GPU residency, measured
I/O, task success (with paired CIs).

**Why.** A capacity claim is only as honest as its row set. Row 4 and row 8 are the
two rows most often missing and most able to kill a result.
**Depends.** P12.9, P7.7. **Acceptance evidence.** one generated table, every cell
traceable to a receipt; schema rejects a row missing any field. **Status.** schema implemented and CPU-tested; no row measured. **Effort.** L.

### P12.11 — Frontier thresholds, fixed before any frontier run

**What.**

- **≥ 10% relative** improvement on a pre-registered verified-task composite, with
  paired confidence intervals;
- a **capacity-scaling curve** at approximately fixed active compute;
- **no hidden teacher at inference** (proved by the replay court, P7.5);
- a **physical Pareto** against measured latency, residency and cold bytes;
- comparison against a strong 3B–8B quantised local baseline;
- a **separate research branch (1B)** and **product branch (3B–8B)**, never pooled.

**Why.** Pre-registered thresholds are the only defence against the frontier being
drawn after the fact. The split branches exist because a 1B research result cannot
be presented as a product result.
**Depends.** P12.10. **Acceptance evidence.** the thresholds in the contract before
the first frontier run; receipts for each bullet; the two branches reported
separately with their own baselines. **Status.** thresholds pre-registered in `docs/capability-preregistration.json`; no frontier run. **Effort.** M.

### P12.12 — Three distinguishable achievement claims

**What.** Maintain three separate claim statements and never merge them:

1. **neural architecture frontier** — a mechanism beats matched controls on verified
   tasks;
2. **local inference frontier** — the assembled system beats a quantised local
   baseline at equal residency;
3. **system capability frontier** — the system completes work of a declared class
   offline.

**Why.** Merging them is how a paper claims three advances with one experiment. Each
has different evidence, and a failure in one does not transfer to the others.
**Depends.** P12.11. **Acceptance evidence.** three claim blocks in
`docs/EXPERIMENT-CONTRACT.md`, each naming its own receipts. **Status.** open.
**Effort.** S.

### P12.13 — Transfer test: equivalent capability libraries on a small and a larger backbone

**What.** Attach *equivalent* capability libraries to a small and a larger backbone
and compare the gains, so a gain can be attributed to the library rather than to the
backbone.

**Why.** Frozen Memory Is Not Enough (2608.17050) shows this is the decisive
question when memory moves across backbones. Without it, every capability result is
backbone-specific folklore.
**Depends.** P9.1, P9.2. **Acceptance evidence.** one receipt pair with matched
libraries and the gain difference with CIs. **Status.** open. **Effort.** L.

### P12.14 — Wildcards: mHC/xHC multi-stream residual capacity, low-rank logits basis

**What.** Two wildcards, each with its own cheap test:

- **mHC / xHC multi-stream residual capacity.** mHC is published and analysed: the
  four-stream residual pathway of a 2026 frontier model effectively uses about two
  streams with depth-dependent routing (2609.05309), and the manifold-constrained
  variant appears in 2608.05549; multi-stream attention residuals are published in
  2610.13630. For **xHC**, no source was located (`UNSUPPORTED`). The cheap test: add
  streams to the student at matched active parameters and measure verified success
  and effective stream count.
- **low-rank logits basis.** No source matching the strategy note's "low-rank logits
  (ICLR 2026)" was located by the recorded searches (`UNSUPPORTED`); treat as
  PROPOSED. The cheap test: factorise the output projection and test **novel-task
  transfer** — whether a basis trained on one task family helps an unseen one.

**Why.** Both are cheap shape changes with a decisive, small experiment, and both
have an honest novelty status: one published-and-narrow, one unfound.
**Depends.** P12.1. **Status.** open. **Acceptance evidence.** one receipt per
wildcard with a matched-parameter control. **Effort.** M each.

### P12.15 — Two pre-scaling technical problems, tracked as blockers

**What.** Carry exactly two known technical problems into every scaling decision:

1. **routed overflow** — the validity mask applied after restoring token order
   (P12.1);
2. **the H\* utility/deployment mismatch** — full-bank labels driving a top-4
   deployed read (P12.2).

**Why.** Both can silently invalidate a scale-up, and both are diagnosable without a
GPU run. Naming them as blockers keeps them from being discovered at 8B.
**Depends.** none. **Acceptance evidence.** an entry in the contract's risk section
per problem, each closed by a named receipt; no Level-2 (P12.8) run may be
registered while either is open. **Status.** open (both tracked; P12.1's audit is
running). **Effort.** S.

### P12.16 — Anti-pattern guard: no new implementation program before the first three steps report

**What.** A hard rule: no sweeping new implementation program may start until the
first three staged steps (T3 latent geometry, T4 generated weights, T5 fast-weight
adaptation — P12.3–P12.5) identify **which capability representation actually
improves verified problem solving**.

**Why.** The program's failure mode is breadth without verdicts: many mechanisms,
no attribution. This rule converts "breadth preserved" (operator directive 8) from a
licence to start everything into an ordering constraint.
**Depends.** none. **Acceptance evidence.** the rule in the contract, and a
`WORK-LOG` entry for any exception with the operator's reason. **Status.** open.
**Effort.** S.

### P12.17 — The successor utility target: interaction-aware, deployed-budget, absolute utility retained

**What.** Register the corrected `H*` target before the successor experiment. It has
three parts, each addressing one measured property of the current target
(`docs/T2-DIAGNOSTICS.md` §3–§5):

1. **Deployed-budget reference set.** Labels are measured against the router's own
top-`k` set, not the full bank:
   `u^dep_e(t) = L(S(t) \ {e}) - L(S(t))` for `e ∈ S(t)` and `0` otherwise, with
   `S(t) = top-k(z(t))`; a second pass measures the held-out re-selection after the
   router is frozen, so the target cannot chase its own current choice.
2. **Absolute utility retained; only the noise sets the temperature.** No per-expert
   standardisation. The router is `z_e(t) = a_e + w_e · h_t` with `a_e` initialised
   from the measured mean deployed utility (in nats) and trained; the listwise
   target is `softmax(u^dep / tau_u)` with `tau_u` fixed to the **measured label
   noise** (median per-expert standard deviation), not to 1 standardised unit. This
   keeps the magnitude the deployed decision actually reads and uses the spread as
   calibration instead of deleting it.
3. **Interaction-aware set value.** There is no additive utility for a top-`k` set,
   so measure the pairwise residual on a calibration subset,
   `I_{e,e'} = L(S \ {e,e'}) - L(S) - u^dep_e - u^dep_e'`, fit a rank-`r` model of
   `U(S) = sum_{e in S} a_e + half sum_{e != e' in S} I_{e,e'}`, and train the
   router against the *set* value with a listwise loss over candidate sets (or a
   differentiable greedy top-`k` surrogate). `I` is reported, not hidden: a near-zero
   `I` says the additive target was adequate and the miss was elsewhere.

**Why.** The current target measures the marginal value of omitting an expert from
a full bank, then asks a top-4 router to choose which experts to *add*. It uses full
standardisation, so it deletes the absolute utility, and it assumes additivity. Each
of those is measurable from the label artifact, and each has a specific failure mode
named in `docs/T2-DIAGNOSTICS.md`; the successor target is the smallest one that
addresses all three without changing the arms' geometry.
**Depends.** P12.2 (the diagnostics decide whether each part is needed: drop part 1 if
the full-bank and deployed-budget utilities rank identically, drop part 3 if `I` is
inside its noise, drop part 2's offset if the identity term is not what the router
uses). **Acceptance evidence.** the pre-registered target written into the successor
contract with its label recipe, pass count and cost; a label receipt; and a
pre-registered kill condition that says what an additive-only target would have to
achieve for the interaction term to be declared unnecessary.
**Status.** open, written before any measurement of the successor. **Effort.** M.

---

## 16. Dependency-ordered execution queue

The queue is the order in which the new work runs. An item may move earlier only if
its dependencies are already satisfied; it may not move earlier because it is
cheaper or more interesting.

```mermaid
flowchart TD
    A[P12.1 finish T2 and audit overflow mask] --> B[P12.2 H* target diagnostics]
    B --> SU[P12.17 successor target registered]
    A --> C[P7.1 P7.3 P7.5 P7.7 doctrine amendments]
    C --> D[P12.15 blockers closed]
    SU --> D
    N9[P12.9 fresh Rust suite] --> CM[P12.10 control matrix and row set]
    AC[P7.7 accounting triple] --> CM
    CM --> TH[P12.11 frontier thresholds fixed]
    TH --> D
    D --> E[P11.21 anti-specialisation suite]
    D --> G[P12.3 T3 latent geometry]
    G --> H[P12.4 T4 generated weights]
    G --> I[P10.6 P10.13 allocation]
    D --> J[P12.5 T5 fast-weight adaptation]
    J --> K[P10.17 writable PKM]
    K --> L[P11.11 leasing P11.19 transactionality]
    D --> M[P12.6 T6 latent reasoning]
    M --> N[P11.14 state compression]
    E --> O[P8.1 task-level D/M]
    O --> P[P8.2 compositional residuals]
    P --> Q[P9.3 combination utility]
    Q --> R[P9.7 recurrent selection]
    P --> TS[P8.3 two timescales]
    TS --> SB[P8.4 sublinearity P8.5 coherence]
    H --> U[P12.7 joint court]
    I --> U
    L --> U
    N --> U
    SB --> U
    U --> V[P12.8 levels 1-3]
    V --> W[P12.12-P12.14 claim wording]
```

Hard order:

1. **P12.1** — the running pilot decides what exists; nothing scales before its
   verdict and its overflow audit. **P12.1a** follows it (evaluation-only first,
   then the corrected-dispatch replication tier) and publishes a sensitivity
   note, never a replacement number.
2. **P12.2 + P12.17 + P12.15** — the target diagnostics and the successor target,
   then the two pre-scaling blockers; all diagnosable without a GPU run.
3. **Doctrine (WS7) + P7.5/P7.6** — the constraints and the plan record, because
   every later receipt must carry them.
4. **Decision criteria before any governed experiment (the P-correction): P12.9,
   P12.10, P12.11.** The suite exists; the control matrix's schema and its row
   set must be fixed, and the thresholds pre-registered, **before** the first
   experiment they judge. On 2026-11-15 this step was last (old step 8); it is
   now step 4. See §0.2.
5. **P11.21** — the anti-specialisation suite, because every WS9/WS11 item needs
   it.
6. **T3–T6 (P12.3–P12.6)** — the four cheapest architecture decisions.
7. **WS8 → P9.3** — composition, then the interaction-aware target.
8. **T7 and P12.8** — only winners, then the three levels. Infrastructure may be
   finalised here; no criterion may be.
9. **P12.12–P12.14** — the claim wording and the wildcards. Thresholds, controls
   and kill criteria are **read** at this point, never rewritten.

## 17. NOW

**The pilot is running: do not start a GPU job.** The single next *program* step is
P12.1's overflow audit plus P12.2's three H*-target diagnostics, both CPU-only, while
the pilot's remaining arm (`H*`) trains and is evaluated on the GPU. The single next
*implementation* step after the pilot's verdict is T3 (P12.3).

Tier reading (S16C, §0.1): the ACTIVE work is the T2 pilot (P12.1) and the one next
decisive test — the successor utility target P12.17 applied to a single registered
run against the dense control — plus the criteria that must exist before it
(P12.10, P12.11) and the two blockers that gate its registration (P12.15).
Everything in WS13 is CANDIDATE: defined, with a cheap decisive experiment and a
kill criterion, and **not started**. Nothing in WS13 may be registered as a run
before P12.1's verdict and the P12.15 blockers.

---

## 18. WS13 — Published mechanisms and sixteen inventions, run as competing alternatives

**Phase rule.** Each item below is a separate programme with its own registered
court. Integration happens **only** after the individual winners report
(P13.5), never as an automatic combination of plausible parts. The failure mode
this forbids is the one P12.16 names: breadth without attribution.

**Accounting rule for every item in this workstream.** The ledger counts
*reused* parameters **once per output token** (an expert read once and used at
three layers is one parameter read for that token), and counts the *FLOPs* of
every repeated use **separately**. Accessible capacity, active unique parameters,
bytes moved and cold bytes are four different numbers and are never merged
(P7.7). Bits spent on depth embeddings, routers, compensators and compilers are
counted in the *active* ledger, not the accessible one.

**Evidence status (corrected 2026-11-16, S20; from `docs/PRIOR-ART-AND-NOVELTY.md`
§12).** All four published mechanisms are verified at abstract level: MoSE
(PMLR 306), MoRE (arXiv 2609.18176), GMoE (ACL 2026 long 2065) and TYLER (EMNLP
2026 main 2004). The round-18 finding that MoSE could not be located and that
GMoE/TYLER were title-only was a **retrieval false negative** (arXiv-only search;
landing pages that had not yet served an abstract) and is superseded. P13.1,
P13.3 and P13.4 are therefore **published baselines to reproduce and extend**,
each with its own measured SELVEDGE delta and falsifying control — they are not
proposed SELVEDGE mechanisms. Of the sixteen inventions, nine have a verified
nearest neighbour, one partial, and six have none located; those remain the
novelty surface.

**Tier enforcement gate (dated statement, 2026-11-16).** MoSE/GMoE/TYLER
reproductions and the sixteen inventions (P13.6–P13.21) **do not start** until
(1) H\* has reported its result and (2) the first defensible capability
measurement exists (a registered T2 verdict, not an exploratory partial). This is
a hard gate: no workstream may begin a P13 reproduction or invention ahead of it.
Any exception must be registered in advance in `docs/WORK-LOG.md` with a written
justification (what is unblocked, why it cannot wait, what it could contaminate)
and the operator's approval; an unregistered exception is a defect.

### P13.1 — Adaptive expert widths: nested slimmable experts under one budget

**Status: published baseline to reproduce and extend** (MoSE, Tastan et al., ICML
2026, PMLR 306:119816–119836; `https://proceedings.mlr.press/v306/tastan26a.html`).
MoSE already states "nested, slimmable" experts "executed at variable widths"
under a fixed budget; SELVEDGE re-runs that baseline on its byte-accounted store
and adds the shard-aligned, per-expert allocation and its falsifying control.
**What.** Each expert is a nested stack at **25 / 50 / 75 / 100%** of its width,
trained so every prefix is usable. Width tiers must be **aligned to shard
boundaries** (the current store reads whole shards; a tier that is not
expressible as whole shards needs a partial-read path and its cost in the same
receipt, or it is not executable under the ledger). Width is allocated per
expert under one global active-parameter and I/O budget.
**Why.** A single width per expert spends the same on a rarely-needed expert as
on an always-needed one, and the T2 pilot's whole question is whether *active*
bytes can buy capability. Width turns the bank from a fixed set of experts into a
per-token allocation along an axis the reader already controls.
**Decisive experiment.** Mixed-width vs fixed-width banks at **matched bytes and
matched compute** (same active unique parameters per token, same bytes/token,
same training tokens) on the T2 instrument and the fresh suite.
**Falsifying control.** (a) a fixed-width bank at the same *mean* width; (b) a
mixed-width bank whose widths are **hash-allocated**, so any gain from mere
heterogeneity is separated from a gain from *allocated* width.
**Kill criterion.** If mixed-width does not beat both controls by the registered
margin at matched bytes and compute, in two of three seeds, the width axis is a
reallocation and not a capacity gain: retire P13.1 and record the negative.
**Depends.** P12.1, P12.9, P12.10, P14.1. **Acceptance evidence.** one paired
receipt per control with the bytes/compute equality printed; a shard-alignment
proof for every tier used. **Status.** open. **Effort.** L. **Tier.** CANDIDATE.

### P13.2 — Cross-layer expert reuse: shared pools over 2 / 4 / 7 layers

**What.** A shared expert pool spanning groups of adjacent layers, each layer
keeping its own router, plus a learnable depth embedding so a shared expert can
distinguish which layer is calling it. The comparison is pool size 2 / 4 / 7
against an **independent per-layer bank** matched on total accessible capacity,
and against MoRE's published shape at 4.
**Why.** MoRE's abstract is verified: sharing pools across adjacent layers
"expand[s] the diversity of routing combinations without additional parameters".
If that holds on SELVEDGE's byte-accounted execution, an expert can be paid for
once and used at several depths, which is the cheapest capacity this program can
buy. If it does not hold, reuse is a memory trick that costs quality.
**Decisive experiment.** Independent bank vs 2/4/7-layer pools at matched
accessible capacity and matched active bytes per token, on the fresh suite.
**Ledger requirement.** Reused parameters count **once per output token**;
repeated FLOPs count **per use**; the depth embeddings count as active.
**Falsifying control.** The independent bank at the *same* accessible capacity,
plus a pool with **no depth embedding** (which must lose if the embedding is
load-bearing).
**Kill criterion.** Retire at pool size 4 if it does not beat the independent
bank at matched accessible capacity and bytes, or if routing entropy across the
grouped layers shows the depth embedding changes nothing (measured, not asserted).
**Depends.** P12.1, P12.9, P12.10. **Acceptance evidence.** the four-arm receipt
with the reuse-aware ledger and the routing-entropy measurement. **Status.** open.
**Effort.** L. **Tier.** CANDIDATE.

### P13.3 — Global experts plus a recurrent routing state

**What.** A small set of **global** experts available at every layer, one **local**
expert per layer, and a router that carries routing state forward across layers
(a recurrent hidden state over routing decisions). Compare against MoRE-style
pools and against independent per-layer experts.
**Why.** Global experts amortise a capability that every layer needs; a recurrent
router lets a layer's choice depend on the path taken to reach it, which a
per-token independent router cannot express.
**Evidence boundary (corrected 2026-11-16, S20).** This is a **reproduction and
extension of published GMoE** (ACL 2026 long 2065; `https://aclanthology.org/2026.acl-long.2065/`),
whose abstract states a "GRU-based recurrent component shared across layers"
with layer-specific heads that "propagate routing logits across layers", plus
"Global Experts shared across all layers" and a "Local Expert per layer". The
recurrent routing state is therefore **published, not PROPOSED**. The SELVEDGE
delta to measure is the byte-accounted cost and the state-reset ablation, not
the existence of the mechanism.
**Decisive experiment.** Three arms at matched active parameters: global+local
with a recurrent router, global+local with a stateless router, and an independent
bank. Measure **useful computation** (the drop when the global experts are
removed, and the drop when routing state is reset per layer), not only parameter
duplication.
**Falsifying control.** The stateless router **at the same parameter count** — the
recurrent state's parameters must not be the gain.
**Kill criterion.** If resetting the routing state per layer changes task success
by less than the registered margin, the state is decorative: retire the recurrent
part and keep only the global/local split.
**Depends.** P13.2 (shares the pool machinery). **Acceptance evidence.** the three
receipts plus the state-reset ablation. **Status.** open. **Effort.** XL.
**Tier.** CANDIDATE.

### P13.4 — Typed latent reasoning with a learned allocation budget

**What.** A bounded set of internal operators with declared types — **planning**,
**state update**, **procedural abstraction** — applied for a variable number of
steps, with a learned allocation budget that decides which operator runs and how
much of the budget it gets.
**Why.** Ordinary recurrence applies the same operator at every step, and
explicit chain-of-thought spends output tokens. A typed operator set with a
budget is the smallest architecture that can spend *internal* compute differently
per problem.
**Evidence boundary (corrected 2026-11-16, S20).** This is a **reproduction and
extension of published TYLER** (EMNLP 2026 main 2004; `https://aclanthology.org/2026.emnlp-main.2004/`),
whose abstract states that Tyler "learns a policy" that at each step chooses to
emit a token or "invoke a typed latent operator", with operators supporting
"global planning, local state updates, or reusable procedural abstraction". The
typed operator set and the learned allocation policy are therefore **published,
not PROPOSED**. The SELVEDGE delta is the matched-latency comparison and the
budget-variance measurement.
**Decisive experiment.** Typed operators vs explicit chain-of-thought vs ordinary
recurrence, matched on **latency, learned-parameter usage and task success**, on
the fresh suite.
**Falsifying control.** Ordinary recurrence with the **same** number of steps and
the same parameter count; and a CoT arm given the same latency budget.
**Kill criterion.** If typed operators do not beat both controls in at least two
of the seven categories at matched latency, or the learned budget collapses to a
constant across tasks (measured by budget variance), retire P13.4.
**Depends.** P12.1, P12.9, P12.6 (shares the T6 instrument). **Acceptance
evidence.** the three-arm matched-clock receipt and the budget-variance
measurement. **Status.** open. **Effort.** XL. **Tier.** CANDIDATE.

### P13.5 — Integration controller (only after the individual winners report)

**What.** One controller that jointly chooses **capability** (which module),
**width** (P13.1), **depth** (P13.4 / recurrence), **reuse** (P13.2/P13.3) and
**residency** under the 1e9 active-parameter ceiling, a latency target and a
bytes-per-token budget.
**Why.** The individual items each buy one axis. A joint controller is the only
thing that can trade them, and it is also the easiest way to build an
uninterpretable system, which is why it is last.
**Acceptance evidence.** a joint receipt with a **leave-one-axis-out ablation**
(each axis removed in turn) showing the controller's gain is attributable, and no
axis whose removal is within noise.
**Depends.** every reporting item among P13.1–P13.4 and the sixteen below.
**Status.** open, deliberately unscheduled. **Effort.** XL. **Tier.** HYPOTHESIS.

### The sixteen inventions (P13.6–P13.21)

Each entry states the mechanism, why it could work, the **decisive experiment**,
the **falsifying control** (the comparison that can kill it), and the **kill
criterion** (what result retires it). None is a commitment to implement; all are
CANDIDATE with a registered court before any run.

#### P13.6 — Capability Seed Compilation

**What.** Compile a specialist **once** into a compact *task seed* (a small set of
tensors plus a conditioning vector) that is reused across all output tokens,
instead of executing the specialist per token. Prefill counts the compiler's
parameters; inference counts the seed's. Report the **task-level causal parameter
footprint** (what the seed actually depends on).
**Why.** A 0.6B specialist executed per token is unaffordable under the ceiling;
compiled once per task it may be affordable. Task vectors (2212.04089, VERIFIED)
say a task is carried by a direction in weight space, which is the premise.
**Decisive experiment.** Four arms at matched active bytes: (a) full specialist per
token, (b) specialist resident per token, (c) learned seed, (d) prefix
conditioning only.
**Falsifying control.** Prefix conditioning alone — if a 32-token prefix matches
the seed, the seed is a prompt in disguise.
**Kill criterion.** If the seed does not beat prefix conditioning at matched bytes
while staying within the ceiling, retire.
**Depends.** P12.1, P9.1. **Acceptance evidence.** the four-arm receipt with the
causal-footprint measurement. **Status.** open. **Effort.** L. **Tier.** CANDIDATE.

#### P13.7 — Constraint-Debt Reasoning

**What.** A persistent **obligations register**: as the model plans, it records
constraints it has taken on (borrow that must end, generic bound that must be
satisfied, invariant that must hold) and carries them across steps; it is trained
against an **executable verifier** that says whether each obligation was
discharged.
**Why.** Many failures are not reasoning errors but forgotten obligations.
Reflexion (2303.11366, VERIFIED) shows verbal feedback persists across attempts;
the register makes the persistence structured and checkable rather than prose.
**Decisive experiment.** Register-equipped arm vs (a) more context and (b) more
recurrence, at matched cost; measure **constraint violations per task**.
**Falsifying control.** A register that is recorded but never read back (same
parameter count, same training signal).
**Kill criterion.** If violations do not fall by the registered margin against
both controls, or the never-read control matches, retire.
**Depends.** P12.9, P12.10. **Acceptance evidence.** the violation-rate receipt
with per-category counts. **Status.** open. **Effort.** M. **Tier.** CANDIDATE.

#### P13.8 — Metamorphic Quotient Routing

**What.** Learn an **address invariant to semantics-preserving transformations**:
renamed identifiers, reordered independent statements, reformatted output all map
to the same routing decision; deceptive **near**-equivalences (a rewrite that
changes behaviour) must map elsewhere.
**Why.** A router keyed on surface form wastes capacity re-learning the same
expert under a new name, and fails when a deceptive rewrite looks equivalent.
**Evidence boundary.** The intended citation (robustness ReCode) could not be
located; no verified prior art for this routing claim. **PROPOSED.**
**Decisive experiment.** Train with a metamorphic objective; test on **unseen**
rewrite families (not seen at training) and on a held-out deceptive set.
**Falsifying control.** An augmentation-only arm (same rewrites in training, no
invariance term) — the standard way to get this effect without the mechanism.
**Kill criterion.** If the invariant arm does not beat augmentation-only on unseen
rewrite families, retire; transfer across rust/numeric/logic is reported per
domain and a single-domain win is not a transfer claim.
**Depends.** P12.9, P14.5 (executed Rust-case coverage). **Acceptance evidence.** the
unseen-family receipt and the deceptive-set false-accept count. **Status.** open.
**Effort.** L. **Tier.** CANDIDATE.

#### P13.9 — Semantic Refinement Ladders

**What.** An ordered ladder of representational **ΔE** steps, each trained on the
*marginal verified gain* it buys, so the model chooses depth by expected benefit
per byte.
**Why.** MatFormer (2310.07707, VERIFIED) shows nested granularity works; the
missing part is spending the next increment only where a verifier says it buys
something.
**Decisive experiment.** Fixed-width vs progressive-quantised vs ladder, **at
matched bytes**, on the fresh suite.
**Falsifying control.** A ladder whose rungs are selected **randomly** at the same
expected depth — its result is the value of depth alone.
**Kill criterion.** If the ladder does not beat the random-rung control at matched
bytes, the ordering carries no information: retire.
**Depends.** P12.9. **Acceptance evidence.** the matched-byte three-arm receipt
plus the per-rung marginal-gain table. **Status.** open. **Effort.** L.
**Tier.** CANDIDATE.

#### P13.10 — Verified Constraint Distillation

**What.** Train on the **difference** between a failing and a corrected program:
extract the violated constraint, generate varied examples of that constraint class,
and distil on those rather than on correct solutions.
**Why.** Self-Debugging (2304.05128, VERIFIED) shows execution feedback teaches
repair; the difference signal is more targeted than an answer.
**Decisive experiment.** Constraint-distilled arm vs **correct-solution**
distillation at matched tokens, on **held-out structural families**.
**Falsifying control.** Correct-solution distillation at matched tokens — the
standard baseline and the one that usually wins.
**Kill criterion.** If the gain does not survive on held-out structural families
(only on the families whose constraints were extracted), retire: it memorised
constraints, not the constraint discipline.
**Depends.** P12.9, P14.5. **Acceptance evidence.** the held-out-family receipt
with per-family deltas. **Status.** open. **Effort.** L. **Tier.** CANDIDATE.

#### P13.11 — Certified Read Elision

**What.** A **conservative bound** ε on the effect of omitted computation; a read
is skipped only when the proven margin exceeds 2ε. The bound must be a **real
bound** (interval or Lipschitz-style), **not a confidence interval**.
**Why.** Every omission result in this repository's history is post-hoc and
unbounded (M1, M2). A real bound is what makes skipping *safe* rather than
*measured to be lucky*.
**Evidence boundary.** No certified-bound work could be located by the recorded
searches (CROWN unlocated). The machinery is PROPOSED until read.
**Decisive experiment.** Bound tightness measured first (fraction of tokens whose
ε is small enough to certify anything); then how many reads can actually be
certified on real held-out windows, and task success under certification.
**Falsifying control.** The **same skip rule with ε from a bootstrap CI** — if the
CI version performs identically, the "certified" claim is cosmetic and must be
dropped.
**Kill criterion.** If the certified fraction is ~0 at a useful quality level, or
the CI control matches, retire the certification and report the bound as too
loose.
**Depends.** P12.1, P12.10. **Acceptance evidence.** the tightness receipt and the
certified-read count. **Status.** open. **Effort.** L. **Tier.** CANDIDATE.

#### P13.12 — Interaction Compilation

**What.** Compile a **measured pairwise synergy** into a compact residual operator
that is executed once, rather than executing both modules.
**Why.** P9.3 measures pairwise interaction; if a pair's interaction is stable, it
can be compiled. SMEAR (2306.03745) already compiles a weighted mixture of expert
parameters into one merged expert, so the *shape* is published; conditioning the
compiled operator on a **measured interaction** is the invention.
**Decisive experiment.** Compiled-pair arm vs executing both modules, at matched
bytes and matched FLOPs, on items where P8.1/P9.3 measured positive synergy.
**Falsifying control.** Executing both modules **is** the control; a second
control compiles pairs whose interaction was measured to be **zero** (which must
show no gain).
**Kill criterion.** If compiled pairs that were measured as synergistic do not
beat executing both at matched bytes, or zero-interaction pairs gain too, retire.
**Depends.** P9.3, P8.2. **Acceptance evidence.** the compiled-vs-executed receipt
split by measured interaction sign. **Status.** open. **Effort.** L.
**Tier.** CANDIDATE.

#### P13.13 — Conditional Representation Expansion

**What.** A temporary higher-dimensional working state, opened only for selected
layers and closed again, so width is a per-token rental rather than a permanent
cost.
**Why.** CoLT5-style conditional width is the published shape (unlocated here;
PROPOSED); SELVEDGE's version is sized by the active ledger, so the rental is
paid in bytes on the same row as the rest of the token.
**Decisive experiment.** Expansion arm vs a **width-matched dense adapter** at
matched active parameters and bytes.
**Falsifying control.** The width-matched adapter; plus an arm that opens the
state at **random** layers at the same expected cost.
**Kill criterion.** If expansion does not beat the adapter at matched bytes, or
the random-layer control matches, retire.
**Depends.** P12.1, P12.10. **Acceptance evidence.** the matched-byte receipt and
the layer-selection distribution. **Status.** open. **Effort.** L.
**Tier.** CANDIDATE.

#### P13.14 — Omission-Conditioned Recovery

**What.** A compensator conditioned on **which** computation was omitted, trained
to recover the omitted contribution. Its parameters count in the **active**
ledger.
**Why.** Skipping a group changes the residual stream in a way a fixed model
cannot undo; a compensator that knows what is missing is the cheapest repair. The
baseline is the hard one: **strong layer-wise distribution alignment**, which is
exactly what GPTQ (2210.17323, VERIFIED) does for quantisation error.
**Decisive experiment.** Compensated omission vs layer-wise distribution
alignment, at matched bytes (compensator included) and matched omission rate.
**Falsifying control.** The alignment baseline; and a compensator conditioned on a
**shuffled** omission set (knows the rate, not the identity).
**Kill criterion.** If the conditioned compensator does not beat alignment at
matched bytes, or only matches the shuffled condition, retire.
**Depends.** P12.1, P4.3. **Acceptance evidence.** the matched-byte receipt with
the shuffled-condition ablation. **Status.** open. **Effort.** L. **Tier.** CANDIDATE.

#### P13.15 — Counterexample-Preserving Compression

**What.** Allocate compression bits to rare, **verifier-confirmed** distinctions
(the cases where a small weight difference changes the answer) rather than to
average logit error.
**Why.** The M1/M2 negative results say the loss is concentrated in a small
number of cases; average-error allocation (weight-error or average-logit) spends
bits where they do not matter.
**Evidence boundary.** HAWQ-style mixed precision is standard but unlocated by
the recorded searches; the *counterexample-conditioned allocation* is PROPOSED.
**Decisive experiment.** Equal-bytes comparison against (a) weight-error
allocation and (b) average-logit-error allocation; report **flip rate on the
rare distinctions** and overall task success.
**Falsifying control.** The two allocation baselines at exactly equal bytes.
**Kill criterion.** If the rare-distinction flip rate does not improve at equal
bytes, or overall success drops, retire.
**Depends.** P12.9, P4.3. **Acceptance evidence.** the equal-byte receipt with the
flip-rate table. **Status.** open. **Effort.** M. **Tier.** CANDIDATE.

#### P13.16 — Latent Branch-and-Reconcile

**What.** At a **genuine ambiguity** (measured, not guessed), fork the latent state
into branches, let them proceed, and **merge consistent findings** back into one
state.
**Why.** Sampling and voting (2203.11171, VERIFIED) pays output tokens for every
branch and cannot merge partial findings; a latent fork can.
**Decisive experiment.** Fork/merge vs sampling, vs beam search and vs more
recurrence, at **equal wall-clock cost**, on items with a detected ambiguity.
**Falsifying control.** Two independent samples reconciled by a vote at the same
cost; and a fork that is **not** merged (to show merging matters).
**Kill criterion.** If fork/merge does not beat sampling at equal cost, or the
unmerged fork matches it, retire.
**Depends.** P12.9, P12.6. **Acceptance evidence.** the equal-cost receipt and the
ambiguity-detection precision. **Status.** open. **Effort.** L. **Tier.** CANDIDATE.

#### P13.17 — Resident-Side-Information Coding

**What.** Encode **cold** modules as conditional differences against resident
weights, including decode cost, instead of storing them standalone.
**Why.** LoRA (2106.09685, VERIFIED) shows a module can be carried as a low-rank
factorisation. If cold modules are correlated with resident ones, a difference
code is smaller than the module, and cold bytes are the quantity that decides
whether capacity is affordable.
**Decisive experiment.** Difference-coded cold modules vs standalone: **NVMe
traffic per token** and quality, with decode CPU time included.
**Falsifying control.** Standalone storage of the same modules at the same
precision; and a difference code against a **random** resident basis (which must
lose).
**Kill criterion.** If NVMe bytes do not fall, or decode cost eats the gain at the
measured p95, retire.
**Depends.** P4.4, P14.1. **Acceptance evidence.** the traffic receipt with decode
cost in the same row. **Status.** open. **Effort.** M. **Tier.** CANDIDATE.

#### P13.18 — Phase-Transition Capability Switching

**What.** Switch capability at **learned phase boundaries** of a generation (a
measured change in the task regime), not at every token and not once per task.
**Why.** Token routing pays a decision per token and thrashes; task-level
selection cannot adapt inside a long task. A phase boundary is between them.
**Evidence boundary.** No located prior art supports learned phase-boundary
switching; the per-token conditional-depth family is the control this must beat.
**PROPOSED.**
**Decisive experiment.** Phase switching vs (a) token-level routing and (b) fixed
task-level selection, at matched bytes, with the **number of switches** reported.
**Falsifying control.** Token-level routing at matched bytes; and a phase detector
that fires at **fixed** positions.
**Kill criterion.** If phase switching does not beat both, or the learned boundary
is indistinguishable from a fixed schedule, retire.
**Depends.** P12.1, P12.9. **Acceptance evidence.** the three-arm receipt with the
switch count and boundary-position statistics. **Status.** open. **Effort.** L.
**Tier.** CANDIDATE.

#### P13.19 — Executable Capability Contracts

**What.** Each module ships an **applicability test** and bounded **pre/post
conditions** that are executed, so a module cannot be activated outside its stated
envelope.
**Why.** Selective classification (1705.08500, VERIFIED) is the published analogue
for abstention at a declared risk; the executable contract extends it to
composition: two modules may only compose if the postcondition of one satisfies
the precondition of the other.
**Decisive experiment.** Measure **harmful activations** (activations where the
module runs outside its envelope) and safe-composition rate, with and without
contracts.
**Falsifying control.** A **confidence threshold** at matched coverage (the
standard alternative that needs no contract).
**Kill criterion.** If contracts do not reduce harmful activations against the
threshold control at matched coverage, or safe-composition rate does not rise,
retire.
**Depends.** P12.9, P12.10. **Acceptance evidence.** the activation-envelope
receipt and the composition matrix. **Status.** open. **Effort.** M.
**Tier.** CANDIDATE.

#### P13.20 — Training for Future Locality

**What.** Train routing to be **both accurate and predictable** — a router that
optimises accuracy *and* the predictability of its own future needs — so the
prefetcher can act before the need.
**Why.** SeqMoE (2609.12978, VERIFIED) treats prediction as sequence modelling;
the gap is that accuracy and predictability are usually traded silently. Making
predictability a training term is the invention.
**Decisive experiment.** Joint-objective router vs accuracy-only router at matched
quality: **cold bytes per token and p95 latency**, with quality loss reported, not
hidden.
**Falsifying control.** Accuracy-only router at matched quality; and a router with
a **predictability term on a shuffled label** (surface-level regularisation).
**Kill criterion.** If cold bytes and p95 do not fall at **no** quality loss
(within the registered margin), retire: predictability bought at quality is the
current behaviour, not an advance.
**Depends.** P3.7, P4.4, P14.2. **Acceptance evidence.** the cold-byte/p95 receipt
with paired quality CIs. **Status.** open. **Effort.** L. **Tier.** CANDIDATE.

#### P13.21 — Verifier-Calibrated Internal Stopping

**What.** Stop internal computation when **independent evidence** indicates the
obligations are discharged (a verifier or an executable check), rather than when
the model's own distribution converges or its confidence is high.
**Why.** PonderNet (2107.05407, VERIFIED) learns halting, and DEER-style work uses
convergence/uncertainty — but both read the model's own state, which is exactly
what is wrong when the model is confidently wrong.
**Decisive experiment.** Verifier-calibrated stopping vs (a) convergence-only
halting and (b) confidence-only halting, on the fresh suite, with **task success
per unit compute** reported.
**Falsifying control.** The two halting baselines above at matched expected
compute; plus a verifier-check arm that runs the verifier but **ignores** its
answer (to separate the cost of verification from its benefit).
**Kill criterion.** If verifier-calibrated stopping does not beat both baselines
at matched compute, or the ignore-answer control matches, retire.
**Depends.** P12.9, P14.5. **Acceptance evidence.** the matched-compute receipt
with the verifier-cost row. **Status.** open. **Effort.** M. **Tier.** CANDIDATE.

---

## 19. WS14 — Runtime and evaluation corrections (physical-execution path)

These seven items are not new mechanisms. They are the corrections that decide
whether a **runtime** claim measures the model or the instrument. Each is
CPU-testable today. **P14.1–P14.3 do not gate the T2 capability run** — that court
executes through the PyTorch adapter (§0.0). They gate runtime and latency claims.

### P14.1 — Decoded-weight residency: stop re-decoding on every read

**What.** `GroupReader::read_group` (`crates/selvedge-runtime/src/bundle.rs`)
fills a byte buffer per shard and calls `decode_f32_in_place`, which converts
little-endian f32 bytes into a `Vec<f32>` — **on every call, including cache
hits**. Every shard the exact 0.6B token reads is decoded again
(arithmetic, not a measurement: 28 layers × 48 groups = 1,344 reads per token,
the structure recorded in `docs/RUNTIME.md`). Add (a) an **execution-ready
decoded representation** that
the compute step consumes directly, and (b) a **separation of the storage cache
from execution residency**, so "cached" and "resident as f32" are different,
separately measured states.
**Why.** Until this is fixed, a cache hit saves I/O but not conversion, and every
bytes/token and latency comparison is measuring the decode loop as much as the
model. It also blocks P13.1 (width tiers) and P13.20 (cold-byte p95), both of
which need to distinguish a byte saved from a decode avoided.
**Acceptance evidence.** a CPU receipt with (i) exact-mode numerical parity
against the current path (bitwise where the reduction order is unchanged),
(ii) decode passes per token before/after, (iii) peak decoded residency bytes,
and (iv) the existing Court E parity matrix still passing.
**Depends.** none. **Status.** open. **Effort.** M. **Tier.** ACTIVE.

### P14.2 — Prefetch is synchronous: separate fetch and compute timelines

**What.** In `crates/selvedge-runtime/src/run.rs` the prefetch is a blocking loop
over every fragment of the layer (`self.reader.prefetch(...)`), followed by the
compute loop; there is no overlap and no bounded staging buffer. Add separate
**fetch** and **compute** timelines, **bounded staging buffers** (with a declared
high-water mark), and a **measured readiness dependency** (what compute may start
before which shard is ready), so a stall is attributed to a named dependency
instead of to "the loop".
**Why.** The simulator already models overlap (`docs/SIMULATOR.md`) and the live
comparison was recorded as inconclusive because the host was saturated; the
runtime-side claim of overlap is currently unmeasurable because the runtime has no
overlap. Without this, p95 latency claims (P8.4, P13.20) measure a serial loop.
**Acceptance evidence.** a CPU receipt with per-token fetch and compute intervals,
the staging high-water bytes, the number of stalls attributable to each
dependency, and byte/cache-hit parity with the synchronous path.
**Depends.** P14.1. **Status.** open. **Effort.** L. **Tier.** ACTIVE.

### P14.3 — Token-at-a-time prefill: keep the scalar oracle, add a batched path

**What.** Prefill is executed token by token through the same scalar path as
decode. Keep the scalar path **as the reference oracle** (it is what Court E
proves against) and add a **batched prefill** path with a numerical-parity test
against that oracle on real windows.
**Why.** Any prefill/throughput comparison against an external system is
meaningless while prefill is a scalar loop, and the compiler/seed items (P13.6)
count prefill cost explicitly. The scalar path must survive as the oracle, or the
parity evidence disappears.
**Acceptance evidence.** a CPU receipt with prefill latency per token and per
batch, max-abs/max-rel parity against the scalar oracle on the published windows,
and a statement of the reduction order the batch path uses.
**Depends.** P14.1. **Status.** open. **Effort.** L. **Tier.** ACTIVE.

### P14.4 — Expand the evaluation suite beyond 24 tasks

**What.** Grow the fresh suite from 24 items (plus the 3-item anti-specialisation
trio) toward **hundreds of independently authored tasks**, with the same
authoring-source discipline, hash binding and per-item verifier. `docs/
CAPABILITY-COURT.md` records the current count; the target is stated as a range
and reached incrementally, with the suite hash re-bound in a new pre-registration
at each expansion (never re-bound in place).
**Why.** A 24-item suite cannot support a 10% relative claim with category
stratification: one item moves a category by 25–33%, and the paired CI is wide by
construction. The current registered threshold stays valid for the current suite;
the *frontier* claim needs the larger one.
**Acceptance evidence.** the expanded authoring source, its manifest and hash, an
updated contamination receipt, and a **new** pre-registration that supersedes
(does not edit) the current one.
**Depends.** P12.9. **Status.** partial: S21 expanded the suite from 24 to 72
scored tasks plus a 9-task trio, re-hashed it, re-ran contamination and wrote
`docs/capability-preregistration-v2.json` as an append-only supersession. The
"hundreds" target and the sealed set (P14.7) remain open. **Effort.** XL.
**Tier.** CANDIDATE.

### P14.5 — Strengthen checker coverage and scale for Rust cases

**What.** Rust cases are **already executed** through `cargo`/`rustc`: the
capability court runs its ordinary `cargo test` suite and the 3-item trio through
`cargo test`, and the compiler exit status and test result are the correctness
signal. The remaining gap is **not** a missing executable path. It is (a)
**coverage** — some items are still scored by reference-continuation likelihood
or a checker's structural comparison instead of an executed test — and (b)
**scale** — the executed set and the per-item isolation must grow with the suite.
**Why.** Reporting "Rust cases are not executed" would be false. The honest
deficit is that execution is the rule for the items that have it and not yet for
all of them, and that the sandbox must keep compile error, assertion failure,
timeout and allocation failure distinct (`docs/CAPABILITY-COURT.md`) as the set
grows. Stronger checkers (property/round-trip tests where a single expected-output
test is too weak) are the other half of the gap.
**Acceptance evidence.** a receipt showing, per tested item, the executed
command, exit status and outcome class; plus the count of items still scored
without execution, with a reason for each, and the growth of the executed set.
**Depends.** P12.9. **Status.** partial, improved by S21: all 81 scored tasks
(suite plus trio) now have an executed `cargo test` checker, a reference, a
build-time measurement and a rejected mutation, so the suite has no item scored
by reference-continuation likelihood; the executor and the OOM/timeout probes
are self-tested. What remains is scale and independence of the sandbox, plus
stronger property-based checkers. **Effort.** L. **Tier.** CANDIDATE (gates each
new suite expansion).

### P14.6 — Keep correctness separate from reference-continuation likelihood

**What.** Report two numbers that are never merged: **verified correctness**
(executed verifier outcome) and **reference-continuation likelihood** (KL/NLL
against the teacher's next-token distribution, the T2 instrument's metric). Every
capability table carries both columns, or explains why one is null.
**Why.** The T2 instrument is a distillation metric; the capability court is a
correctness metric. A single "score" that mixes them is how a distillation gain
is reported as a capability gain. The M1/M2 results are all likelihood results and
must never be read as capability results.
**Acceptance evidence.** the schema change in the result receipt (two named
fields), a test that refuses a capability table whose rows omit either, and a
paragraph in `docs/LIMITATIONS.md` naming the distinction.
**Depends.** P12.9, P14.5. **Status.** partial: the rule is written in
`docs/LIMITATIONS.md` and `docs/CAPABILITY-COURT.md` — verified success is the
endpoint and likelihood/NLL is only a selector signal — and the result receipt
reports no likelihood as capability. The schema test that refuses a table
missing either named column is not implemented yet. **Effort.** S. **Tier.**
CANDIDATE.

### P14.7 — A sealed evaluation set for external comparison

**What.** Hold back a **sealed** portion of the expanded suite, authored by the
same process, never used for development, opened only for a declared external
comparison and reported once per revision.
**Why.** An externally comparable number requires a set the program did not tune
against. Without it, every comparison is in-sample by construction, and the
contamination receipt (which currently covers public corpora) says nothing about
SELVEDGE's own development loop.
**Acceptance evidence.** the sealed set's hash recorded in the contract before any
arm sees it, a custody record, and a rule that a sealed result is published once
and never re-run for tuning.
**Depends.** P14.4. **Status.** open. **Effort.** M. **Tier.** CANDIDATE.
