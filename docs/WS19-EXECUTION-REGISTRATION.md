# WS19 execution registration — run order, teacher throughput, sparse-row training

Status: **registration and arithmetic, 2026-11-17 (S38).** No training run was
started. Nothing in this document is a measurement of a run that has not
happened; the three GPU jobs it registers are scheduled, not done. Every storage,
throughput, resource or verification-count number below is emitted from a receipt
between sentinels by `scripts/perf_numbers.py`; the arithmetic receipt is
`results/frontier-models/20261117T004000Z-sparse-memory-training.json` and the
state receipt named in §2 is `results/reference/20261117T003300Z-qwen35-base-state-contract.json`.

This document answers a question the plan left open after S37: *what exactly runs
next, in what order, with what bar for a win, and with what arithmetic for the
table that does not fit?* It registers four things — the run order (§2), the
throughput measurement that has to precede everything else (§4), the sparse-row
training design that the 8M-row table requires (§5), and the provenance rule that
the anchor scoreboard failed to follow (§6). The literature verification that
narrows the claims is in `docs/PRIOR-ART-AND-NOVELTY.md` §19.

## 1. What changed, and why it had to

Three defects in the S37 registration are corrected here and in the documents
they came from:

1. **The anchor was ambiguous.** The certificate was for `Qwen/Qwen3.5-0.8B`
   (chat) while the scoreboard scored `Qwen/Qwen3.5-0.8B-Base`. §10 of
   `docs/FRONTIER-TARGET.md` now measures the difference; every document names
   `-Base`.
2. **The per-pass state rule was backwards.** P19.4 said a pass's state is reset
   between tokens. It is carried across tokens and reset only between sequences
   (corrected in `docs/PROGRAM-PLAN.md` P19.4, `docs/RUNTIME.md` §13.3,
   `docs/TRAINING.md` §7, and proved by the state receipt below).
3. **A missing smoke was rendered as a zero.** `docs/STAGE2-PLAN.md` §5 printed
   `0.0` for every arm delta. It now says no smoke receipt exists.

## 2. The run order, registered

The sequence below is the registered order. Each step names its acceptance
evidence; a step may not begin before the previous step's evidence exists.

### 2.0 Job 0 — the teacher+student throughput measurement (GPU, first)

Registered in §4. It is first because every later budget is derived from it and
because it is the only step that can be done in one GPU session without training
anything.

### 2.1 Job 1 — the distillation-only control (the bar)

**What.** `Qwen/Qwen3.5-0.8B-Base` distilled **online** from
`Qwen/Qwen3.5-2B-Base` on the pinned token stream, with no memory, no loop and no
architecture change, then scored on the anchored harness
(`docs/anchor-preregistration-v1.json`) exactly as the S36 scoreboard was.

**Why it is first among mechanisms.** Every mechanism in stages 2–4 is trained
with a teacher signal. If the teacher signal alone moves the score, then a
mechanism that also moves the score has not been shown to do anything. The
distilled control **sets the bar every mechanism must beat**; a mechanism that
only beats the frozen anchor has demonstrated distillation, not memory.

**The bar is not the frozen anchor.** The registered win bar for stages 2–4 is
"beat the distilled control on the family under test, at equal unique active
parameters and equal bytes per token". The frozen-anchor comparison is still
reported (it is what the S36 scoreboard gives), but it cannot be the win.

**Registration.** The control's own config is a dated pre-registration —
`docs/ws19-distillation-control-preregistration-v1.json` — registered **before**
it runs and subject to the `prereg_provenance` court (§6). It fixes: the stream,
its token count, the objective, the optimiser, the seed set, the evaluation, and
the fact that the teacher is the 2B Base checkpoint whose tokenizer is
byte-identical to the student's (measured: `docs/FRONTIER-TARGET.md` §10).

**Acceptance evidence.** a run receipt, a harness receipt per task, and the
per-family score with its standard error, under the same pre-registration digest
as the anchor so `gap` will compare them.

### 2.2 Job 2 — P19.2 grafted memory with its four controls

The memory table (§5) at the development scale, with the four arms the stage
already registers: **frozen** (no hook), **dense adapter** (parameter-matched),
**NGM** (training-free), and **learned grafted memory**. The teacher is the same
2B checkpoint and the same stream as Job 1, so the only difference from the
distilled control is the mechanism.

**Win bar.** The memory arm must beat **the distilled control** (Job 1) on the
registered family, and beat the dense adapter and NGM at equal unique active — the
P19.2 bar, now expressed against the correct baseline.

### 2.3 Job 3 — P19.3 loop retrofit

The weight-tied middle-block loop with an identity-preserving one-loop path and a
re-entry bridge, one loop count fixed for the claim, and the compute-matched
control. It is trained **after** memory because it is the registered next stage
and because a memory result is needed before the two are combined: a loop that
changes what each pass reads cannot be interpreted if the read itself has not
been measured.

**The counter-argument, recorded because it is real.** Looping has published
matched-compute evidence: RecurTrace reports `56.9%` on MathQA at `2.0` loops,
`+2.2` points over the best fixed loop depth at matched compute, with a
loop-time memory attention and a halting head (`arXiv:2609.03379`,
`https://arxiv.org/abs/2609.03379`, read 2026-11-17); SMELT loops the middle half
of a MoE transformer twice at matched per-token FLOPs, parameters and KV cache
and saves `6.8–18.0%` of training FLOPs on the compute-optimal frontier, across
four sizes to 54B (`arXiv:2609.01343v3`, `https://arxiv.org/abs/2609.01343`, read
2026-11-17); the retrofitted-recurrence line reports GSM8K `46.2 -> 52.0` for a
TinyLlama-based retrofit at *test* recurrence 32 (`arXiv:2511.07384`,
`https://arxiv.org/html/2511.07384v1`, read 2026-11-17 — note that the gain is at
32× inference compute, not at matched compute). So looping is the better-evidenced
mechanism, and the memory-first order is **not** a claim that memory is stronger.
It is a scheduling choice, and the distilled control decides it empirically: if
Job 2 misses the bar and Job 3 would have been cheaper to run first, the order is
revisited with both receipts in hand rather than argued from priors.

### 2.4 Job 4 — P19.4 pass/state-conditioned memory

The table read inside every pass, addressed by the pass index and the pass's own
hidden state, with the three registered arms (one retrieval injected every pass;
the same rows refetched; genuinely pass/state-conditioned rows) and the four
controls. Depends on Jobs 2 and 3.

**Win bar.** (iii) beats **both** (i) and (ii) by more than the interval, **and**
the arm beats the distilled control. The second clause is new and is the point:
beating the frozen anchor is no longer a win at any stage.

## 3. Why the order is memory-then-loop, stated honestly

- The registered next stage is stage 2 (memory). P19.4 — the only narrow
  novelty — *requires* a working table and a working loop; running the loop
  first would leave the table untested and the novelty stage unstarted.
- A memory win is needed before combining: the loop changes the number of passes,
  so if the table does not move quality at one pass, its pass-conditioned version
  has no baseline to move from.
- The counter-argument is that looping is the better-evidenced mechanism (§2.3),
  and a rational scheduler might run it first. The registration accepts that risk
  and states the decision rule: **the distilled control in Job 1 runs regardless**,
  so whichever mechanism runs second is still measured against a same-compute
  baseline rather than against the frozen anchor.

## 4. Teacher+student throughput: the registered measurement protocol

**What replaces the estimate.** The plan currently quotes a teacher rate derived
from a measured student token rate and declared parameter ratios, labelled as an
estimate. That labelling stays until the following protocol has run. It is
scheduled as **the first GPU job of the next session**, because every budget in
`docs/STAGE2-PLAN.md` §7 is converted from tokens to GPU-hours using it.

**The measurement.** One *full* training step, repeated with warmup discarded:

```
teacher forward   : Qwen/Qwen3.5-2B-Base, bf16, no_grad (the same teacher as Job 1)
student forward   : Qwen/Qwen3.5-0.8B-Base, frozen core + memory arm, bf16 autocast
student backward  : gradients for the arm's trainable parameters only
loss              : CE + 0.5 * T^2 * KL(teacher_T || student_T), T = 2.0
optimizer step    : AdamW over the arm's parameters
```

Fixed configuration, identical to the smoke protocol so the number transfers:
sequence length 512, micro-batch 1, gradient accumulation 4 (so one optimiser step
is 2,048 tokens), one site after decoder layer 2, 65,536-row table, 8 reads per
token.

**What is reported, per run, in the receipt**
(`results/stage2/throughput-<stamp>.json`):

1. **tokens/s** — 2,048 tokens per optimiser step divided by the measured median
   optimiser-step wall time, and also the per-micro-batch rate;
2. **step time** — median and p95 wall time of the optimiser step, in ms, with the
   number of measured steps and the number of discarded warmup steps;
3. **peak VRAM** — `torch.cuda.max_memory_allocated` and `max_memory_reserved`, in
   bytes, recorded after the last measured step, with the device total;
4. **the breakdown** — per-phase CUDA-event time for teacher forward, student
   forward, loss, backward, optimiser and data transfer, each as a fraction of the
   step, so a regression can be attributed rather than guessed;
5. **the load** — the checkpoint paths and their config hashes, the arm
   configuration, the dtype, the CUDA and PyTorch versions, and the exact argv.

**Protocol rules.**

- ≥ 20 discarded warmup steps, then ≥ 50 measured steps; report the median, never
  the best.
- The teacher runs under `no_grad` and its weights are loaded once, outside the
  timed region.
- No `torch.compile` on the first measurement: the compiled variant is a second
  receipt if it is used at all, so the baseline is comparable to the S36 smoke.
- The job is launched with an explicit memory limit sized from this protocol's own
  first-pass peak (measure, then re-run under the limit); it never inherits the
  8 GiB court default.
- If the measured rate makes the registered token budgets exceed the session's
  GPU-hour cap, that is a **budget finding**: the 4B teacher is dropped and the
  2B route retained, recorded as an amendment, not silently rescaled.

**Acceptance evidence.** the receipt above, with all five groups of fields
present, and a sentence in `docs/STAGE2-PLAN.md` §7 that converts the registered
token budgets using the measured rate with the receipt stamp named.

## 5. Training the large table: sparse rows and an offloaded optimiser

### 5.1 The plain statement

**The 8,388,608-row table is untrainable with a dense optimiser on this device.**
AdamW needs a float32 master weight and two float32 moments — three float32
vectors per element — and the table alone is already

<!-- BEGIN GENERATED: sp-table-8m-gib -->
16.0 GiB
<!-- END GENERATED: sp-table-8m-gib -->

in bf16. Dense optimiser state for it is

<!-- BEGIN GENERATED: sp-dense-opt-8m-gib -->
96.0 GiB
<!-- END GENERATED: sp-dense-opt-8m-gib -->

so the dense training footprint is

<!-- BEGIN GENERATED: sp-dense-total-8m-gib -->
112.0 GiB
<!-- END GENERATED: sp-dense-total-8m-gib -->

against a 16 GiB device: a shortfall of

<!-- BEGIN GENERATED: sp-dense-shortfall-8m-gib -->
96.0 GiB
<!-- END GENERATED: sp-dense-shortfall-8m-gib -->

. No placement, quantisation or accumulation trick closes that with dense state.
The 1M-row scale is

<!-- BEGIN GENERATED: sp-dense-total-1m-gib -->
14.0 GiB
<!-- END GENERATED: sp-dense-total-1m-gib -->

dense, which fits only if nothing else is resident, and the development scale is

<!-- BEGIN GENERATED: sp-dense-total-65k-gib -->
0.875 GiB
<!-- END GENERATED: sp-dense-total-65k-gib -->

dense, which is why the development arm can be a dense reference.

### 5.2 The registered design

1. **Sparse per-row gradients.** A token reads 8 rows, so a step produces
   gradients for exactly the rows it read. At 2,048 tokens per optimiser step that
   is

   <!-- BEGIN GENERATED: sp-distinct-rows-step -->
16,384
   <!-- END GENERATED: sp-distinct-rows-step -->

   row-gradients as an upper bound (fewer when addresses collide). There is no
   dense gradient tensor at any point.

2. **Values gathered, gradients scattered.** Rows are gathered from host memory
   into a device buffer of the step's rows only:

   <!-- BEGIN GENERATED: sp-working-set-mib -->
64.0 MiB
   <!-- END GENERATED: sp-working-set-mib -->

   for values and gradients together, per optimiser step. The table is **never**
   fully resident on the device.

3. **Offloaded optimiser state, with a documented schedule.** The optimiser state
   lives in host pageable memory (and, at the 8M scale, partly on NVMe), one
   `(master, m, v)` triple per *touched* row, at

   <!-- BEGIN GENERATED: sp-row-opt-bytes -->
12,288 bytes
   <!-- END GENERATED: sp-row-opt-bytes -->

   per row. Host residency is capped at

   <!-- BEGIN GENERATED: sp-host-cap-rows -->
1,048,576 rows
   <!-- END GENERATED: sp-host-cap-rows -->

   — LRU-evicted, with `m` and `v` re-zeroed on a miss (a recorded trade-off, not
   a hidden one). The cap costs

   <!-- BEGIN GENERATED: sp-host-cap-gib -->
12.0 GiB
   <!-- END GENERATED: sp-host-cap-gib -->

   of host memory, which is a fraction of what the machine has and is the cheapest
   tier that keeps the update local.

   **The update schedule.** Gradients for a row are accumulated in the device
   buffer and flushed to the host tier in buckets: a bucket is flushed every

   <!-- BEGIN GENERATED: sp-bucket-steps -->
16
   <!-- END GENERATED: sp-bucket-steps -->

   optimiser steps, so the host↔device traffic is amortised by that factor and the
   NVMe write cost per step is

   <!-- BEGIN GENERATED: sp-nvme-write-mib -->
192.0 MiB
   <!-- END GENERATED: sp-nvme-write-mib -->

   at flush time, or

   <!-- BEGIN GENERATED: sp-nvme-write-amortised-mib -->
12.0 MiB
   <!-- END GENERATED: sp-nvme-write-amortised-mib -->

   per optimiser step amortised. The flush is ordered by row index, so it is a
   sequential scan of the touched range, not random I/O.

4. **Staleness bounds, stated and enforced.** Because a row's update is applied at
   its bucket flush, the gradients it uses are at most

   <!-- BEGIN GENERATED: sp-staleness-steps -->
16
   <!-- END GENERATED: sp-staleness-steps -->

   optimiser steps old, i.e. at most

   <!-- BEGIN GENERATED: sp-staleness-tokens -->
32,768
   <!-- END GENERATED: sp-staleness-tokens -->

   tokens stale. The bound is a *registered constant*, and the trainer asserts on
   it: a bucket older than the bound is a failed run, not a slower one. There is no
   asynchronous parameter server and no unbounded queue.

5. **Row-utilisation tracking.** Every row carries a touch counter and a
   last-touched step (a dense counter array at the 8M scale is affordable; beyond
   it the sketch route in the literature is the fallback). The receipt reports, per
   evaluation window: rows touched at least once, rows touched exactly once, the
   occupancy of the top 1% of rows, and the fraction never touched. Those two
   numbers decide the admission policy (which rows deserve to exist) and the row
   scale itself, which is the P19.2 question.

6. **Learning rate for rarely touched rows.** Adam's bias correction uses the
   **global** step, which under-corrects a row that is updated once every `J`
   steps: its first update is scaled as if it were the `t`-th step of a dense
   parameter. The registered correction is a **per-row step count** in the bias
   correction, so a row's first update is scaled by its own `1 / (1 - beta^1)`, and
   the per-row learning rate is the registered base rate. No gradient-magnitude
   compensation is added, because that would change the objective's effective
   weighted loss across rows and is not registered.

7. **Peak VRAM, as an inequality with one measured term.** The peak is
   `backbone + measured activation peak + sparse working set + per-pass state`.
   The backbone is

   <!-- BEGIN GENERATED: sp-backbone-gib -->
1.401 GiB
   <!-- END GENERATED: sp-backbone-gib -->

   the sparse working set is the working set above, and the per-pass state is

   <!-- BEGIN GENERATED: sp-per-pass-state-mib -->
18.6 MiB
   <!-- END GENERATED: sp-per-pass-state-mib -->

   (DeltaNet state in fp32 plus the convolution windows — the state receipt's own
   breakdown). The activation term is the one thing this document does **not**
   supply, because it is a measurement: §4 measures it, and the arithmetic closes
   with that number rather than with a guess. What the design changes is that the
   table and its optimiser state contribute **zero** accelerator bytes.

### 5.3 The small-scale validation plan

Three scales, each with a job and a bar. The second and third are blocked until
the first passes.

| scale | rows | residency | what it proves |
| --- | ---: | --- | --- |
| A | `65,536` | device, **dense** optimiser | the reference: sparse-row updates must equal dense AdamW bit-for-bit on the same data order and seed |
| B | `1,048,576` | host optimiser tier, device working set | the offload path: state never materialises on the device, and the run completes inside its measured limit |
| C | `8,388,608` | host + NVMe optimiser tier | the claim: table absent from device memory, staleness bound observed, utilisation measured |

**Acceptance evidence, per scale.** a run receipt carrying the measured peak
VRAM, the tokens/s, the staleness counters (maximum observed bucket age), the
row-utilisation block, and the exact argv and limits; plus, for scale A, the
sparse-equals-dense equivalence with its tolerance (bitwise equality is the
target and the receipt states whether it was reached); plus, for every scale, the
score on the fixed smoke subsets so the quality path is comparable across scales.

**Why scale A is the gate.** If sparse-row updates with the per-row step count do
not reproduce dense AdamW at 65,536 rows, the difference is a bug or a changed
objective. Measuring that at 65K is cheap; discovering it at 8M is not.

## 6. Pre-registration provenance: the defect and the court

### 6.1 The defect, with evidence

`docs/anchor-preregistration-v1.json` declares

```
registered_utc = 2026-11-16T19:40:00Z
```

The file was first committed at **2026-11-16T21:31:57Z** (`67ea47d`), and the
first scoring receipt for the anchor has stamp `20261116T195953Z` — 19:59:53 UTC,
**1 h 32 m before the registration entered the repository history**. The gap
reference's run started at 21:26:22 UTC, also before the commit. The file names
`results/frontier/anchor-smoke5-20261116T200500Z.json` in `limits_derived_from`,
and that smoke's own `started_utc` is 19:54:45 UTC — five minutes after the
declared registration time it is supposed to precede. So the recorded time is not
the authoring time, and the ordering that a pre-registration exists to guarantee
was not satisfied.

**Mitigation, stated without excusing it.** No tuning occurred. The task list,
few-shot counts, sample limits, prompt behaviour, seed set, family membership and
the win-condition text were fixed in the file's first commit (`67ea47d`) and the
file has not been modified since (one commit touches it). The thresholds are
derived mechanically from measured anchor and reference scores by
`scripts/s36_scoreboard.py`, so no post-hoc choice entered the win condition. That
is why the scoreboard is still quotable. It is **not** a reason to call the
provenance sound: a registration committed after the scores could have been
written with the scores in view, and a reader cannot verify that it was not from
the repository alone.

### 6.2 The court

`prereg_provenance` (`scripts/prereg_provenance.py`, run as
`scripts/court prereg_provenance`) fails closed when:

1. a registration file is **not tracked** by git — an uncommitted registration is
   not a registration;
2. the commit that first added it is **not earlier** than the first scoring
   receipt that names it (matched by the receipt's `preregistration.path` or
   `preregistration.sha256`, ignoring receipts flagged `smoke: true`);
3. the file's declared `registered_utc` is **later** than that commit;
4. the declared `registered_utc` is **earlier** than a receipt the file itself
   names as an input (`limits_derived_from`, `smoke_runs_without_receipts`) — a
   registration that claims to precede its own inputs.

It applies to every `<*preregistration*.json>` under `docs/`, so a new
registration is covered the moment it is written. The anchor is a **declared
historical exception** for rule 2 and rule 4, with the evidence above and the
mitigation; the exception is a named entry in the court, printed on every run, and
it does not extend to any future registration. The distillation control (§2.1) is
subject to the court with no exception.

## 7. What this document does not establish

- It does not establish that the memory table helps, that the loop helps, or that
  the combination helps. It registers the order and the bars.
- It does not measure the teacher or the student. §4 is a protocol.
- It does not measure the sparse trainer. §5.3 is a validation plan.
- It does not re-open the anchor's provenance. The defect stands as recorded, and
  the exception is a historical carve-out, not a repair.
- It does not claim novelty for sparse-row training of large tables: that is
  standard practice in recommendation systems, and §19 of
  `docs/PRIOR-ART-AND-NOVELTY.md` records the lineage.
