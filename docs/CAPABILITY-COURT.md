# CAPABILITY COURT — the instrument every frontier claim will use

Status: **instrument implemented and CPU-tested; the suite is authored, hashed
and contamination-checked; no arm has been evaluated on it.** The GPU belongs to
the running T2 pilot, so this session builds and proves the measurement machinery
and records the exact commands for the first real run. The evidence is two
receipts at revision `d7d62061c966a5b57d945fc3fc96893389fb49b9`:

- `results/capability/selftest-20261116T023557Z.json` — every checker outcome;
- `results/capability/contamination-20261116T023550Z.json` — disjointness.

The pre-registration is `docs/capability-preregistration.json`, written before
any arm run and bound to the suite hash.

This document implements `docs/PROGRAM-PLAN.md` P12.8 (three levels), P12.9
(fresh Rust suite), P12.10 (control matrix), P12.11 (thresholds) and P11.21
(anti-superficial-specialisation trio). It changes no registered T1/T2 arm,
threshold or verdict.

## 1. What the instrument is

Two stages, because the model and the checker cannot share a process safely or
honestly:

1. **generate** — the arm receives a rendered prompt and returns text. Runs in
   the oracle court (GPU checkpoints) or the rust court (the Rust runtime).
   Greedy decoding, fixed seed, fixed stop set, fixed prompt templates.
2. **check** — a separate process extracts fenced files, restores the hidden
   tests, and runs `cargo test` in a bounded sandbox. Runs in the rust court,
   which carries `cargo` and a standard-library Python.

The split is the structural form of "no access to the checker at inference": the
checker process does not exist while the model generates.

One harness then produces three levels:

- **Level 1 (neural mechanism):** verified task success on the fresh suite,
  compared against matched controls with paired, category-stratified confidence
  intervals (`compare`).
- **Level 2 (physically executable capacity):** success, cold bytes per token
  and p95 latency as *accessible* capacity grows at approximately fixed active
  work (`capacity_scaling`).
- **Level 3 (useful local assistant):** verified task success offline plus
  responsiveness, with the Court F latency receipts as the other half
  (`level3` combines the check receipt's success and measured offline evidence
  with a named Court F p95, and reports a null with a reason when none is
  supplied).

A level is never inferred from another. Level 2 without Level 1 is a storage
demo; Level 3 without Level 2 is a product demo with no capacity story.

## 2. The fresh Rust suite (P12.9)

Suite size, from the self-test receipt:

<!-- BEGIN GENERATED: cap-suite-tasks -->
24 tasks
<!-- END GENERATED: cap-suite-tasks -->

Every item is original to this repository, authored in
`fixtures/capability/tasks/`, and licensed `MIT OR Apache-2.0`, the same terms as
the tree (`LICENSE`). No prompt, test or solution is copied from HumanEval,
GSM8K, TruthfulQA, a public Rust benchmark or a code-repair dataset; §4 checks
that claim rather than asserting it.

Items per category, from the same receipt:

- lifetime-borrow:
<!-- BEGIN GENERATED: cap-cat-lifetime -->
4
<!-- END GENERATED: cap-cat-lifetime -->
- failing-test:
<!-- BEGIN GENERATED: cap-cat-failing -->
4
<!-- END GENERATED: cap-cat-failing -->
- api-migration:
<!-- BEGIN GENERATED: cap-cat-migration -->
4
<!-- END GENERATED: cap-cat-migration -->
- dependency-compat:
<!-- BEGIN GENERATED: cap-cat-dependency -->
3
<!-- END GENERATED: cap-cat-dependency -->
- concurrency:
<!-- BEGIN GENERATED: cap-cat-concurrency -->
3
<!-- END GENERATED: cap-cat-concurrency -->
- trait-impl:
<!-- BEGIN GENERATED: cap-cat-trait -->
3
<!-- END GENERATED: cap-cat-trait -->
- repo-scale:
<!-- BEGIN GENERATED: cap-cat-repo -->
3
<!-- END GENERATED: cap-cat-repo -->

The failing state and the checker for each category:

| category | failing state | checker |
| --- | --- | --- |
| lifetime-borrow | does not compile: `E0515`, `E0502`, `E0597`/`E0505` | `cargo test` after the repair |
| failing-test | compiles; a visible assertion fails | `cargo test` |
| api-migration | does not compile against a vendored 2.0 dependency | `cargo test` |
| dependency-compat | version-resolution, edition-2024 or feature-gate error | `cargo test` |
| concurrency | deadlock (timeout), lost update (assertion), `!Send` (compile error) | `cargo test` |
| trait-impl | missing associated type, coherence conflict, non-object-safe trait | `cargo test` |
| repo-scale | needs a repair propagated through three or four files | `cargo test` |

Each task directory has the same shape, and `scripts/capability_authoring.py
--check` fails if the tracked tree and the authoring source disagree:

```
fixtures/capability/tasks/<id>/
    task.json      metadata, limits, declared failing evidence, declared mutation
    initial/       the failing state handed to the arm
    hidden/        tests the harness owns and restores over anything the arm writes
    reference/     one reference solution that must pass
```

Three anti-gaming properties are structural, not procedural:

1. **The tests are hidden.** An arm may not write `tests/`; the harness deletes
   the directory and restores its own. A write to `tests/` is rejected during
   extraction and recorded as a failure.
2. **A locked manifest cannot be edited.** For most tasks `Cargo.toml` is
   harness-owned; where the task *is* a manifest repair, the arm may edit a
   strict subset (package/lib/dependencies/workspace) and `build.rs`, patches,
   extra targets, profiles and dev-dependencies are rejected.
3. **Vendored dependencies are read-only.** A migration task cannot be answered
   by reverting the dependency: `dep/` is restored after the arm writes.

The self-test proves each task discriminates before the suite can be used. At
revision `0c4b050`:

- failing states that fail exactly as declared:
<!-- BEGIN GENERATED: cap-initial-ok -->
27
<!-- END GENERATED: cap-initial-ok -->
- tasks checked:
<!-- BEGIN GENERATED: cap-tasks-checked -->
27
<!-- END GENERATED: cap-tasks-checked -->
- reference solutions that pass:
<!-- BEGIN GENERATED: cap-reference-ok -->
27
<!-- END GENERATED: cap-reference-ok -->
- deliberately wrong mutations that fail:
<!-- BEGIN GENERATED: cap-mutation-ok -->
27
<!-- END GENERATED: cap-mutation-ok -->
- sandbox probes with the expected outcome:
<!-- BEGIN GENERATED: cap-probes-ok -->
5
<!-- END GENERATED: cap-probes-ok -->
- sandbox probes defined:
<!-- BEGIN GENERATED: cap-probe-tasks -->
5 probes
<!-- END GENERATED: cap-probe-tasks -->

The last two lines are the point: a checker that cannot reject a wrong answer is
not a measurement, and a wrong answer that hangs is not a failure it can see.

### Verifier quality, stated per item

P12.9 asks for the verifier caveat per item, so it is written down rather than
implied:

- Every checked property is executable. There is no rubric, no textual match and
  no model-as-judge.
- The hidden tests are a superset of the visible failure for the debugging
  tasks, so satisfying only the visible assertion still fails.
- A task's tests can still be *incomplete* with respect to the intent: they
  check the registered behaviour at the registered boundaries and not every
  possible caller. The self-test's mutation is one independent wrong answer per
  task, not a proof of test completeness.
- The declared mutation is authored, so it tests the test against a *plausible*
  wrong answer rather than an adversarial search. A stronger check would be
  property-based or a mutation campaign; that is recorded as a gap (§9).

## 3. The anti-superficial-specialisation trio (P11.21)

Trio size, from the self-test receipt:

<!-- BEGIN GENERATED: cap-trio-tasks -->
3 tasks
<!-- END GENERATED: cap-trio-tasks -->

| surface | task | what the hidden tests require |
| --- | --- | --- |
| `rust-lifetime-chain` | `trio-01-lifetime-chain` | annotate one lifetime through lookup → resolve → chain so the returned `&str` is tied to the table, never to a short-lived alias slice |
| `numeric-dependency` | `trio-02-math-dependency` | return the minimal dependency-ordered recompute set for a target, excluding known values, rejecting cycles, deterministic tie-break |
| `quantifier-chain` | `trio-03-logic-quantifier` | return the shortest premise chain for a quantified conclusion and the indices that must be strengthened from `Some` to `All` |

**The shared operation**, annotated per hidden test in each `task.json`:
*build the dependency relation induced by a target, close it from the required
node while excluding already-satisfied nodes, reject an inconsistent cycle, and
emit the minimal ordered set of declarations to change.*

**How a win is judged.** Not on a pooled score. The check stage reports each
surface separately (`scores.trio`) and sets `trio_all_surfaces` only when every
surface is fully solved. The pre-registration says a transfer claim requires
success on every surface with the per-surface breakdown reported; a gain on one
surface is not transfer.

**Honest boundary.** This is a trio, not the P11.21 suite. P11.21 asks for at
least forty items per domain and an inter-annotator agreement check; this has
one task per surface with five or six annotated operation items each, annotated
by the author only. The operation annotation is therefore an author claim, not
an independently audited one, and §9 records that as an open gap. The trio is
usable now as a *pre-registered transfer probe*; it is not usable as the
published anti-specialisation instrument until the item count and the agreement
check exist.

## 4. Contamination

Method, in the two units the brief requires:

- **8-word shingles** over lower-cased, whitespace-collapsed text, compared per
  item against the union of every source;
- **13-token shingles** with the registered Qwen3-0.6B tokenizer, compared per
  item against the pinned held-out windows the KL metric uses.

Items compared per source, from
`results/capability/contamination-20261116T023550Z.json`:

- HumanEval items (prompt, solution, test):
<!-- BEGIN GENERATED: cap-contam-humaneval -->
164 HumanEval items
<!-- END GENERATED: cap-contam-humaneval -->
- GSM8K items (question, answer):
<!-- BEGIN GENERATED: cap-contam-gsm8k -->
1319 GSM8K items
<!-- END GENERATED: cap-contam-gsm8k -->
- TruthfulQA items (question, choices):
<!-- BEGIN GENERATED: cap-contam-truthfulqa -->
817 TruthfulQA items
<!-- END GENERATED: cap-contam-truthfulqa -->
- T2 evaluation windows, text:
<!-- BEGIN GENERATED: cap-contam-t2-windows -->
160 windows
<!-- END GENERATED: cap-contam-t2-windows -->
- T2 evaluation windows, tokens:
<!-- BEGIN GENERATED: cap-contam-t2-tokens -->
47468 tokens
<!-- END GENERATED: cap-contam-t2-tokens -->
- T1/T2 training corpus documents sampled (fineweb-edu, codeparrot,
  hendrycks-math):
<!-- BEGIN GENERATED: cap-contam-corpus-docs -->
4498 documents
<!-- END GENERATED: cap-contam-corpus-docs -->
- characters in that sample:
<!-- BEGIN GENERATED: cap-contam-corpus-chars -->
22736717 characters
<!-- END GENERATED: cap-contam-corpus-chars -->

Overlapping items across every source:

<!-- BEGIN GENERATED: cap-contam-offenders -->
0 overlapping items
<!-- END GENERATED: cap-contam-offenders -->

Verdict:

<!-- BEGIN GENERATED: cap-contam-verdict -->
disjoint
<!-- END GENERATED: cap-contam-verdict -->

The corpus comparison is a **bounded deterministic sample** — the first
documents in sorted file order, capped per source — and the receipt records the
document and character counts plus the rule. It is not a full-corpus pass, and
the script fails closed if a corpus source is absent rather than reporting a
vacuous pass. The four dataset sources and the held-out windows are compared in
full.

The check is not decorative: the first run found exactly one overlapping
13-token shingle — a sequence of three decimal literals that also occurs in a
held-out GSM8K question — in `failing-04-nan-ordering`. The fixture's numbers
were changed and the run re-passed. The overlap was a coincidence, not copied
text, but the rule is fail-closed and the fix is in the fixture, not in an
exception list.

## 5. Sandbox and checker evidence

Every checker invocation runs under all of the following, recorded per item:

- wall clock:
<!-- BEGIN GENERATED: cap-checker-timeout-s -->
45 s
<!-- END GENERATED: cap-checker-timeout-s -->
- CPU time, a backstop above the wall bound:
<!-- BEGIN GENERATED: cap-checker-cpu-s -->
60 s
<!-- END GENERATED: cap-checker-cpu-s -->
- address space (`RLIMIT_AS`):
<!-- BEGIN GENERATED: cap-checker-as-gib -->
3.00 GiB
<!-- END GENERATED: cap-checker-as-gib -->
- run-only probes (`RLIMIT_AS`):
<!-- BEGIN GENERATED: cap-run-only-as-mib -->
512 MiB
<!-- END GENERATED: cap-run-only-as-mib -->
- run-only probes (wall clock):
<!-- BEGIN GENERATED: cap-run-only-timeout-s -->
10 s
<!-- END GENERATED: cap-run-only-timeout-s -->
- court container memory:
<!-- BEGIN GENERATED: cap-court-mem-gib -->
12 GiB
<!-- END GENERATED: cap-court-mem-gib -->
- workdir budget:
<!-- BEGIN GENERATED: cap-workdir-budget-mib -->
2048 MiB
<!-- END GENERATED: cap-workdir-budget-mib -->

Per-file size, open files and the deterministic environment are fixed in
`oracle/capability/limits.py` and recorded in every receipt's `limits` object.

The address-space value is measured, not guessed. In the rust court on
2026-11-16 the pinned toolchain failed to link a one-file crate at a smaller
`RLIMIT_AS` — the linker aborted in thread creation — and linked at the value
above; the two probe court receipts are
`results/courts/rust-20261116T020832Z-746bdd.json` (fails) and
`results/courts/rust-20261116T020837Z-e7e5e5.json` (passes), both at revision
`da6354c`. Peak RSS for a tiny crate build measured about 114 MiB, so the
address-space bound is not the RSS bound: the container's memory limit is, and
the per-checker bound exists to stop an allocation amplifier and to make the
probe outcomes reproducible.

Evidence that each outcome is handled, from the self-test receipt:

| probe | expected | recorded | evidence |
| --- | --- | --- | --- |
| `probe-pass` | pass | `pass` | exit 0 |
| `probe-wrong` | assertion failure | `test_failed` | exit 101, `test result: FAILED` |
| `probe-compile-error` | compile error | `compile_error` | exit 101, `E0308` |
| `probe-timeout` | timeout | `timeout` | killed by signal 9 after the wall bound |
| `probe-oom` | allocation failure | `oom` | aborted by signal 6 with the allocator's message under the run-only `RLIMIT_AS` |

The OOM probe is an **allocator abort under `RLIMIT_AS`**, not a cgroup OOM
kill: its `cgroup_oom_kill_delta` is zero and the receipt says so. A cgroup kill
would be recorded with a non-zero delta from `/sys/fs/cgroup/memory.events` and
the container's `memory.peak`; both counters are read for every run, so the two
events are distinguishable rather than conflated.

Other recorded properties:

- **No network:** the sandbox refuses to run unless `/proc/net/route` has no
  default route. The receipt records `no-default-route` (the measured
  corroboration of the court's `--network none`).
- **No inherited file descriptors:** `close_fds=True`, a new session, stdin from
  `/dev/null`, and the whole process group killed on timeout.
- **Deterministic environment:** built from scratch with `HOME`, `TMPDIR` and
  `CARGO_HOME` inside the workdir, `CARGO_NET_OFFLINE=true`, a fixed
  `SOURCE_DATE_EPOCH`, `LC_ALL=C` and a recorded environment digest.
- **Bounded scratch:** `TMPDIR` is inside the workdir, `RLIMIT_FSIZE` bounds each
  file, and the workdir's byte count is measured per run.

Largest observed workdir:

<!-- BEGIN GENERATED: cap-max-workdir-mib -->
15.0 MiB
<!-- END GENERATED: cap-max-workdir-mib -->

Slowest checker run:

<!-- BEGIN GENERATED: cap-max-check-ms -->
164 ms
<!-- END GENERATED: cap-max-check-ms -->

## 6. The harness: adapters, scoring and anti-gaming

**Uniform adapters.** `oracle/capability/adapters.py` defines one interface with
four implementations: a local HF/PyTorch checkpoint, a trained T2 arm (base +
checkpoint, active parameters from the Rust admission record), the Rust runtime
in exact mode (measured device bytes, active unique parameters and p95 from its
own summary), and a scripted fake used by the CPU tests. An adapter only turns
prompts into text and reports what it measured; the harness owns everything else.

**Prompt and decoding contract.** `fixtures/capability/tasks/<id>/task.json`
plus the initial files render a deterministic prompt; greedy decoding, seed
`20261115`, and a fixed stop set. Prompt hashes are recorded at generation and
recomputed at check time; a mismatch refuses the result rather than scoring a
different prompt.

**Extraction.** Fenced blocks with `path=`, a single unnamed fence as the entry
file, or whole-text fallback for a Rust body. Paths outside `src/**.rs`, the
manifest policy, duplicate paths and harness-owned paths are rejected and
recorded as an invalid answer — never silently repaired.

**Outcomes.** `pass`, `compile_error`, `test_failed`, `timeout`, `oom`,
`invalid_solution`, `nonzero_exit`. Counts are reported per run; a timeout is a
failure with a duration, not a hang.

**Scoring.** Per item, success is 1 or 0. The primary composite is the
equal-weight mean of the seven suite categories, each the arithmetic mean of its
items; this prevents the category with the most items from setting the score.
The CI is a paired percentile bootstrap resampling items within category, with
categories weighted equally, 10,000 resamples, seed `20261115`. The trio is
scored separately by surface and never pooled into the primary.

**Anti-gaming checks**, all implemented and CPU-tested:

| check | mechanism | failure mode |
| --- | --- | --- |
| prompt-order invariance | the same items in a rotated order must give identical outcomes | an order-sensitive arm is caught by `order-invariance` |
| no test leakage into few-shot | 8-word shingles of every example against every item's prompt, code, hidden tests and reference | generation refuses on any overlap |
| no checker access at inference | two processes; the checker exists only after generation | structural, recorded per result |
| tool parity and no-tools mode | the local adapters are structurally tool-free (`tool_mode: "no-tools"`); a tool-using arm must declare it and may only be compared with tool-using baselines | `compare` raises |
| feedback parity | an arm that saw compiler/test feedback must be compared with one that did | `compare` raises |
| manifest and suite binding | prompt hashes and the suite hash are checked at scoring time | a stale or mutated suite refuses the result |

**Feedback is recorded but not yet looped.** The generation receipt carries
`feedback: "none"` and `compare` enforces parity. The repair loop that would
feed compiler output back (with the same budget for every arm) is not
implemented; §9 records it. No hidden tool subsidy is possible today because no
arm receives feedback.

**Control matrix (P12.10).** `court.control_row` validates the ten registered
fields for each of the eight registered rows; a missing field, or a null without
a reason, is an error. `task_success` must carry the paired CI.
`render_control_table` emits one table; no row is ever fabricated. The planned
3B–8B quantised baseline is a row with `status: "planned"` and a reason for every
null.

## 7. Pre-registered thresholds (P12.11)

Written before any arm run in `docs/capability-preregistration.json`, bound to
this suite hash:

<!-- BEGIN GENERATED: cap-suite-sha -->
092303e36da8ed3096990bc3df05aea8d6f49ee3426d26a57726c3c08a868ccc
<!-- END GENERATED: cap-suite-sha -->

The registered rules:

- **Primary endpoint.** The category-balanced mean verified task success on the
  twenty-four-task suite. A win requires a **10% relative** improvement with a
  paired 95% CI lower bound above zero, and no category regressing by more than
  0.05 absolute. A single-item difference is not a win (the CPU test asserts
  the decision rule rejects it).
- **Trio.** A transfer claim requires success on every surface, reported
  per surface.
- **Capacity scaling (Level 2).** At approximately fixed active unique
  parameters (within 5% of the smallest rung), fit log(cold bytes per token)
  against log(total accessible capacity); the slope's 95% CI upper bound must be
  below one.
- **No hidden teacher at inference.** Declared zero teacher calls **and** a
  named replay-court receipt (P7.5). An adapter that cannot expose the field is
  recorded `unverified`, never `satisfied`.
- **Physical Pareto.** No control row may dominate the arm on success, cold
  bytes per token, p95 latency and residency simultaneously.
- **Branches.** A 1B research branch and a 3B–8B product branch, reported
  separately and never pooled. The 3B–8B quantised local baseline is a planned
  control: no such checkpoint is present locally, and its row will carry nulls
  with reasons rather than a fabricated number.
- **Controls.** The eight registered rows, including the two most often missing
  (a Memory-Grafting-style latent baseline and a matching library without routing
  optimisation).

## 8. What a win on this suite establishes — and what it does not

A win establishes, at the pre-registered threshold and on this instrument:

1. that the arm solves more of these verified repair tasks than the named
   baseline, with a paired interval above zero and no category collapse;
2. that the gain is not explained by prompt order, leaked examples, checker
   access, tools or feedback asymmetry;
3. if the capacity criterion is also met, that the gain is reachable at
   approximately fixed active work and that cold bytes per token grow
   sublinearly in accessible capacity;
4. nothing about the model's behaviour outside these tasks.

A win does **not** establish:

- **general coding ability.** These are twenty-four authored repair tasks in
  seven categories. They are not a benchmark, they are not a random sample of
  programming work, and the item count is small. A confidence interval over
  items is not a claim about the population of software tasks.
- **contamination-free pretraining.** The suite is disjoint from every
  registered evaluation source and from a bounded sample of the training corpus.
  No local test can prove a model never saw these patterns; disjointness is a
  property of *our* data, not of the checkpoint's pretraining.
- **that a Level 1 gain is a Level 3 product.** Level 3 needs offline
  responsiveness and a product-branch baseline; neither is measured here.
- **that the trio proves transfer.** It is a pre-registered probe with an author
  annotation and three surfaces, not the P11.21 instrument.
- **that the verifier is complete.** One declared mutation per task is a
  discrimination check, not a proof that the hidden tests reject every wrong
  program.
- **anything about latency or memory.** Court F owns those numbers; this
  instrument reports task success and, at Level 2, consumes Court F receipts.

## 9. Remaining gaps, explicitly

1. **No arm has been evaluated.** Every number here describes the instrument.
   The pre-pilot commands are §10.
2. **The trio is under P11.21's item count.** One task per surface, author-only
   operation annotation, no inter-annotator agreement.
3. **Feedback repair loops are not implemented.** Feedback is recorded and
   parity-enforced; the loop that feeds compiler output back is future work.
4. **The corpus contamination check is a bounded sample**, not the full 754 MB
   cache. Coverage is in the receipt; a full pass needs a streaming implementation
   with a bounded shingle set.
5. **Mutation testing is shallow.** One authored wrong answer per task; a
   mutation campaign or property-based tests would strengthen the verifier.
6. **No 3B–8B quantised baseline exists locally**, so the product branch is
   planned, not measured.
7. **The Rust-runtime adapter is untested end to end** — it needs the built
   binary and the 0.6B bundle; its interface and its accounting fields are
   covered by the CPU tests, but no generation has run through it.
8. **The HF adapter's residency and I/O fields are null with reasons.** The
   PyTorch path does not expose device read counters; that is a limitation of the
   adapter, not of the model.

## 10. Exact commands for the first real runs

All of them are ready; none was run in this session.

### (a) The T2 checkpoints (D, M, X, H, H\\*), when the GPU frees

Step 1, generate (GPU, oracle court). One command per arm; substitute the arm
letter and the checkpoint directory. `H*` is `Hstar` on disk and `H*` in the
config name.

```bash
ARM=H; CKPT=results/artifacts/train/t2/$ARM/step-001024.pt
scripts/court --gpu oracle -- python3 scripts/capability_court.py generate \
  --t2-arm "$ARM" \
  --t2-checkpoint "$CKPT" \
  --t2-config "results/train/admission-configs/t2/$ARM.json" \
  --t2-admission "results/train/admission/t2/$ARM.json" \
  --preregistration docs/capability-preregistration.json \
  --out "results/capability/runs/t2-$ARM/generations.json"

# the same for D, M, X, and H* with
#   ARM=Hstar; --t2-arm 'H*'  --t2-config 'results/train/admission-configs/t2/H*.json'
```

Step 2, check (CPU, rust court; the court needs `cargo` and the checker sandbox):

```bash
SELVEDGE_MEM=12g SELVEDGE_CPUS=8 scripts/court rust -- \
  python3 scripts/capability_court.py check \
  --generations "results/capability/runs/t2-$ARM/generations.json" \
  --preregistration docs/capability-preregistration.json \
  --out "results/capability/runs/t2-$ARM/result.json"
```

Step 3, the Level 1 comparison against a control row:

```bash
scripts/court python -- python3 scripts/capability_court.py compare \
  --arm results/capability/runs/t2-H/result.json \
  --baseline results/capability/runs/t2-D/result.json \
  --out results/capability/runs/t2-H-vs-D/comparison.json
```

### (b) The Rust runtime, exact mode, on a CPU subset

The runtime is the only arm that runs in the rust court, so both stages can run
there. The model and bundle paths are the ones the runtime courts use.

```bash
BUNDLE=$(ls results/artifacts/qwen3-0.6b-ffn-g16/*.slv | head -1)
SELVEDGE_MEM=12g SELVEDGE_CPUS=8 scripts/court rust -- \
  python3 scripts/capability_court.py generate \
  --adapter-spec results/capability/adapters/rust-exact.json \
  --only borrow-01-local-reference,failing-01-chunk-boundary,trio-01-lifetime-chain \
  --preregistration docs/capability-preregistration.json \
  --out results/capability/runs/rust-exact-subset/generations.json
```

with `results/capability/adapters/rust-exact.json` containing:

```json
{
  "kind": "rust-runtime",
  "binary": "/work/target/release/selvedge",
  "model_dir": "/hf-cache/models/Qwen/Qwen3-0.6B/c1899de289a04d12100db370d81485cdf75e47ca",
  "bundle": "/work/BUNDLE_PATH",
  "threads": 8,
  "cache_bytes": 1200000000,
  "mode": "exact"
}
```

Substitute `BUNDLE_PATH` with the relative path printed by the `ls` above. The
subset is a subset *by construction*: a subset run cannot produce a primary
composite (the check stage refuses a missing category), so it measures the
instrument's integration, never a capability claim.

### (c) D/M complementarity (P8.1) through the verifier

After (a) has produced both result receipts:

```bash
scripts/court python -- python3 scripts/capability_court.py complementarity \
  --arm-a results/capability/runs/t2-D/result.json \
  --arm-b results/capability/runs/t2-M/result.json \
  --out results/capability/runs/t2-D-M/complementarity.json
```

The payload reports per-category means for each arm, the per-item oracle union,
and the headroom over the better single arm, with `deployable: false` because
the union needs each item's outcome. That is the same arithmetic
`oracle/eval/complementarity.py` used on the loss-based instrument, now applied
to verified task success.

## 11. Reproducing the instrument's own evidence

```bash
# 1. the tree matches the authoring source, and the manifest is current
SELVEDGE_MEM=4g scripts/court rust -- \
  python3 scripts/capability_authoring.py --check

# 2. every checker outcome, with a fresh self-test receipt
SELVEDGE_MEM=12g SELVEDGE_CPUS=8 scripts/court rust -- \
  python3 scripts/capability_selftest.py \
  --out results/capability/selftest-$(date -u +%Y%m%dT%H%M%SZ).json

# 3. contamination, fail-closed
SELVEDGE_MEM=8g SELVEDGE_CPUS=4 scripts/court oracle -- \
  python3 scripts/capability_contamination.py \
  --out results/capability/contamination-$(date -u +%Y%m%dT%H%M%SZ).json

# 4. the harness's CPU tests (statistics, controls, anti-gaming, sandbox)
SELVEDGE_MEM=8g SELVEDGE_CPUS=4 scripts/court python -m pytest \
  oracle/tests/test_capability_suite.py oracle/tests/test_capability_stats.py \
  oracle/tests/test_capability_court.py oracle/tests/test_capability_sandbox.py

# 5. the staged path end to end through the real checkers, with a scripted arm
#    (an instrument check, not an arm result: adapter.kind is `fake`)
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
SELVEDGE_MEM=12g SELVEDGE_CPUS=8 scripts/court rust -- sh -c "
  python3 scripts/capability_court.py fake-reference --out /tmp/cap-spec.json &&
  python3 scripts/capability_court.py generate --adapter-spec /tmp/cap-spec.json \
    --preregistration docs/capability-preregistration.json \
    --out results/capability/instrument-smoke-$STAMP/generations.json &&
  python3 scripts/capability_court.py check \
    --generations results/capability/instrument-smoke-$STAMP/generations.json \
    --preregistration docs/capability-preregistration.json \
    --out results/capability/instrument-smoke-$STAMP/result.json"
```

The committed smoke run is
`results/capability/instrument-smoke-20261116T023255Z/`: the scripted arm emits
the reference solutions, every real checker runs, and the pre-registration
verifies. Its receipt is evidence that the pipeline works, not evidence about
any model.
