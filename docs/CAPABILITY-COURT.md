# CAPABILITY COURT — the instrument every frontier claim will use

Status: **instrument implemented and CPU-tested; the scored suite is at the size
recorded in §2, the trio is unchanged, a sealed holdback is frozen behind a
custody gate, and no arm has been evaluated on it.** The GPU belongs to the
running H\* training job, so this session extends and proves the measurement
machinery and records the exact commands for the first real run. The evidence is five receipts at
revisions `2beba67` (instrument, sealed, contamination, similarity) and
`5e5543f` (power):

- `results/capability/selftest-20261116T090000Z.json` — every checker outcome,
  the per-task build time and the six sandbox-probe outcomes;
- `results/capability/sealed-score-20261116T090100Z.json` — the sealed set's
  instrument self-check, behind its unlock record;
- `results/capability/contamination-20261116T090400Z.json` — disjointness over
  the scored suite and the sealed set;
- `results/capability/similarity-20261116T090600Z.json` — the near-duplicate
  screen over all 177 authored items;
- `results/capability/power-20261116T090700Z.json` — the power arithmetic for
  the registered composite at n = 144.

**Dated correction, 2026-11-16 (S28).** The five-arm T2 run scored 0.0000 on
every arm. §12 records the failure taxonomy of those stored receipts and the
conclusion it forces: the v4 format — complete replacement files for multi-file
crates — is outside what a 0.6B arm can produce in this harness, so the 0.0000
result measures an instrument floor and is **uninformative**, and the power
analysis above (which assumed a 0.5 baseline) is invalid. §13 defines the
redesigned, executable formats, §14 registers the calibration protocol that
must run before any capability claim, and §15 gives the exact GPU commands. The
registered T2 verdict, its thresholds and every historical receipt are
unchanged.

The original pre-registration is `docs/capability-preregistration.json` (the
historical suite, retained unchanged). The first expansion was registered in v2;
a same-session near-duplicate audit replaced sixteen tasks and the fixture is
registered in `docs/capability-preregistration-v3.json`. The P14.4 expansion
supersedes v3 with `docs/capability-preregistration-v4.json`, written before any
arm run against it and bound to the final suite hash.

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
144 tasks
<!-- END GENERATED: cap-suite-tasks -->

Items are spread across three author-assigned difficulty bands (the label is an
editorial judgement, not a measured difficulty):

- easy:
<!-- BEGIN GENERATED: cap-diff-easy -->
53
<!-- END GENERATED: cap-diff-easy -->
- medium:
<!-- BEGIN GENERATED: cap-diff-medium -->
85
<!-- END GENERATED: cap-diff-medium -->
- hard:
<!-- BEGIN GENERATED: cap-diff-hard -->
21
<!-- END GENERATED: cap-diff-hard -->

Every item is original to this repository, authored in
`fixtures/capability/tasks/`, and licensed `MIT OR Apache-2.0`, the same terms as
the tree (`LICENSE`). No prompt, test or solution is copied from HumanEval,
GSM8K, TruthfulQA, a public Rust benchmark or a code-repair dataset; §4 checks
that claim rather than asserting it.

Items per category, from the same receipt:

- lifetime-borrow:
<!-- BEGIN GENERATED: cap-cat-lifetime -->
18
<!-- END GENERATED: cap-cat-lifetime -->
- failing-test:
<!-- BEGIN GENERATED: cap-cat-failing -->
18
<!-- END GENERATED: cap-cat-failing -->
- api-migration:
<!-- BEGIN GENERATED: cap-cat-migration -->
18
<!-- END GENERATED: cap-cat-migration -->
- dependency-compat:
<!-- BEGIN GENERATED: cap-cat-dependency -->
18
<!-- END GENERATED: cap-cat-dependency -->
- concurrency:
<!-- BEGIN GENERATED: cap-cat-concurrency -->
18
<!-- END GENERATED: cap-cat-concurrency -->
- trait-impl:
<!-- BEGIN GENERATED: cap-cat-trait -->
18
<!-- END GENERATED: cap-cat-trait -->
- repo-scale:
<!-- BEGIN GENERATED: cap-cat-repo -->
18
<!-- END GENERATED: cap-cat-repo -->
- semantic-repair:
<!-- BEGIN GENERATED: cap-cat-semantic -->
18
<!-- END GENERATED: cap-cat-semantic -->

The categories are deliberately separate. `failing-test` starts from a red
visible assertion; `semantic-repair` starts from a **green** visible assertion
and a behavioural defect that only the hidden tests expose
(`initial_failure.kind = hidden_test_failure`). Mixing them would hide that
distinction in the aggregate.

The failing state and the checker for each category:

| category | failing state | checker |
| --- | --- | --- |
| lifetime-borrow | does not compile: `E0515`, `E0502`, `E0597`/`E0505`, `E0506`, `E0507`, `E0509`, `E0596`, `E0382`, `E0373`, or a runtime double borrow | `cargo test` after the repair |
| failing-test | compiles; a visible assertion fails | `cargo test` |
| api-migration | does not compile against a vendored 2.0 dependency | `cargo test` |
| dependency-compat | version resolution, edition, feature, workspace or manifest-type error | `cargo test` |
| concurrency | deadlock (timeout), lost update (assertion), `!Send`/`!Sync` (compile error) | `cargo test` |
| trait-impl | missing or mismatched item, coherence conflict, non-object-safe trait, dyn associated type | `cargo test` |
| repo-scale | needs a repair propagated through three or four files | `cargo test` |
| semantic-repair | compiles and its visible test passes; hidden behavioural tests fail | `cargo test` with hidden tests restored |

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

The self-test proves each task discriminates before the suite can be used. At the
receipt stamp `20261116T090000Z` (revision `2beba67`):

- failing states that fail exactly as declared:
<!-- BEGIN GENERATED: cap-initial-ok -->
153
<!-- END GENERATED: cap-initial-ok -->
- tasks checked (suite plus trio):
<!-- BEGIN GENERATED: cap-tasks-checked -->
153
<!-- END GENERATED: cap-tasks-checked -->
- reference solutions that pass:
<!-- BEGIN GENERATED: cap-reference-ok -->
153
<!-- END GENERATED: cap-reference-ok -->
- deliberately wrong mutations that fail:
<!-- BEGIN GENERATED: cap-mutation-ok -->
153
<!-- END GENERATED: cap-mutation-ok -->
- declared naive fixes that compile and then fail the hidden tests:
<!-- BEGIN GENERATED: cap-naive-rejected -->
8
<!-- END GENERATED: cap-naive-rejected -->
- naive fixes declared (one per category):
<!-- BEGIN GENERATED: cap-naive-declared -->
8
<!-- END GENERATED: cap-naive-declared -->
- sandbox probes with the expected outcome:
<!-- BEGIN GENERATED: cap-probes-ok -->
6
<!-- END GENERATED: cap-probes-ok -->
- sandbox probes defined:
<!-- BEGIN GENERATED: cap-probe-tasks -->
6 probes
<!-- END GENERATED: cap-probe-tasks -->

The **naive fix** is a second discrimination check, distinct from the mutation:
it is the plausible shallow answer an arm reaches for first (a hard-coded
literal, an ignored parameter, a value clamped where the contract does not clamp
it). The self-test requires it to compile *and* fail a hidden behavioural test,
so "the tests reject a shallow string-level patch" is measured rather than
claimed. One task per category carries one.

Per-task build time (compile only, `cargo test --no-run`) and full checker time
are recorded for every item in the receipt (`items[].build.duration_ms` and
`items[].reference.duration_ms`). The slowest build observed in each category:

- lifetime-borrow:
<!-- BEGIN GENERATED: cap-build-lifetime -->
164 ms
<!-- END GENERATED: cap-build-lifetime -->
- failing-test:
<!-- BEGIN GENERATED: cap-build-failing -->
114 ms
<!-- END GENERATED: cap-build-failing -->
- api-migration:
<!-- BEGIN GENERATED: cap-build-migration -->
114 ms
<!-- END GENERATED: cap-build-migration -->
- dependency-compat:
<!-- BEGIN GENERATED: cap-build-dependency -->
114 ms
<!-- END GENERATED: cap-build-dependency -->
- concurrency:
<!-- BEGIN GENERATED: cap-build-concurrency -->
1616 ms
<!-- END GENERATED: cap-build-concurrency -->
- trait-impl:
<!-- BEGIN GENERATED: cap-build-trait -->
114 ms
<!-- END GENERATED: cap-build-trait -->
- repo-scale:
<!-- BEGIN GENERATED: cap-build-repo -->
114 ms
<!-- END GENERATED: cap-build-repo -->
- semantic-repair:
<!-- BEGIN GENERATED: cap-build-semantic -->
164 ms
<!-- END GENERATED: cap-build-semantic -->
- trio:
<!-- BEGIN GENERATED: cap-build-trio -->
164 ms
<!-- END GENERATED: cap-build-trio -->

The largest build over every scored item is:

<!-- BEGIN GENERATED: cap-max-build-ms -->
1616 ms
<!-- END GENERATED: cap-max-build-ms -->

so the 45 s checker bound is not close to binding and every crate is small
enough to compile and test inside it. The last two self-test lines are the point:
a checker that cannot reject a wrong answer is not a measurement, and a wrong
answer that hangs is not a failure it can see.

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
9 tasks
<!-- END GENERATED: cap-trio-tasks -->

Three operation families are applied on each of three surfaces, each surface in
its own crate with its own hidden checker:

| operation family | Rust | numeric/graph | logic |
| --- | --- | --- | --- |
| dependency-closure minimal repair | `trio-01-lifetime-chain` | `trio-02-math-dependency` | `trio-03-logic-quantifier` |
| invariant preservation | `trio-04-invariant-ledger` | `trio-05-invariant-graph` | `trio-06-invariant-substitution` |
| quantifier/scope resolution | `trio-07-scope-resolver` | `trio-08-scoped-quantifier-eval` | `trio-09-quantifier-occurrences` |

**The shared operations**, annotated per hidden test in each `task.json`:

1. *dependency-closure minimal repair* — build the dependency relation induced
   by a target, close it from the required node while excluding already-satisfied
   nodes, reject an inconsistent cycle, and emit the minimal ordered set of
   declarations to change.
2. *invariant preservation* — identify the state relation, apply a transition to
   every component that carries it, and reject (leaving state unchanged) any
   transition that would break it.
3. *quantifier/scope resolution* — resolve each occurrence against the nearest
   enclosing binding for that name, distinguish free from bound occurrences, and
   never leak an inner binding.

**How a win is judged.** Not on a pooled score. The check stage aggregates by
surface across all three operation families (`scores.trio`) and sets
`trio_all_surfaces` only when every surface is fully solved (all three tasks in
that surface pass). The pre-registration says a transfer claim requires success
on every surface with the per-surface breakdown reported; a gain on one surface,
or on one operation family, is not transfer.

**Honest boundary.** This is the trio sized above, not the P11.21 suite. P11.21
asks for at least forty items per domain and an inter-annotator agreement check;
this has the per-surface task count above, with the operation annotations in
each task, annotated by the author only. The operation annotation is therefore
an author claim, not an independently audited one, and §9 records that as an
open gap. The trio is usable now as a *pre-registered transfer probe*; it is not
usable as the published anti-specialisation instrument until the item count and
the agreement check exist.

### 3.1 The near-duplicate screen (P14.4)

An expansion is only an expansion if its items are not restatements of the
existing ones. `scripts/capability_similarity.py` compares every pair of
authored items in three views, with set-Jaccard over token shingles:

- **diagnostic** — the declared failure kind, required evidence and observed
  diagnostic, with line numbers and literals normalised (2-grams);
- **code** — the initial and reference Rust sources with comments removed
  (4-grams);
- **assertion** — the hidden tests' assertion lines with literals normalised
  (3-grams).

The thresholds were registered in the script before the screen ran on the
expansion: reject at code Jaccard ≥ 0.60, or at diagnostic ≥ 0.75 and code ≥ 0.30,
or at assertion ≥ 0.60 and code ≥ 0.30. The receipt records every rule, the
maximum observed similarity in each view, and every pair above a rule.

Pairs compared over every authored item (scored suite, trio and sealed set):

<!-- BEGIN GENERATED: cap-similarity-pairs -->
15576
<!-- END GENERATED: cap-similarity-pairs -->

Pairs rejected, whose items were replaced before the freeze:

<!-- BEGIN GENERATED: cap-similarity-rejected -->
0
<!-- END GENERATED: cap-similarity-rejected -->

Pairs above a threshold but outside the replaceable scope (both items are from
the already-registered historical suite; replacing them would change a
registered fixture, so they are reported for a future supersession):

<!-- BEGIN GENERATED: cap-similarity-historical -->
2
<!-- END GENERATED: cap-similarity-historical -->

The screen is not decorative. It rejected, and forced the replacement or
rework of, twelve items across two authoring passes: two dependency tasks that both
computed the same multiplicative checksum and asserted the same values, a
repo-scale task that repeated the api-migration padding task's assertions, a
sealed atomic-increment task whose diagnostic restated a scored atomic task,
and two tasks that shared the same integer-mean assertions. Each replacement
changed the computation, the diagnostic or the assertions, not the threshold.
The two historical pairs are named in the receipt; they are
`deps-04`/`deps-05` (shared hidden assertion, code 0.61) and
`deps-08`/`deps-09` (assertion 0.63), both from the v3-registered suite.

### 3.2 The sealed holdback (P14.7)

`fixtures/capability/sealed/` holds a further three items per category,
authored by the same process and **frozen** by `fixtures/capability/SEALED.json`.
The registration records the sealed aggregate, every item hash, the freeze date,
the rule and the exact one-shot unlock procedure. Items authored for the sealed
set:

<!-- BEGIN GENERATED: cap-sealed-items -->
24
<!-- END GENERATED: cap-sealed-items -->

Sealed aggregate:

<!-- BEGIN GENERATED: cap-sealed-sha -->
7666768f3abdbd93c268cc6d10250ddefa13a3a58c94268863fdcab79e7984bc
<!-- END GENERATED: cap-sealed-sha -->

The gate is code, not prose. `oracle/capability/sealed.py::authorize` refuses
unless `--unlock-sealed` is supplied **and** the supplied SHA-256 equals the
registered aggregate; an accepted unlock returns a record naming the flag, the
hash and the UTC time, which `scripts/capability_sealed.py` writes *before* it
checks any item. `check_registration` recomputes every item hash, the aggregate
and the counts and fails closed on drift. The registration is append-only.
`oracle/tests/test_capability_sealed.py` proves refusal without the flag, without
a hash and with a wrong hash; acceptance with both; the recorded unlock time; and
tamper and extra-item detection.

During authoring the sealed instrument was self-checked more than once: the first
two unlocks (records `sealed-unlock-20261116T080133Z.json` and the following
one) were instrument checks that failed or preceded a fixture repair, and the
freeze was rewritten before any model was evaluated. No arm has been run against
the sealed set, and the final self-check
(`results/capability/sealed-score-20261116T090100Z.json`) passes every item.
Those superseded unlock records are retained, not deleted: they are part of the
custody trail. The registration must not be rewritten again without a recorded
reason.

## 4. Contamination

Method, in the two units the brief requires:

- **8-word shingles** over lower-cased, whitespace-collapsed text, compared per
  item against the union of every source;
- **13-token shingles** with the registered Qwen3-0.6B tokenizer, compared per
  item against the pinned held-out windows the KL metric uses.

Items compared per source, from
`results/capability/contamination-20261116T090400Z.json`. The comparison covers
the scored suite, the trio and the sealed set; the receipt records the counts
for each of the three collections and binds both hashes.

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
- authored items hashed and shingled (suite, trio and sealed):
<!-- BEGIN GENERATED: cap-contam-items -->
177
<!-- END GENERATED: cap-contam-items -->

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
full. The script walks the fixture tree rather than a list, so every item
authored for the expansion — including the sealed items — was hashed and
shingled without a second registration step.

The check is not decorative. The first run found exactly one overlapping
13-token shingle — a sequence of three decimal literals that also occurs in a
held-out GSM8K question — in `failing-04-nan-ordering`. The expansion's first run
found one more, in `sealed-borrow-01-sort-closure-mutation`: the shingle decoded
to the integer sequence `1, 2, 3, 3, 3`, which a held-out window also contains.
Both fixtures had their numbers changed and the runs re-passed. The overlaps were
coincidences, not copied text, but the rule is fail-closed and every fix is in
the fixture, never in an exception list.

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
10 GiB
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
| `probe-stdout-forgery` | assertion failure | `test_failed` | exit 101; the test printed `test result: ok. 3 passed` and then asserted a falsehood, so stdout did not decide the outcome |

The OOM probe is an **allocator abort under `RLIMIT_AS`**, not a cgroup OOM
kill: its `cgroup_oom_kill_delta` is zero and the receipt says so. A cgroup kill
would be recorded with a non-zero delta from `/sys/fs/cgroup/memory.events` and
the container's `memory.peak`; both counters are read for every run, so the two
events are distinguishable rather than conflated.

The stdout-forgery probe closes the last non-behavioural escape hatch: an answer
that prints the expected test summary does not pass, because the recorded status
is the subprocess outcome and the hidden behaviour still fails. The self-test
also requires the forged marker to appear in the captured stdout, so the probe
cannot silently stop exercising the path.

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
15.1 MiB
<!-- END GENERATED: cap-max-workdir-mib -->

Slowest checker run:

<!-- BEGIN GENERATED: cap-max-check-ms -->
1025 ms
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
recorded as an invalid answer — never silently repaired. This is the registered
rule for the v4 suite and it is unchanged. The calibrated suite (§13) sets
`extraction_policy: "tolerant"` per task, which additionally reads a `path=`
line written *inside* a fence, takes the last block for a repeated path and
picks the most Rust-like block among several unnamed fences; it still rejects
every forbidden path, and it never invents code. §12 reports what that tolerance
actually rescues: nothing that passes.

**Formats.** `oracle/capability/formats.py` owns the prompt, answer parsing and
executable judging for the four registered formats (§13). The legacy
`file-replacement` path renders exactly the v4 prompt (the stored prompt hashes
still recompute), and the three new formats are opt-in per task through
`metadata.format` + `metadata.format_spec`, validated by
`suite.load_task` so a malformed format declaration refuses the suite rather
than silently falling back.

**Outcomes.** `pass`, `compile_error`, `test_failed`, `timeout`, `oom`,
`invalid_solution`, `nonzero_exit`. Counts are reported per run; a timeout is a
failure with a duration, not a hang. Every row now also carries `format`, `tier`,
`metric`, `failure_class`, `failure_flags`, the parsed answer's screen and (in
`calibrate`-style runs) `partial_credit`; `failure_class` uses the registered
classes of §12 and `verify-formats`/`diagnose` recompute them from stored text.

**Scoring.** Per item, success is 1 or 0. The primary composite is the
equal-weight mean of the eight suite categories, each the arithmetic mean of its
items; this prevents the category with the most items from setting the score.
The CI is a paired percentile bootstrap resampling items within category, with
categories weighted equally, 10,000 resamples, seed `20261115`. Every category
is also reported with its own interval, its `n`, and the registered per-category
minimum (18 in the v4 registration); a category below that minimum refuses the
result. `--category` on
`generate` and `check` scores a stratified subset (all selected categories must
still meet the minimum), and the receipt records `suite.category_filter`. The
trio is scored separately by surface and never pooled into the primary; the
pooled item mean is reported for transparency but never used as the endpoint.

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

The original registration is `docs/capability-preregistration.json`, written for
the historical suite. It is retained byte-for-byte. **Dated addition, 2026-11-16
(S21, P14.4):** the first expansion was registered in
`docs/capability-preregistration-v2.json` with the same 10% relative threshold,
the same paired bootstrap and seed, the registered per-category minimum and the
three shared surfaces. **Dated correction, same day (S21, QA):** a near-duplicate
audit replaced sixteen tasks that reused an existing capability; the corrected
fixture is registered in `docs/capability-preregistration-v3.json` and
supersedes v2 without editing it. **Dated addition, 2026-11-16 (S26, P14.4):**
the suite was doubled to the size in §2 and registered in
`docs/capability-preregistration-v4.json`, which supersedes v3 without editing
it. The threshold, the bootstrap seed and resample count, the decision rule and
the trio rule are unchanged; only the suite binding, the item count, the
per-category minimum (18) and the sealed-holdback policy are new. A run against
the 144-task suite must name v4; a run against the historical suite must name v1.
The hash below is the v4 binding:

<!-- BEGIN GENERATED: cap-suite-sha -->
7347e1e94b7ddcd3d04f1580646b392fc1591a25e71b299a663fc7f837f0a207
<!-- END GENERATED: cap-suite-sha -->

The registered rules:

- **Primary endpoint.** The category-balanced mean verified task success on the
  expanded suite (size in §2). A win requires a **10% relative** improvement with
  a paired 95% CI lower bound above zero, and no category regressing by more than
  0.05 absolute. A single-item difference is not a win (the CPU test asserts
  the decision rule rejects it).
- **Per-category minimum.** A category result is valid only with at least the
  registered number of scored items (see
  `docs/capability-preregistration-v4.json`: 18). The suite has that count in
  every category (§2), so one item moves a category by about 5.6%, not the
  quarter-to-third movement of the original 4-item categories.

  **Correction, 2026-11-16 (S16C).** This rule was written correctly and the code
  did not enforce it: `court.compare()` computed the regression set and then
  decided on the primary condition alone, so an aggregate winner that wrecked one
  category was reported as a win. The verdict is now **conjunctive**, the result
  reports `improvement_met` separately from `threshold_met`, and the verdict text
  names the regressing categories. Regression test:
  `oracle/tests/test_capability_court.py::test_compare_is_conjunctive_a_category_regression_kills_an_aggregate_win`.

  **Registration binding, same revision.** A comparison is now *registered* when
  either result carries a pre-registration digest. For a registered run the
  threshold comes from the receipt, the contract is checked against the suite hash
  and the registered threshold, `check()` refuses a run with no recorded digest
  (it used to skip the comparison), and `compare()` refuses an externally
  supplied threshold, a tampered recorded threshold, and any registered /
  unregistered mix.
- **Trio.** A transfer claim requires success on every surface (`rust`,
  `numeric-graph`, `logic`), aggregated over the three operation families and
  reported per surface; results are never pooled across surfaces.
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

### 7.1 Power at the registered n (P14.4)

The expansion was made for power, so the power is reported rather than implied.
The registered estimand is the category-balanced mean of paired item
differences; `oracle/capability/power.py` implements the arithmetic and
`results/capability/power-20261116T090700Z.json` is the receipt. The model is
stated: each item's paired difference is in {gain, loss, none}, the baseline
success rate is 0.5, and the "with regressions" case reverses 20% of the gains.

Items in the registered composite:

<!-- BEGIN GENERATED: cap-power-n -->
144
<!-- END GENERATED: cap-power-n -->

Items per category:

<!-- BEGIN GENERATED: cap-power-per-category -->
18
<!-- END GENERATED: cap-power-per-category -->

A *single category's* 95% CI half-width on its own success rate (p = 0.5) is:

<!-- BEGIN GENERATED: cap-power-marginal-ci -->
0.231
<!-- END GENERATED: cap-power-marginal-ci -->

The paired composite CI is narrower than that marginal interval because the
comparison is paired; the two are reported separately and never substituted.
Every category has the per-category n above and therefore the same marginal
half-width at the same proportion; the receipt records one row per category with
its name, `n` and half-width, so the per-category intervals are enumerated rather
than implied.

Analytic minimum detectable effect (normal approximation to the paired
composite, 95% two-sided, 80% power), absolute:

<!-- BEGIN GENERATED: cap-power-mde-abs -->
0.0517
<!-- END GENERATED: cap-power-mde-abs -->

as a fraction of the 0.5 baseline:

<!-- BEGIN GENERATED: cap-power-mde-rel -->
0.1034
<!-- END GENERATED: cap-power-mde-rel -->

and with 20% of gains reversed:

<!-- BEGIN GENERATED: cap-power-mde-rel-reg -->
0.1551
<!-- END GENERATED: cap-power-mde-rel-reg -->

The analytic number is a normal approximation, so it was checked against the
registered paired category bootstrap by simulation (120 replications, 400
resamples, 1% effect grid). At the analytic MDE the simulated probability of
satisfying both the CI and the 10%-relative decision rule is only:

<!-- BEGIN GENERATED: cap-power-decision-power -->
0.492
<!-- END GENERATED: cap-power-decision-power -->

so the analytic approximation is optimistic for *this decision rule*, not
conservative. The simulated 80%-power MDE (registered bootstrap, same model) is:

<!-- BEGIN GENERATED: cap-power-empirical-rel -->
0.14
<!-- END GENERATED: cap-power-empirical-rel -->

relative to the baseline, and with 20% of gains reversed:

<!-- BEGIN GENERATED: cap-power-empirical-rel-reg -->
0.18
<!-- END GENERATED: cap-power-empirical-rel-reg -->

**What this says.** The instrument's simulated 80%-power MDE (in the block above)
is *above* the registered 10% threshold, so the registered threshold cannot be
resolved at 80% power at this n. That is a statement about the instrument's
power; the registered threshold is unchanged and remains the decision rule. A
future expansion, a lower-variance endpoint or a larger n is what would change
it. The simulation is a model of the paired comparison, not measured arm data,
and it is labelled as such in the receipt.

**Dated correction, 2026-11-16 (S28).** The model above assumed a 0.5 baseline.
The measured baseline was 0.0 on all five arms (§12), so this analysis is not a
conservative estimate of the instrument's power — it describes an instrument
that did not exist. The numbers in this subsection remain as the historical
receipt of what was registered; the corrected power statement is §14's: it is
computed from a measured baseline and measured paired variance, and it is not
available until the calibration pass runs.

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

- **general coding ability.** These are authored repair tasks in the categories
  listed in §2. They are not a benchmark, they are not a random sample of
  programming work, and the item count is small. A confidence interval over
  items is not a claim about the population of software tasks. The suite size
  above is the only authoritative count in this document.
- **a difficulty ranking.** The easy/medium/hard labels are author-assigned and
  were not calibrated against any solver; they exist to keep the suite from
  being uniformly trivial, not to support a difficulty claim.
- **contamination-free pretraining.** The suite is disjoint from every
  registered evaluation source and from a bounded sample of the training corpus.
  No local test can prove a model never saw these patterns; disjointness is a
  property of *our* data, not of the checkpoint's pretraining.
- **that a Level 1 gain is a Level 3 product.** Level 3 needs offline
  responsiveness and a product-branch baseline; neither is measured here.
- **that the trio proves transfer.** It is a pre-registered probe with an author
  annotation, the surfaces and operation families listed in §3, not the P11.21
  instrument. With so few tasks per surface, an operation-family effect cannot be
  separated from a surface effect.
- **that the verifier is complete.** One declared mutation per task is a
  discrimination check, not a proof that the hidden tests reject every wrong
  program.
- **anything about latency or memory.** Court F owns those numbers; this
  instrument reports task success and, at Level 2, consumes Court F receipts.

## 9. Remaining gaps, explicitly

1. **No arm has been evaluated on the redesigned instrument.** The five T2 arms
   were run against the v4 suite and scored 0.0000; §12 shows that result is an
   instrument floor. The calibrated formats (§13) and the calibration protocol
   (§14) are registered and CPU-verified but no arm has been run on them. The
   pre-pilot commands are §10 and §15.
2. **The trio is under P11.21's item count.** The per-surface and per-operation
   counts in §3 are small, the operation annotation is author-only, and there is
   no inter-annotator agreement.
3. **The per-category count is still small.** One item moves a category by the
   fraction implied by the per-category count in §2, and §7.1's simulated MDE is
   above the registered 10% threshold. The suite is no longer a smoke signal,
   but it is not a calibrated benchmark; item difficulty and discrimination are
   not estimated by any solver.
4. **The sealed set exists but is not independent.** The 24 sealed items were
   authored by the same process and the same author as the scored suite, so a
   sealed result controls for *tuning against these exact tasks*, not for the
   author's shared taste. No arm has been run against it; the custody gate and
   its tests are in place, and the freeze predates any model evaluation.
5. **Feedback repair loops are not implemented.** Feedback is recorded and
   parity-enforced; the loop that feeds compiler output back is future work.
6. **The corpus contamination check is a bounded sample**, not the full 754 MB
   cache. Coverage is in the receipt; a full pass needs a streaming implementation
   with a bounded shingle set.
7. **Mutation testing is shallow.** One authored wrong answer per task, plus one
   declared naive fix for one task per category. That is two independent wrong
   answers for eight tasks and one for the rest; a mutation campaign or
   property-based tests would strengthen the verifier further.
8. **No 3B–8B quantised baseline exists locally**, so the product branch is
   planned, not measured.
9. **The Rust-runtime adapter is untested end to end** — it needs the built
   binary and the 0.6B bundle; its interface and its accounting fields are
   covered by the CPU tests, but no generation has run through it.
10. **The HF adapter's residency and I/O fields are null with reasons.** The
    PyTorch path does not expose device read counters; that is a limitation of the
    adapter, not of the model.
11. **Two historical pairs exceed the similarity thresholds.** `deps-04`/`deps-05`
    and `deps-08`/`deps-09` share assertions and code structure above the
    registered thresholds. They are in the v3-registered suite; replacing them
    would change a registered fixture, so the screen reports them for a future
    supersession rather than silently accepting or silently rewriting them.
12. **The MDE exceeds the registered threshold at this n.** §7.1 reports a
    simulated 80%-power MDE above the registered 10% threshold. A null result near the threshold would therefore be
    under-powered, and must be reported as such rather than as evidence of no
    effect.
13. **The power model is a model.** The MDE is computed from a stated paired
    discordance model and the registered bootstrap, not from measured arms. It
    predicts what the instrument could resolve; it does not measure an effect.
14. **The v4 capability result is an instrument floor, not a capability result.**
    All five arms scored 0.0000 because the format demanded complete multi-file
    replacements. It ranks nothing (§12).
15. **The calibrated suite is not yet calibrated.** The derivation is verified
    with `cargo` and the protocol is registered, but item difficulty,
    discrimination and the ladder are unmeasured until the calibration pass
    runs. No capability claim may use the calibration split.
16. **The diagnostic metric is not capability.** `verified_diagnostic` proves an
    arm can name the right compiler error and that the named repair resolves the
    failure; it says nothing about producing the repair. It is reported
    separately and never pooled.
17. **The scored split is not derived yet.** It depends on which cells survive
    calibration, so its item count is not fixed in the registration.

## 10. Exact commands for the first real runs

**Dated note, 2026-11-16 (S28).** These commands produced the five-arm result
that §12 diagnoses as an instrument floor. They are retained exactly as run; the
commands for the redesigned instrument are §15.

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
  --t2-training-receipt "results/t2/$ARM/run.json" \
  --t2-mask-order legacy \
  --preregistration docs/capability-preregistration-v4.json \
  --out "results/capability/runs/t2-$ARM/generations.json"

# the same for D, M, X, and H* with
#   ARM=Hstar; --t2-arm 'H*'  --t2-config 'results/train/admission-configs/t2/H*.json'
```

**Load-time refusal (S16C).** The adapter no longer loads a checkpoint with an
unchecked `strict=False`. The allowlist is exact: the checkpoint must contain
exactly the arm's trainable tensors (shapes checked) and may omit exactly the
frozen base parameters. With `--t2-training-receipt`, the receipt must also agree
on the arm letter, the config digest, the admission identity, the checkpoint name
and digest, the checkpoint step, and — for routed arms (`X`, `H`, `H*`) — the
recorded `dispatch_mask_order`.

**Consequence for the four finished arms (updated 2026-11-16, S20).** Their
`run.json` receipts predate the `dispatch_mask_order` field, so the routed ones
(`X`, `H`) were refused by this check. That refusal is retained as the default:
the P12.1 audit showed the mask placement changes what the arm computed, so an
arm scored through this court must prove which placement produced it. `D` and
`M` carry no dispatch and are not subject to the rule.

**Provenance amendment (`--provenance-amendment`).** A routed arm trained before
the field may still be scored if a **separate** amendment certifies its mask
order from repository history. The adapter then accepts it under a named flag
and records the amendment digest in every downstream receipt:

```bash
scripts/court --gpu oracle -- python3 scripts/capability_court.py generate \
  --t2-arm X \
  --t2-checkpoint "results/artifacts/train/t2/X/step-001024.pt" \
  --t2-config "results/train/admission-configs/t2/X.json" \
  --t2-admission "results/train/admission/t2/X.json" \
  --t2-training-receipt "results/t2/X/run.json" \
  --t2-mask-order legacy \
  --provenance-amendment results/t2/PROVENANCE-AMENDMENT-X-and-H.json \
  --preregistration docs/capability-preregistration-v4.json \
  --out "results/capability/runs/t2-X/generations.json"
```

The amendment is never trusted by name. It must certify this arm, infer the same
mask order the adapter is building, bind the exact training receipt by sha256,
and reference a replay receipt whose digest still matches. Any failure refuses
the load with the reason named. The original receipts are not edited. For `X`
and `H`, `results/t2/PROVENANCE-AMENDMENT-X-and-H.json` certifies `legacy`: both
arms trained on the same pre-correction `modeling.py` blob (`e61ca0d`), and a
fixed-batch replay reproduces that combine bitwise. The sensitivity of a
capability result to the *corrected* mask is a separate, registered run that does
not retrain.

Step 2, check (CPU, rust court; the court needs `cargo` and the checker sandbox):

```bash
SELVEDGE_MEM=12g SELVEDGE_CPUS=8 scripts/court rust -- \
  python3 scripts/capability_court.py check \
  --generations "results/capability/runs/t2-$ARM/generations.json" \
  --preregistration docs/capability-preregistration-v4.json \
  --out "results/capability/runs/t2-$ARM/result.json"
```

A stratified subset is explicit and still has to clear the category minimum:

```bash
SELVEDGE_MEM=12g SELVEDGE_CPUS=8 scripts/court rust -- \
  python3 scripts/capability_court.py check \
  --generations "results/capability/runs/t2-$ARM/generations.json" \
  --category lifetime-borrow \
  --preregistration docs/capability-preregistration-v4.json \
  --out "results/capability/runs/t2-$ARM/result-lifetime.json"
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
  --preregistration docs/capability-preregistration-v4.json \
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
The subset is a subset *by construction*: the check stage requires every
selected category and at least the registered per-category minimum, so a subset
that small cannot produce a primary composite. It measures the instrument's
integration, never a capability claim.

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
    --preregistration docs/capability-preregistration-v4.json \
    --out results/capability/instrument-smoke-$STAMP/generations.json &&
  python3 scripts/capability_court.py check \
    --generations results/capability/instrument-smoke-$STAMP/generations.json \
    --preregistration docs/capability-preregistration-v4.json \
    --out results/capability/instrument-smoke-$STAMP/result.json"
```

The committed v3 smoke run is
`results/capability/instrument-smoke-20261116T080500Z/`: the scripted arm emits
the reference solutions, every real checker runs, the composite and all eight
per-category intervals are printed, and the category minimum is met. Its receipt
is evidence that the pipeline works, not evidence about any model. The historical
24-task v1 smoke run `results/capability/instrument-smoke-20261116T023255Z/` is
retained unchanged.

## 12. The instrument floor: failure taxonomy of the stored T2 receipts (S28)

The five-arm T2 run scored 0.0000 on every arm. Before any redesign, every
failure was classified from the receipts already in the repository — the five
`results/capability/runs/t2-*/generations.json` files and their `result.json`
checker receipts. **No model forward pass ran for this section.** The generated
report is `results/CAPABILITY-FAILURE-TAXONOMY.md`; its receipt is
`results/capability/taxonomy-*.json`; `diagnose --check` regenerates the report
byte for byte from the receipt.

### 12.1 The registered classes

One primary class per arm/item, plus flags (`truncated`, `degenerate_repetition`,
`empty`, `wrong_file`, `stub`, `unchanged_initial`, `unbalanced_braces`):

| class | meaning |
| --- | --- |
| `pass` | the checker passed |
| `tests_failed` | compiled, a hidden test failed |
| `non_compiling` | reached the checker, did not compile |
| `timeout` / `oom` / `nonzero_exit` | the checker bound fired |
| `empty_answer` | no characters |
| `refusal` | an explicit refusal, no code |
| `degenerate_repetition` | one line repeated at least eight times |
| `truncated` | unterminated fence, or the token cap with no closed answer |
| `wrong_file` | only harness-owned or unwritable paths were returned |
| `duplicate_path_rejected` | the strict rule rejected a repeated file block |
| `no_code_block` | prose or a non-Rust block, nothing extractable |

The classification is recomputed from the stored text, so a historical receipt
keeps its own status and gains a taxonomy class beside it; a receipt written
after S28 carries `failure_class` directly and `diagnose` uses it instead.

Two views are reported and they are not the same view. The **strict rejection
reason** is what the registered extraction rule said when it discarded an answer
("no code block and no recognisable Rust body", a duplicate path, an unwritable
path). The **fine class** asks *why* that answer was unusable (degenerate
repetition, an unterminated fence, an empty answer). Every answer the strict rule
rejected for having no code block was also degenerate or truncated, so the fine
class count for a bare no-code answer is zero while the strict reason count is
not; §12.2 quotes both, labelled.

### 12.2 What the receipts contain

Across the five arms.

Arm/item outcomes classified:

<!-- BEGIN GENERATED: cap-tax-outcomes -->
765 arm/item outcomes
<!-- END GENERATED: cap-tax-outcomes -->

Outcomes that passed:

<!-- BEGIN GENERATED: cap-tax-passes -->
0
<!-- END GENERATED: cap-tax-passes -->

Answers that reached the checker and did not compile:

<!-- BEGIN GENERATED: cap-tax-noncompiling -->
293
<!-- END GENERATED: cap-tax-noncompiling -->

Answers the strict rule rejected with *no code block*:

<!-- BEGIN GENERATED: cap-tax-no-code -->
459
<!-- END GENERATED: cap-tax-no-code -->

Multi-file tasks on which **no** arm produced a compiling answer:

<!-- BEGIN GENERATED: cap-tax-multifile-terminal -->
46
<!-- END GENERATED: cap-tax-multifile-terminal -->

out of the multi-file tasks in the suite:

<!-- BEGIN GENERATED: cap-tax-multifile-tasks -->
46
<!-- END GENERATED: cap-tax-multifile-tasks -->

The per-arm table, the per-category and per-shape tables and the author
difficulty table are in the generated report and are not restated here.

### 12.3 Is the failure the parser, the model, or the format?

Three checks separate those explanations, and only one of them needs `cargo`.

**Tolerance.** The strict extraction rule discarded answers that a tolerant
rule reads (a `path=` line written inside the fence — the observed `M` shape —
and repeated file blocks). The tolerant rule recovers this many
otherwise-unreadable answers:

<!-- BEGIN GENERATED: cap-tax-recovered -->
242
<!-- END GENERATED: cap-tax-recovered -->

Re-judging those recovered answers with the real checker (a CPU pass over stored
text, no forwards) produced this many passes:

<!-- BEGIN GENERATED: cap-tax-recovered-passing -->
0
<!-- END GENERATED: cap-tax-recovered-passing -->

So the parser is not what made the baseline zero. It is still a real defect and
the calibrated suite uses the tolerant policy — but it cannot be the explanation
of the floor.

**Answer shape.** The dominant fine classes are answers that never reach the
checker at all (`degenerate_repetition`, `truncated`), and the strict rule's
reason for discarding them is "no code block". Inspection of the stored text
shows the arms echoing the instruction's test command ("The tests must be run
with `cargo test ...`") or repeating a block until the token cap; this is a
format-comprehension failure, not a wrong fix.

**Reach.** Of 153 tasks, only a handful ever saw a compiling answer: one `H` and
one `H*` item compiled and failed a hidden test. Every multi-file task was
terminal for every arm (the count above).

The conclusion is therefore **not** "the model cannot do these repairs". It is
that this instrument, with this format, at this model scale, measures whether an
arm can emit a complete multi-file Rust crate from scratch — and the answer is
no, uniformly. The 0.0000 is an **instrument floor**: it ranks nothing, its
confidence intervals are degenerate, and the pre-registered relative comparison
is undefined. The registered power analysis assumed a baseline of 0.5; the
measured baseline is 0.0, so that analysis is invalid rather than optimistic.

### 12.4 Structurally out of reach versus merely hard

* **Structurally out of reach in this harness:** every multi-file source (the
  count above). No arm compiled any of them, at any difficulty label, in any
  category. The required output is a whole crate replacement, and the observed
  outputs are both shorter than the reference and structurally malformed.
* **Merely hard:** the evidence does not establish such a tier *within the v4
  format*. Single-entry tasks also produced almost no compiling answer, and the
  failures are concentrated in parse- and shape-level classes rather than in
  near-miss behavioural fixes.
* **What would measure "merely hard":** a format whose required output is small
  enough for the arm to produce, with the crate otherwise solved. That is §13.

## 13. The calibrated formats (S28)

Four formats, three executable and one verified separately. The design rule is:
**the arm is asked only for the piece the source task is about, and every claim
about behaviour is decided by `cargo`.**

| format | tier | the arm returns | judged by | metric |
| --- | --- | --- | --- | --- |
| `body-completion` | easy | one function body | spliced into a template with the signature; `cargo test` | executable |
| `patch-selection` | medium | one candidate letter | the chosen function definition, file or manifest is applied; `cargo test` | executable |
| `file-replacement` | hard | complete files | `cargo test`, unchanged from v4 | executable |
| `diagnostic-identification` | diagnostic | the diagnostic identifier / option letter | that option's registered consequence is applied and compiled | `verified_diagnostic` |

**Why each format.** `body-completion` removes the retyping burden and keeps the
behavioural question: the crate is otherwise solved, the signature is given, and
one visible test fails. `patch-selection` removes generation entirely while
keeping a real decision: several plausible candidates are shown, exactly one
passes, and which one passes is decided by `cargo` — not by an author label.
`file-replacement` is kept only as the hardest tier, including the multi-file
sources, so the old format still exists where it is appropriate.
`diagnostic-identification` measures the one thing a 0.6B arm can plausibly do on
a compile error when it cannot write the fix, and it is **not** executable, so it
is reported as `verified_diagnostic` and never pooled with task success.

**Derivation, not new authoring.** The calibrated items are derived from the same
registered source tasks (`oracle/capability/derive.py`), so the hidden tests,
the reference behaviour, the licence and the category semantics are the source's:

- `body-completion`: the reference file with the target function's body replaced
  by `todo!()`; the signature is shown, plus one visible test. The rest of the
  crate is handed over solved, so the item cannot be blamed on files the arm was
  not asked about.
- `patch-selection`: 3–5 candidates — the reference definition, the source's
  initial definition, the source's declared mutation and naive fix, a
  single-edit revert, and operator inversions when a source offers fewer than
  three. For a source whose fix is in the manifest or in no single function, the
  target is the manifest or the whole file instead of a function. Every candidate
  is compiled; a candidate that the checker shows *also* passes is pruned and the
  pruning is recorded in the verification receipt.
- `file-replacement`: the source task copied under a new id.
- `diagnostic-identification`: only for sources whose failing state is a compile
  error with an `E####` code; the correct option's consequence is the reference
  repair, and a wrong option's consequence is a registered repair that does not
  resolve the error.

**Visible-test rule.** The visible test in a body or patch item is one test taken
from that item's own hidden set, together with any helper functions in that file,
with the crate's self-import removed. The source's own inline test is *not* used:
in the semantic-repair sources that test encodes the old behaviour and passes in
the failing state. The remaining hidden tests are still restored by the harness,
so the visible test cannot stand in for the hidden set.

**Splits.** Sources are partitioned by `sha256('selvedge-calibration-v1:<split>:<id>')`
ascending — a rule fixed before the split was computed — so the calibration and
scored splits are disjoint by construction and the sealed holdback (a separate
collection, never read by the derivation) is excluded by an explicit assertion.

**The calibration split.** Three sources per category, each derived into every
format it can support. Items in the calibration split:

<!-- BEGIN GENERATED: cap-cal-items -->
84 calibration items
<!-- END GENERATED: cap-cal-items -->

By format — `body-completion`:

<!-- BEGIN GENERATED: cap-cal-body -->
23
<!-- END GENERATED: cap-cal-body -->

`patch-selection`:

<!-- BEGIN GENERATED: cap-cal-patch -->
23
<!-- END GENERATED: cap-cal-patch -->

`file-replacement`:

<!-- BEGIN GENERATED: cap-cal-file -->
24
<!-- END GENERATED: cap-cal-file -->

`diagnostic-identification`:

<!-- BEGIN GENERATED: cap-cal-diagnostic -->
14
<!-- END GENERATED: cap-cal-diagnostic -->

Every executable cell (category × format) has at least two items; the receipt
`fixtures/capability-calibrated/DERIVATION.json` records the exact counts and
the two sources the derivation could not use for a patch item. The derivation
contract is bound to the source suite hash and to every source tree hash, so a
fixture edit is a failed `derive-formats --check`, not a silent drift.

**Verification.** `verify-formats` compiles every derived item and proves, per
item: the failing state fails as declared, the reference passes, and a mutated
reference fails; per patch item: exactly one candidate passes and it is the
declared reference; per diagnostic item: exactly one option's consequence
resolves the failure. The receipt is
`results/capability/derivation-verify-calibration.json` and its verdict is the
only evidence that the derived suite is an instrument.

**Instrument smoke (scripted arms, not a model).** The staged path was also run
end to end through the real checkers with `fake-calibrated`, which writes answers
in each format from the fixture (an instrument check, never an arm result). The
receipts are under `results/capability/calibrated-smoke-*/` and the numbers are
not restated here:

- **correct answers** — the executable composite and `verified_diagnostic` are
  both at their maximum on their own metrics, with complete partial credit;
- **deliberately wrong answers** — the executable composite is zero, the answers
  split across non-compiling, compiled-and-failed and stub classes, and the
  **secondary** partial-credit mean is non-zero: the separation is visible in one
  receipt;
- **degenerate answers** (a repeated line, the observed T2 failure shape) — the
  composite is zero and every item is classified `degenerate_repetition` without
  reaching the checker.

Those receipts are evidence that the formats, the taxonomy and the metric
separation work. They say nothing about any arm.

## 14. Calibration registration and acceptance window

Registered in `docs/capability-calibration-preregistration-v1.json` before the
calibration pass. The registration binds the formats, the ladder hypothesis, the
split rule, the window, the statistics, the retirement rule, the metric
separation and the power formula. `court calibrate` refuses to apply a window
that differs from the registered one.

**Five arms, one pass.** The calibration split is run on `D`, `M`, `X`, `H` and
`H*`, greedy, with the same decoding as §10(a), on the calibration split only.
The sealed holdback is never opened. The output is an *item matrix*
(item × arm), not an arm comparison.

**Per item.** Pass probability (mean executable outcome over the five arms),
partial-credit fraction (secondary), partial-credit spread (population standard
deviation across arms) and item–total correlation (point-biserial against each
arm's total with the item removed; reported with the caveat that five arms make
it low-precision).

**Acceptance window.** An item is kept when it is **not zero and not
saturated** — with five binary arms that is exactly one to four passes — **and**
it discriminates: spread at least 0.15 or correlation at least 0.30. Otherwise it
is retired with a named reason. A cell is *reformatted* when fewer than two items
survive or more than a third are retired; a category is scorable only when every
executable cell is scored; a calibrated suite version is emitted only when every
category is scorable **and** the measured ladder is monotone.

**The ladder is a hypothesis.** Tiers are `easy = body-completion`,
`medium = patch-selection`, `hard = file-replacement`, and the registration
states the expected order. The calibration pass measures `1 - mean pass rate`
per cell; a violation refuses the suite version and names the categories, rather
than relabelling them after the fact.

**Primary and secondary metrics, separated.**

- **Primary:** verified task success (executable formats only). The endpoint and
  decision rule are the registered ones (10% relative, paired 95% CI above zero,
  no category regressing by more than 0.05 absolute) — §7 is unchanged.
- **Secondary:** the fraction of hidden tests passed, recorded per item and
  reported per cell. It is **never** substituted for the primary metric and is
  not an input to any decision rule; the CPU tests assert that adding it leaves
  the composite unchanged and that it is flagged `primary: false`.
- **Separate:** `verified_diagnostic`, its own block with its own item counts,
  never pooled with executable success. The CPU test that a passing diagnostic
  leaves the primary composite at zero is the guard.

**Power cannot be finalised in this session.** The registered formula is

    n_per_category = (z_{1-α/2} + z_power)² · Var(d) / (categories · δ²)

with `δ = baseline × 10%` and `d` the per-item paired difference. Every input is
measured, and none exists yet: the baseline is not 0.5 (it was 0.0 on the old
instrument), and Var(d) is the variance of the measured per-item paired
differences over the ten arm pairs. The calibration receipt therefore reports the
measured baseline, the pairwise variances, `required_n` under both the planning
(median) and conservative (maximum) variance, the items available per category
and the shortfall. If the required n exceeds the available items the instrument
is under-powered at that n and P14.8 stays open — the shortfall is published, not
absorbed into a claim.

## 15. Exact commands for the calibration pass

None of these was run in this session; the derivation, its verification and the
taxonomy were (all CPU). The GPU is free, so these are the next session's
commands. The sealed holdback is never named by any of them.

Step 0 — re-derive the calibration split and verify it with `cargo`
(CPU; reproducible from the registered source suite):

```bash
SELVEDGE_MEM=4g SELVEDGE_CPUS=4 SELVEDGE_PIDS=512 SELVEDGE_TMP=1g \
scripts/court rust -- python3 scripts/capability_court.py derive-formats \
  --out fixtures/capability-calibrated --split calibration --per-category 3 --force

SELVEDGE_MEM=6g SELVEDGE_CPUS=4 SELVEDGE_PIDS=512 SELVEDGE_TMP=2g \
scripts/court rust -- python3 scripts/capability_court.py verify-formats \
  --root fixtures/capability-calibrated --prune \
  --out results/capability/derivation-verify-calibration.json
```

Step 1 — size the generation limit from a measured peak, not from a habit. The
same adapter ran the 153-item v4 generation at `SELVEDGE_MEM=24g` with exit 0 and
no OOM (`results/courts/oracle-20261116T122455Z-e627a1.json`), which is the
sizing basis; the smoke run below re-measures it, and the launch record must
carry the measured peak. Do not start this while another memory-heavy job runs:
check `MemAvailable` and refuse if `limit + running peak` exceeds it.

```bash
# 1a. smoke: 8 items, measure the cgroup peak, then set SELVEDGE_MEM from it.
# `--suite` is a top-level option: it must come before the subcommand.
SELVEDGE_MEM=24g SELVEDGE_CPUS=8 SELVEDGE_PIDS=1024 SELVEDGE_TMP=4g \
scripts/court --gpu oracle -- python3 scripts/capability_court.py \
  --suite fixtures/capability-calibrated generate \
  --t2-arm 'H*' \
  --t2-checkpoint results/artifacts/train/t2/Hstar/step-001024.pt \
  --t2-config 'results/train/admission-configs/t2/H*.json' \
  --t2-admission 'results/train/admission/t2/H*.json' \
  --t2-training-receipt 'results/t2/H*/run.json' \
  --t2-mask-order legacy \
  --provenance-amendment results/t2/PROVENANCE-AMENDMENT-Hstar.json \
  --preregistration docs/capability-calibration-preregistration-v1.json \
  --only 'borrow-05-temporary-lifetime--body,borrow-05-temporary-lifetime--patch' \
  --out results/capability/calibration-smoke/generations-Hstar.json
```

Step 2 — generate on the calibration split, one command per arm (GPU). `H*` is
`Hstar` on disk and `H*` in the config name; `X` and `H` need their certified
provenance amendment, `D` and `M` do not.

```bash
# repeat for ARM in D M X H; H* is shown because it needs the amendment
ARM='H*'; DIR=Hstar
CKPT="results/artifacts/train/t2/$DIR/step-001024.pt"
CFG="results/train/admission-configs/t2/$ARM.json"
ADM="results/train/admission/t2/$ARM.json"
REC="results/t2/$ARM/run.json"
mkdir -p "results/capability/calibration/$DIR"

SELVEDGE_MEM=24g SELVEDGE_CPUS=8 SELVEDGE_PIDS=1024 SELVEDGE_TMP=4g \
scripts/court --gpu oracle -- python3 scripts/capability_court.py \
  --suite fixtures/capability-calibrated generate \
  --t2-arm "$ARM" \
  --t2-checkpoint "$CKPT" \
  --t2-config "$CFG" \
  --t2-admission "$ADM" \
  --t2-training-receipt "$REC" \
  --t2-mask-order legacy \
  --provenance-amendment results/t2/PROVENANCE-AMENDMENT-Hstar.json \
  --preregistration docs/capability-calibration-preregistration-v1.json \
  --out "results/capability/calibration/$DIR/generations.json"
```

For `D` and `M` drop the `--provenance-amendment` line; for `X` and `H` use
`results/t2/PROVENANCE-AMENDMENT-X-and-H.json`.

Step 3 — check on CPU, with partial credit (the secondary metric):

```bash
SELVEDGE_MEM=12g SELVEDGE_CPUS=8 SELVEDGE_PIDS=1024 SELVEDGE_TMP=2g \
scripts/court rust -- python3 scripts/capability_court.py \
  --suite fixtures/capability-calibrated check \
  --generations results/capability/calibration/Hstar/generations.json \
  --preregistration docs/capability-calibration-preregistration-v1.json \
  --partial-credit \
  --min-category-items 0 \
  --out results/capability/calibration/Hstar/result.json
```

`--min-category-items 0` is correct here and only here: calibration is not a
capability result and the category-minimum rule belongs to the scored split. The
receipt records the zero, and `calibrate` applies the real cell minimums.

Step 4 — fit item difficulty, apply the window, emit the calibrated suite
version (CPU):

```bash
SELVEDGE_MEM=6g SELVEDGE_CPUS=4 SELVEDGE_PIDS=512 SELVEDGE_TMP=1g \
scripts/court rust -- python3 scripts/capability_court.py \
  --suite fixtures/capability-calibrated calibrate \
  --registration docs/capability-calibration-preregistration-v1.json \
  --result D=results/capability/calibration/D/result.json \
  --result M=results/capability/calibration/M/result.json \
  --result X=results/capability/calibration/X/result.json \
  --result H=results/capability/calibration/H/result.json \
  --result 'H*=results/capability/calibration/Hstar/result.json' \
  --out results/capability/calibration-20261117T000000Z.json \
  --suite-version-out fixtures/capability-calibrated/VERSION.json

# and its fail-closed re-check
scripts/court python -- python3 scripts/capability_court.py \
  --suite fixtures/capability-calibrated calibrate --check \
  --out results/capability/calibration-20261117T000000Z.json \
  --suite-version-out fixtures/capability-calibrated/VERSION.json
```

Step 5 — only after a version exists: derive and verify the **scored** split
(the remaining 120 sources), which is the suite a capability claim may use:

```bash
SELVEDGE_MEM=4g SELVEDGE_CPUS=4 SELVEDGE_PIDS=512 SELVEDGE_TMP=1g \
scripts/court rust -- python3 scripts/capability_court.py derive-formats \
  --out fixtures/capability-scored --split scored --per-category 3 --force
# the calibrated version's accepted cells decide which formats are scored; the
# derivation receipts and the VERSION.json together are the instrument version.
```

The calibration pass must establish, at minimum: a measured pass probability and
partial-credit spread per item; which cells survive and why the others were
retired; whether the ladder is monotone; the measured baseline and paired
variance and therefore the required n; and either a calibrated suite version or
an explicit refusal. It must not open the sealed set, change the T2 verdict, or
report an arm comparison.

## 16. What the stored receipts cannot answer

1. **Per-item truncation.** The stored generation receipts recorded
   `finish_reason: "stop"` for every item and the *batch's* decoded width as
   `tokens_out`, so a 3072-token cut could not be told from an EOS stop. §12 uses
   the answer text (unterminated fence) as the direct evidence and the cap as
   corroboration. The adapter now records `finish_reason` (`length`/`eos`/`stop`)
   and a per-item token count, so the calibration receipt will not have this gap.
2. **A per-item memory peak.** The generation court receipts record the requested
   limits, not `memory.peak`; the calibration smoke run is what produces the
   measured peak the launch record needs.
3. **Whether an arm could do these tasks with a different prompt.** The
   redesign changes the format and the prompt together. A future ablation could
   separate them; nothing here claims the prompt alone caused the floor.
