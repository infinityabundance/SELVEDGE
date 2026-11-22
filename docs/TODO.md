# TODO — S70 onward

This is the executable backlog. It is not prose: every row names what would
prove the item done, what it waits on, and where it stands. When a row quotes a
result, it cites the receipt path — not a number. Numbers live in receipts and
in the generated regions of `README.md` / `results/REPORT.md`; hand-writing one
here is a defect.

## How to use this file

**Statuses.**

| status | meaning |
|---|---|
| `DONE` | The named test or receipt exists at the named path and a court re-derives it. The row says where. |
| `READY` | Unblocked and specified; implementation or execution can start now. |
| `BLOCKED` | A named dependency is not met. The row names the dependency; do not start early. |
| `CANDIDATE` | A research direction with a stated decisive experiment; not yet registered or implemented. |
| `HYPOTHESIS` | Cannot be specified until another item reports, or rests on a claim this program has not established. |

**Thresholds never move.** Every win bar, retention bound, selection rule and
sample-size rule is fixed by the registration that precedes the run. A failed
bar is recorded as a failure; changing it requires a new, dated registration
committed before the next score, never an edit to the old one.

**Four courts before any push.**

```sh
SELVEDGE_MEM=2g SELVEDGE_CPUS=2 SELVEDGE_PIDS=512 SELVEDGE_TMP=1g \
  scripts/court status_consistency
SELVEDGE_MEM=2g SELVEDGE_CPUS=2 SELVEDGE_PIDS=512 SELVEDGE_TMP=1g \
  scripts/court perf_numbers
SELVEDGE_MEM=2g SELVEDGE_CPUS=2 SELVEDGE_PIDS=512 SELVEDGE_TMP=1g \
  scripts/court prereg_provenance
SELVEDGE_MEM=2g SELVEDGE_CPUS=2 SELVEDGE_PIDS=512 SELVEDGE_TMP=1g \
  scripts/court commit_identity
```

Any GPU job additionally names its own measured-peak-derived `SELVEDGE_MEM`;
the limits above are for the CPU-only courts and are not a license to reuse them
for a workload-sized run.

**One experiment at a time.** A row marked `BLOCKED` is not run because a slot
is free. It is run when its dependency's receipt exists and the GPU/RAM budget
is reserved for it, per `AGENTS.md`.

## A. Integrity and bookkeeping

| ID | Item | Acceptance (what proves it done) | Depends on | Status |
|---|---|---|---|---|
| **A1** | Fix the S69 difficulty-weight cancellation in the preference loss: normalise by a fixed corpus-level mean weight (or sample in proportion to difficulty) instead of dividing each batch by that batch's own weight sum (`oracle/frontier/preference.py:396`), which is identically `L_i` at `batch_pairs = 1` (`scripts/s64_preference_launch.py:427`). | A unit test proves two pairs with different `w` produce gradients scaled by `w_i / mean_w`, and that a one-pair batch no longer degenerates to the unweighted loss. The test lives beside `oracle/tests/test_s63_preference.py` and passes through the oracle court. The training receipt records the corpus mean weight and the normalisation actually used. | — | `READY` (implemented; training receipt blocked on B1) — `oracle/frontier/preference.py` supplies `weight_normalizer` and `pair_stream(order="weighted")`, the launcher records `corpus_mean_weight`, `weight_normalization` and the realized mean weight. Tests: `oracle/tests/test_s70_preference_weighting.py`, receipt `results/courts/oracle-20261122T023758Z-3450fa.json`. The run receipt `results/stage2/s69-preference/receipt-preference.json` does not exist yet. |
| **A2** | Add a matched-**compute** control beside the existing matched-step control: the preference arm forwards chosen and rejected continuations, the SFT arm forwards chosen only, so equal steps are not equal work (`scripts/s64_preference_launch.py:311-325`, `oracle/frontier/preference.py:619-631`). | A receipt records both arms' processed forward-token positions and wall time; a test asserts the matched-compute arm consumes the same forward-token total as the preference arm at the selected cadence. The verdict reports both the matched-step and matched-compute comparisons. | A1 | `READY` (basis implemented and receipted on the S64 pair corpus; the S69 arm receipt is blocked on B1) — `scripts/s69_control_budget.py --basis {steps,compute}` emits both bases. Receipts: `results/stage2/s70-control-bases/control-budget-s64-pairs.json` and `results/stage2/s70-control-bases/control-budget-s64-pairs-weighted.json`; the compute-basis test is `oracle/tests/test_s70_preference_weighting.py::test_compute_basis_ratio_is_the_analytic_value`. Wall time and the verdict comparison wait on the S69 run. |
| **A3** | Publish the S69 rollout corpus reproducibly: gzip the accepted/ledger/rejected/prompts files (or use hash-verified external storage) and add `raw/INDEX.json` entries, because tracked docs reference receipts whose raw artifacts are not in the tree. | `results/artifacts/stage2/s69-rollouts/raw/INDEX.json` (or the tracked equivalent) lists every file with its sha256, byte count and provenance; `git ls-files` shows the compressed artifacts and the index; every path named by the S69 receipts resolves, and no file exceeds the GitHub per-file limit. | — | `READY` |
| **A4** | Rust executed-set audit (still pending from earlier phases). | An executed-set receipt shows the declared active set is the set actually read per token, with the auditor's own tests. A parameter-count certificate is not sufficient (`docs/S67-RESULT.md` §8). | P19.0 executor | `BLOCKED` |
| **A5** | Publish the S65/S67 champion checkpoint to a public model hub and record the SHA-256 in `docs/S65-RELEASE.md`; a local-only release contract is not a public release (`docs/S65-RELEASE.md:3-5,81-82`). | A public hub URL plus the artifact SHA-256 appears in `docs/S65-RELEASE.md`; a fresh download re-hashes to the same value. | Operator credentials | `BLOCKED` |
| **A6** | Fix repo status text in generated regions only: `README.md` / `results/REPORT.md` must state the replicated GSM8K win and the unresolved MATH-500 clause accurately (never "no capability win"); numbers come from receipts via the generators, never by hand. The current prose at `README.md:94-98` still says no arm produced a capability win. | The `status_consistency` court passes and the generated block plus surrounding prose agree with `results/stage2/s67-final/verdict.json`; a search for the contradicting sentence finds none outside a historical quotation that names its receipt and revision. | S67 verdict (exists) | `READY` |

## B. S69 execution

| ID | Item | Acceptance (what proves it done) | Depends on | Status |
|---|---|---|---|---|
| **B1** | Run preference training from the S67 soup champion (never the post-trained baseline): rank-64 all-linear LoRA, frozen-reference log-probs from the same init, teacher continuation on the student's exact prefix as chosen, student failure as rejected; merge, then verify the merged round trip. | `results/stage2/s69-preference/receipt-preference.json` names the init checkpoint and its SHA-256 (matching the registration), the geometry, the effective pair set and the zero-weight drop list; every exported checkpoint carries the merge report; a round-trip test reloads the merged full state and reproduces the adapter's logits within the registered tolerance. | A1, teacher pass, reference pass | `BLOCKED` |
| **B2** | Score the chosen checkpoint **once** on the test instruments; select only on the 1,000-item calibration splits. Win bar: beat the S67 soup on MATH-500 **and** full MATH with paired 95% intervals, with no GSM8K or MMLU regression. | `results/stage2/s69-calibration/` selection receipt (MATH-cal first, GSM8K-cal tie-break, MMLU-cal filter) exists before any test read; the one-shot verdict cites paired problem-level bootstrap against the stored soup dumps and reports MATH-500, full MATH, GSM8K and MMLU separately. A failed gate produces a calibration-level failure receipt and no test read. | B1 | `BLOCKED` |
| **B3** | Adaptive rollout budget: after K initial attempts, add attempts on uncertain or apparently-unsolvable problems, to reduce the false-"unsolvable" rate at `p = 0.2, K = 4` (quote the current rate from the S69 rollout analysis receipt; do not restate it here). Record the sampling policy in the per-problem ledger. | A rerun rollout receipt shows the adaptive policy, the extra-attempt counts, and the reduced false-unsolvable rate computed from the ledger; the ledger's `sampling_policy` field distinguishes initial and top-up attempts. | B4 | `READY` |
| **B4** | Per-problem ledger with `p` computed from attempts (not the deduplicated accepted count), storing attempts, verified successes, failures, truncations, duplicate successes and the sampling policy. | A ledger-schema test (`scripts/s59_student_rollouts.py` and its oracle test) asserts every field is present on every row, that `p = verified_correct / attempts` includes truncations and duplicates, and that failures are aggregated without collapsing verdict types. Existing rows already carry attempts/`p`/truncation/duplicate counters; the missing fields are `sampling_policy`, an aggregate `failures`, and an explicit `duplicate_successes`. | — | `READY` |

## C. Capability research

| ID | Item | Acceptance (what proves it done) | Depends on | Status |
|---|---|---|---|---|
| **C1** | BREAD: genuinely local reasoning-branch corrections — branch-local chosen/rejected pairs at the first divergence point, not whole-response preferences. | A registration fixes the divergence-point rule and the pair schema; a unit test builds two branches that share a prefix and diverge at one token; the run receipt reports branch pairs separately from whole-response pairs and the capability delta against the same champion. | B1 (objective and harness) | `CANDIDATE` |
| **C2** | Transfer the winning objective to Qwen3-0.6B (which leads the champion on MATH-500 under our own instrument; cite `results/stage2/s68-frontier/summary.json`, not a number) and to the S67 champion backbone. | A transfer registration exists before scoring; each backbone gets the same pairs, budget and selection rule; receipts report calibration selection and the one-shot comparison separately per backbone. | B2 | `CANDIDATE` |
| **C3** | CURE: counterfactual utility-routed residual operators; outcome-gated, paired intervention vs bypass continuations, measured as capability per byte and per FLOP. | A registration defines the intervention/bypass pair, the gate, the cost accounting and a kill criterion; the receipt reports stored parameters, active parameters, loaded bytes and executed computation separately (`C6`); a negative result is recorded as a negative. | C6 doctrine | `CANDIDATE` |
| **C4** | Conditional latent-rank expansion: nested rank groups (bypass/32/128/256 as illustrative modes), gated by counterfactual verified utility minus cost; unused groups must never be fetched. | A receipt shows the executed-set proof that an unselected rank group is not read, plus the utility gate's paired comparison at matched active bytes; the cost term is measured, not assumed. | A4, C6 | `CANDIDATE` |
| **C5** | LittleBit-2 selective integration, staged: (a) Dual-SVID initialisation plus Joint-ITQ rotation of *additional* corrective operators, never whole-backbone sub-1-bit; (b) compress the proven S69 LoRA delta and require the compressed operator to preserve the demonstrated capability gain; (c) binary vs BF16 vs 4-bit vs unrotated binary at equal stored bytes; (d) multiple specialised binary residual experts vs the best single operator; (e) outcome-trained conditional rank expansion vs static rank and always-on at equal compute; (f) packed binary inference in Rust with signed-accumulate kernels (XNOR-popcount only for quantised activations), measuring total tokens, repetition, truncation, latency and solved-per-second. | One receipt per stage. Stage (b) cannot pass by matching the BF16 operator — it must preserve the *demonstrated gain* from B2. Stage (f) reports the four accounting fields plus the behavioural counters; a latency win that inflates reasoning length is reported as such. | A4, B2, C6 | `CANDIDATE` |
| **C6** | Reporting doctrine for any compression: report stored parameters, unique active parameters, loaded bytes and executed computation separately. Binary parameters are parameters and may not disguise a larger model as sub-1B. The compressed operator must beat the strongest matched BF16 alternative, not merely match it. | Every C3/C4/C5 receipt carries the four fields and the matched-BF16 comparison; a review checklist in that receipt names the alternative and its cost basis. | — | `READY` |
| **C7** | Literature deep-dive tracked as an experiment-generating task, not a survey: identify mechanisms not yet combined, or combined in only one direction, and turn each into a cheap decisive experiment with a kill criterion. Seed directions: WiSE-FT / model soups and update-averaging theory (overshoot, sign/permutation interference, linear mode connectivity); DPO/KTO/SimPO and reference-anchored variants; reverse-KL vs forward-KL off-policy corrections and importance-weighted estimators; branch/prefix-level preference and tree-search preference optimisation; process reward models vs outcome rewards at small scale; verifier-gated self-training and reward hacking; test-time compute for sub-1B models (best-of-n, self-consistency, majority voting distilled into weights); lazy/adaptive depth and early exit under a parameter budget; sub-1-bit quantisation (LittleBit/LittleBit-2, SVDQuant, LoftQ, CALDERA, BitNet); conditional-computation routing with counterfactual utility. | `docs/S70-LITERATURE.md` states, per direction: what is established, what is not, what remains uncombined, the decisive experiment, and the kill criterion; every source resolves to a real citation; each surviving direction produces one registration or is explicitly rejected. | — | `CANDIDATE` |
| **C8** | Statistical power: register an instrument/sample-size rule before scoring so a ~3 pp design target is resolvable (full MATH as the resolver instrument; hierarchical seed×problem analysis for multi-seed claims). The ~3 pp figure is a design target, not a measurement. | A registration under `docs/` fixes the resolver instrument, the sample size and the analysis model before any score; a multi-seed claim's receipt reports a seed×problem hierarchical interval, not only per-seed intervals. The resolver half exists in `docs/s67-resolution-instrument-preregistration-v1.json`; the hierarchical multi-seed half does not. | — | `READY` |

## D. Deferred / do-not-start

| ID | Item | Acceptance (what proves it done) | Depends on | Status |
|---|---|---|---|---|
| **D1** | Latent KV-cache compression — deferred until Qwen3.5 hybrid attention, GQA and positional encoding get their own analysis. | A separate registration and analysis; not started before then. | Explicit un-defer decision | `BLOCKED` |
| **D2** | Whole-backbone sub-1-bit quantisation. | A separate registration; C5(a) explicitly forbids this route. | Explicit un-defer decision | `BLOCKED` |
| **D3** | New architectural implementation programs before A1 and B1 report. | A1's test and B1's run receipt exist; until then this row stays closed. | A1, B1 | `BLOCKED` |
