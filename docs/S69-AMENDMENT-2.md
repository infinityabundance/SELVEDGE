# S69 — Amendment 2: adapter determinism, the weighted SFT control, and the MATH instrument

Status: **registered before any calibration or test score, but after one
training arm was run and superseded.** This amendment amends
`docs/s69-preference-from-champion-preregistration-v1.json` and
`docs/S69-AMENDMENT-1.md` without changing the objective, the primary arm, the
budget, the cadence, the calibration selection rule, the adoption gate or the
win bar. The machine-readable form is
`docs/s69-amendment-2-adapter-init-and-controls-v1.json`.

The operator's S70 review found four bounded defects and one roadmap
refinement. Each is corrected here with the smallest change that removes the
defect. None of them is a result, and none is allowed to move a threshold.

## 1. The adapter initialisation is seeded and provable

### Defect

`LoRALinear.__init__` draws the `A` factor with `kaiming_uniform_` from the
global `torch` RNG. Nothing seeded that RNG immediately before
`install_lora`, so two runs with the same `--seed 6900` could start from
different random adapters. Pair ordering uses `random.Random(seed)`, a
different stream; the run seed did not cover the adapter.

### Fix

`preference.install_lora(..., seed=...)` now seeds `random`, `numpy` (when
importable) and `torch` (including CUDA) immediately before the factors are
drawn. `prepare_registered_lora` passes the run seed and records

```json
"adapter_init": {"seed": 6900, "sha256": "<digest>", "basis": "..."}
```

in the arm's `geometry`, where the digest is a canonical SHA-256 over the name,
dtype, shape and bytes of every `lora_A` and `lora_B` in module order
(`preference.adapter_state_sha256`). This is evidence of the initial state,
not an assertion that a seed was set.

### Consequence for the arms already run

Arm 1 completed twice before this amendment (its host court wrapper was killed
with its terminal the first time). Neither receipt records the adapter digest,
so neither can prove its initial state. Both are retained under
`results/stage2/s69-preference/preference/superseded-20261122T025943Z/` and
`results/courts/oracle-20261122T031400Z-759e4e.json`; arm 1 is **re-run after
this amendment** and only the new receipt is used for selection. The
matched-SFT arm's first invocation was stopped at revision
`a68aeeebc9a07ad29e5eea01fb17df8fc53ebdb0`
(`results/courts/oracle-20261122T032856Z-3259ad.json`, exit 143, no OOM) for
the same reason and is re-run.

## 2. A difficulty-weighted SFT control (arm 4)

### Defect

The matched-SFT control is unweighted, so a preference-vs-SFT difference
conflates the objective with the weighting mechanism.

### Fix

`--sft-weighting {uniform,difficulty}` (default `uniform`, today's behaviour)
and a distinct control arm `matched-sft-weighted`. The weighted control
minimises

```
sum_i w_i CE_i / mean_w
```

on the same chosen continuations, with the same effective problem set, budget
and cadence as the preference arm; `mean_w` is the corpus mean positive weight.
It therefore differs from the preference arm only in the objective (CE versus
the reference-anchored margin), not in whether the difficulty weight is
realized. It is registered as an **additional control**: it does not change
the primary win bar, the calibration-selection rule, or which arm is primary.

The arm name is authoritative; the flag is validated against it and recorded.

## 3. The sizing smoke is traced

### Defect

The smoke that sized the arm was untraced, while the arms run with
`--trace-out`. Telemetry copies the trainable parameters once per step, so the
measured workload was not the run workload.

### Fix and measurement

Both a traced and an untraced 20-step smoke were run with the registered
geometry and seed at revision
`a68aeeebc9a07ad29e5eea01fb17df8fc53ebdb0`:

| smoke | court receipt | step time | peak reserved | limit basis |
|---|---|---|---|---|
| traced | `results/courts/oracle-20261122T033325Z-159124.json` | 639.5 ms | 10.55 GiB | 15.83 GiB |
| untraced | `results/courts/oracle-20261122T033350Z-e62755.json` | 618.2 ms | 10.38 GiB | 15.57 GiB |

The derived record is `results/stage2/s69-preference/trace-overhead.json`
(created `2026-11-22T03:34:30Z`, revision
`a68aeeebc9a07ad29e5eea01fb17df8fc53ebdb0`), written from those two receipts
and not typed. Trace overhead is **+3.44%** of step time
(21.30 ms/step). Both smokes produced the same adapter digest
`db1db8d97d7d82ec3b4086334e0c894f948f11e6b6bdae712987107d8638ff77`, which is
independent evidence that the seeding is effective and reproducible across
invocations.

The arm limit stays **16g**: the traced smoke's formula
`max(1.5 * 10.55, 10.55 + 1) = 15.83` rounds up to 16, and the completed
first arm's host cgroup peak was well below it. The launch record for each arm
states this basis.

## 4. Correction to the MATH instrument description

### Defect

`docs/S69-AMENDMENT-1.md` and its JSON described MATH-500 and the
5,000-problem full MATH as "two independent MATH instruments". MATH-500 is a
500-problem subset of the 5,000-problem MATH test set.

### Correction

They are **two evaluation scales of the same test set**. Their paired
intervals are correlated, not independent replication. The win bar's
requirement that both be above zero is a **resolution requirement** -- the
larger scale reduces sampling noise while the smaller is the historical
comparison -- not evidence from two independent samples. The win bar itself is
unchanged; both Amendment 1 files now carry the corrected wording and point to
this amendment.

## 5. BREAD refinement (roadmap only)

`docs/TODO.md` item C1 now defines a candidate divergence as a **decision whose
intervention measurably changes downstream correctness**, identified by paired
do/undo continuations from the same state, with wording-only divergence
explicitly rejected. Four corpus gaps found by the review are added as
explicit roadmap items:

1. adaptive K on uncertain and all-failure problems;
2. student-generated *successful* alternatives as chosen-side candidates, not
   only teacher continuations;
3. teacher coverage for the all-failure problems that produced no recovery
   pair;
4. a recorded structural limitation: at K = 4 the weight `w = p(1-p)` takes
   only three distinct values, including zero.

None of these changes the S69 execution; they are recorded so the next corpus
is not built with the same gaps.

## 6. What does not change

* the objective functional form and `beta = 0.1`;
* `lr = 1e-4`, `--grad-clip 1.0`, rank 64 / alpha 32, base frozen;
* the 2,000,000 forward-token-position budget and the 500k / 1M / 2M cadence;
* `--batch-pairs 1`, `--seed 6900`, `--pair-order shuffled`,
  `--weight-normalization corpus`;
* the identical-effective-problem-set rule and the `w <= 0` drop;
* the recovery arm's matched mean positive weight;
* the calibration-only selection rule, the MMLU-cal retention filter and the
  adoption gate;
* the win bar (both MATH scales' paired intervals above zero; GSM8K and MMLU
  not worse than -1.0);
* the one-shot test rule and the no-enlargement clause;
* the frozen reference identity (the S67 soup champion).

## 7. Provenance

No calibration or test score existed when this amendment was registered. The
only S69 training receipts that existed were the two superseded preference-arm
runs and the stopped matched-SFT invocation, none of which is a score and none
of which is used for selection. The calibration splits named by the
registration have not been read. The test splits have not been read.
