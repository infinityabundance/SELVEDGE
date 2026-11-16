# SIMULATOR — trace-driven execution, deadlines and placement (M3, part 2)

Status: **implemented, calibrated against the measured store, validated on four
held-out arms, and explicitly separated from measurement.** This document
describes `selvedge-sim`, its trace format, its event model, its calibration and
its validation. Every number in it is emitted from a receipt by
`scripts/perf_numbers.py` in §10; the prose states the method, not the values.

## 1. What is simulated, and what is measured

| quantity | status | evidence |
| --- | --- | --- |
| cold per-shard transfer time | **measured** | `results/store/metrics-06b-r1/plan-order.jsonl`, `results/store/metrics-17b-r1/plan-order.jsonl` |
| SHA-256 verification time | **measured** | same files |
| queue wait, cache insertion, eviction, host-RAM hit copy | **measured** | same files; the hit path uses the `lru-1of4` arm |
| read latency as a function of request size | **measured** at four sizes | `results/io/io-scan-{06b,17b}-r1.json` |
| raw device throughput at queue depths 1/4/16 | **measured** | `results/io/io-scan-*-r1.json` |
| cache policy decisions (LRU/LFU/least-stale/pinned) | **reproduced exactly** | the replay comparison in §5 |
| per-record stall under a policy and capacity | **simulated** | §5 |
| device bytes and waste under a policy | **simulated**, and exact in the replay | §5 |
| GPU staging / PCIe transfer | **not modelled** | no measurement exists; the tier is declared only |
| compute time per token-layer | **assumed input** (`--compute-ns`) | never a measured quantity |
| a trained router's selection trace | **not available** | §9 |

The simulator is not a benchmark. A simulation-only improvement is not a result
in this program; the reports label every simulated quantity and the validation
receipts price the distance to the real engine.

## 2. Trace format

`selvedge/execution-trace/v1` is JSONL plus a sidecar manifest
(`selvedge/execution-trace-manifest/v1`). The manifest carries the shard table
(id, bytes, payload offset), the record count, the SHA-256 over
`record_line_bytes || "\n"` for every record, the source
(`measured`/`reconstructed`/`synthetic`), the provenance (origin, receipt path
and digest, revision, stamp, notes), the bundle identity and the selection mode.

The one invariant that makes the simulator honest:

> **A trace may describe what was selected, never what the model should have
> selected.**

There is no field for an expected set, a target, a gold answer or a future
token. Every record struct uses `deny_unknown_fields`, so a trace that carries
one fails to deserialize; this is a unit test (`trace_rejects_semantic_fields`).
The selected set is the only thing the simulator executes.

**Exact versus approximate.** A record's `mode` is `exact` or `approximate` and
must match the manifest. `exact` is a claim about the selection: the recorded set
is the model's native computation set for that step. The format cannot prove it,
so an exact trace must declare a basis, and the only basis this build accepts is
`whole_bundle_plan_read`: every record selects the complete validated shard
table, which is what the exact-mode storage pass executes. A selector that
predicts a subset therefore cannot be exported as exact.

Trace sources in this session:

- **measured** — the whole-bundle storage pass, one record per pass over the real
  0.6B bundle (1344 shards), aggregated from the per-request JSONL; each record
  carries the measured stall, device bytes and hit count as observations;
- **reconstructed** — the M2 fixture's per-context selector decisions
  (`magnitude`), projected onto the real 1.7B bundle's own shard ids through its
  channel ranges. The fixture carries no request identity, so each record is one
  independently sampled context, sorted by (position, layer); it is not a causal
  token stream, and the trace says so in its notes;
- **synthetic** — in-crate generation with hot/cold skew, phase switching and
  chunk unions, labelled `synthetic` with its generator parameters in the
  manifest.

## 3. Event model

Time is nanoseconds. The consumer walks records in file order. For each record
the needed shards are classified as resident (a host-RAM copy, charged), already
in flight as a prefetch (the consumer waits for completion), or missing.

- **Tiers.** NVMe → host RAM is modelled; GPU-ready staging is a declared
  capacity, not a device. Host RAM is a hard byte budget; an insertion that
  cannot fit is refused by the policy, exactly as in the measured engine.
- **Coalescing.** Missing shards are grouped into runs over their physical
  order. Two extents merge when the gap between them is at most
  `--coalesce-gap-bytes` and the run stays within `--max-run-bytes`; bridged gap
  bytes are counted as `bytes_overread`, never as useful bytes. A run costs one
  transfer plus one verification, so merging amortises the per-request overhead;
  the size law supplies the sublinear component from measured data.
- **Transfer and verification.** Transfer duration is sampled from the measured
  empirical distribution, scaled by the fitted log-log size law when the run size
  differs from the calibrated shard size. Verification is linear in bytes
  (hashing is) and is charged on every cold and speculative byte unless
  `--verify false`.
- **Queue depth and backpressure.** `--queue-depth` servers run concurrently; a
  run starts on the earliest-free server, ties broken by the lowest server index.
  The in-flight byte budget is a hard limit: a required transfer that would
  exceed it waits (`wait`) or fails the run closed (`fail`). Waits are counted.
- **Eviction cost.** Insertion and eviction costs are sampled per insertion and
  per eviction, on the same path as the store.
- **Cancellation.** A speculative transfer can be cancelled before completion
  (preempted by a required transfer, dropped at a phase change, or drained at the
  end of the trace). Cancelled bytes are reported separately and are never
  counted as read.
- **Stall accounting.** A record's exposed stall is
  `max(run completion, awaited prefetch completion) - need_time`, plus host-copy
  time for resident shards. A record whose data is ready contributes zero. The
  consumer's next `need_time` is `need_time + stall + compute_ns`, so compute is
  the only window in which a prefetch can overlap for free.
- **Deadlines.** `--deadline-ns` is a per-record latency budget; a larger stall
  is a counted deadline miss.
- **Determinism.** One `SplitMix64` stream seeded by `--seed`; records in file
  order; shards in physical-placement order; server ties by index; cache victims
  by the policy key then shard id; prefetch candidates sorted by shard id;
  preemption cancels the latest-finishing speculative transfer, ties by lowest
  transfer id. Two runs with the same seed and configuration are byte-identical
  (`determinism_under_a_fixed_seed`).

`SimSummary` defines the byte accounting once and checks it on every run:

```text
bytes_needed      = bytes_hit + bytes_required_miss
bytes_device_read = bytes_required_miss + bytes_phase_cold + bytes_overread + bytes_prefetch
bytes_prefetch    = bytes_prefetch_used + bytes_prefetch_wasted
bytes_wasted      = bytes_overread + bytes_prefetch_wasted
```

Phase warm-up has its own term because those bytes are neither needed by the
record that triggered them nor wasted. The conservation identities are checked
by `SimSummary::check` on every report and are property-tested.

## 4. Policies

Cache policies (`none`, `lru`, `lfu`, `least_stale`, `pinned`) reproduce the
measured engine's semantics and sequence-number protocol, including victim
tie-breaks, so a simulated hit/miss sequence can be compared with a recorded
one. The replay comparison in §5 confirms exact agreement on every read of four
arms.

Prefetch predictors are advisory and behind a trait: `none`, `last-value`
(causal, re-proposes the current set), `envelope` (the active envelope's
candidate list), and `oracle-next` (reads the next record; **non-deployable**,
labelled as such in every report because `uses_future()` is true). A predictor
returns candidates; the engine decides what to fetch, and a wrong prediction is
a stall or a wasted byte. It can never add a shard to a record's selected set:
`prefetch_never_changes_the_selected_set` compares the `bytes_needed` sequence
across predictors and policies.

## 5. Validation against the measured engine (Court F)

Two independent checks, both CPU-only.

**(a) Recorded-metrics replay.** The measured engine writes one JSON line per
read. The simulator replays the cache policy over that read order and compares:

1. every hit/miss decision with the engine's own record — an exact test of the
   policy implementation;
2. the stall distribution predicted by the calibrated model against the measured
   one — out of sample, because the calibration inputs and the target arm are
   different files.

Tolerance was fixed before the numbers: p50 within 25%, p95 within 35%, p99
within 50%, device bytes within 1%, and zero hit/miss mismatches. All four
held-out arms passed, with zero hit/miss mismatches and an exact device-byte
prediction; the discrepancies are in §10 and in
`results/sim/validation-*.json`.

**(b) Live store comparison.** The same trace is executed twice: through the real
`ShardStore::read_required` path (synchronous, one server, page cache dropped
before the cold pass) and through the simulator configured to match
(`--queue-depth 1`, no compute overlap). Both receive the identical executed set.
The receipt reports absolute p50/p95/p99 discrepancies, device bytes, and a
**shape** statistic (`p95/p50`, `p99/p50`) that is robust to a uniform
environment slowdown.

This session's live run is a genuine negative finding about the *measurement
environment*, not about the model:

- device bytes agreed exactly and hit counts agreed;
- the absolute latency discrepancy is very large, while the shape discrepancy is
  small;
- the same-window quick `io-scan` (`results/io/io-scan-06b-contended.json`)
  measured the device and CPU far slower than at calibration time, because the
  detached T2 training pilot was saturating the host.

So the absolute check failed and the shape check passed. That combination is the
signature of environment drift: the calibration describes an idle host, and this
comparison was not run on one. The receipt keeps the mechanical verdict `defect`
and records the environment note; the honest reading is **inconclusive for
absolute latency, corroborated for bytes and residency shape**. A re-run on an
idle host is the outstanding work (§11).

## 6. Calibration

`selvedge sim calibrate` reads the per-request JSONL and the `io-scan` receipt
and writes `results/sim/calibration-{06b,17b}.json`. It fits the distribution of
each measured component and records quantile residuals for an exponential and a
lognormal family.

The empirical quantile grid is the **model of record**. The parametric fits are
reported so a reader can see whether an exponential assumption would have been
wrong: on both the transfer and the verification samples the exponential's
largest relative quantile error is an order of magnitude worse than the
lognormal's. The simulator therefore never assumes exponential latency. A
parametric fit also inherits the tails of the sample it was fitted to, which is
the second reason the empirical grid is primary.

The one extrapolation is the size law: read latency at a request size that was
not measured is interpolated log-log between the measured points and
extrapolated with the nearest segment's slope. No advertised bandwidth is used
anywhere in the model.

## 7. Pareto and placement

**Pareto.** Quality is fixed in exact mode: every arm executes the trace's
selected sets, so the table trades bytes and tail latency only. On the
reconstructed 1.7B M2 trace the no-cache arm reads the full per-record selection
from the device; LRU at half the unique payload roughly halves the bytes and the
p50 stall, and at full residency LRU and LFU coincide because both fit. The p95
improves much less than the p50: the tail is set by excursions the cache cannot
hold, not by the mean. The full table is `results/sim/pareto-m2-17b.json` and the
plot is `results/sim/pareto-m2-17b.svg`.

**Placement.** The mandate's problem is to minimise expected stall plus bytes
plus eviction cost over the physical ordering. `selvedge sim placement` builds
four candidate orders — sequential, frequency-sorted, co-access-blocked, and
co-access blocks plus replicated hot fragments — and the **simulator** decides
between them; a surrogate only proposes them.

The surrogate edge weight is not raw co-occurrence:

```text
w(a,b) = lift(a,b) * marginal_ns(a,b)
lift   = P(a,b) / (P(a) P(b))
margin = L(ba) + L(bb) - L(ba + bb)     from the calibrated size law
```

`lift` says how much more often a pair co-occurs than independence predicts;
`marginal_ns` says how much a coalesced transfer would save. Uncertainty is
reported: a seeded record-level bootstrap gives a percentile interval for the
lift of the strongest edges, and the receipt states how many of the checked edges
are stable (interval above 1). The weighting and its caveats are in the
receipt's `weights` object.

On the synthetic alternating trace (448 shards, two phases, chunk unions) the
co-access-blocked layout has the lowest p95, replicated hot fragments are within
a fraction of a percent of it at a small storage amplification, and
frequency-sorted is the slowest — sorting by global frequency destroys the
adjacency the original order already had. The gain is deliberately reported as
small, because a record's selected groups are mostly already adjacent. On the reconstructed M2 trace all four layouts are within a few
percent and the surrogate objective is zero, because that sample's selected
shards are not physically adjacent; the honest reading is that placement has
nothing to gain on that trace's structure. Replication is never free: the extra
bytes appear as amplification, and the surrogate's eviction penalty is an
explicit input, which is why a high penalty can prefer a non-replicated layout.

## 8. Envelope results

On the synthetic alternating trace, with an LRU cache at a quarter of the unique
payload and an assumed compute window (`--compute-ns 5000000`, an assumption, not
a measurement), the envelope transfers less, saves exposed p95 time, and pays a
phase warm-up. All four quantities — bytes not transferred, exposed time saved,
cold bytes charged and unused-prefetch bytes — are reported with their signs
rather than clipped, so a net loss would be visible. The envelope was invalidated
during evaluation, which is the intended behaviour when the phase's selections
leave the candidate superset.

## 9. Known limitation: the traces are not a trained-router workload

The M2 selector decisions come from a degraded, untrained-router setting; the T2
pilot's per-step records carry no routing field, so no trained-router trace
exists yet. The reconstructed M2 trace is therefore a *sample of contexts*, not a
token stream, and every locality conclusion in this document is **conditional on
a trained model's routing statistics**. That limitation is recorded in the trace
notes, in the receipts and in `docs/LIMITATIONS.md`. Synthetic traces exist
precisely so that the mechanisms (hot/cold skew, phase switching, chunk unions)
can be studied without pretending they are measurements.

## 10. Generated numbers

Every value below is emitted by `scripts/perf_numbers.py` from the named receipt.

Calibration, 0.6B bundle, 768 KiB shards (`results/sim/calibration-06b.json`):

- cold transfer p50 (ns):
  <!-- BEGIN GENERATED: sim-cal-transfer-p50 -->
930308
  <!-- END GENERATED: sim-cal-transfer-p50 -->
- cold transfer p95 (ns):
  <!-- BEGIN GENERATED: sim-cal-transfer-p95 -->
1254064
  <!-- END GENERATED: sim-cal-transfer-p95 -->
- verification p50 (ns):
  <!-- BEGIN GENERATED: sim-cal-verify-p50 -->
307905
  <!-- END GENERATED: sim-cal-verify-p50 -->
- host-RAM hit copy p50 (ns):
  <!-- BEGIN GENERATED: sim-cal-hit-copy-p50 -->
23164
  <!-- END GENERATED: sim-cal-hit-copy-p50 -->
- exponential transfer fit, largest relative quantile error:
  <!-- BEGIN GENERATED: sim-cal-transfer-exp-maxerr -->
1.6581
  <!-- END GENERATED: sim-cal-transfer-exp-maxerr -->
- lognormal transfer fit, largest relative quantile error:
  <!-- BEGIN GENERATED: sim-cal-transfer-lognormal-maxerr -->
0.2658
  <!-- END GENERATED: sim-cal-transfer-lognormal-maxerr -->
- exponential verification fit, largest relative quantile error:
  <!-- BEGIN GENERATED: sim-cal-verify-exp-maxerr -->
2.9533
  <!-- END GENERATED: sim-cal-verify-exp-maxerr -->
- lognormal verification fit, largest relative quantile error:
  <!-- BEGIN GENERATED: sim-cal-verify-lognormal-maxerr -->
0.0739
  <!-- END GENERATED: sim-cal-verify-lognormal-maxerr -->

Held-out replay validation, percentage discrepancy (predicted minus measured):

- 0.6B `lru-1of2` p50 / p95 / p99:
  <!-- BEGIN GENERATED: sim-val-06b-lru1of2-p50 -->
-7.88%
  <!-- END GENERATED: sim-val-06b-lru1of2-p50 -->
  <!-- BEGIN GENERATED: sim-val-06b-lru1of2-p95 -->
-7.44%
  <!-- END GENERATED: sim-val-06b-lru1of2-p95 -->
  <!-- BEGIN GENERATED: sim-val-06b-lru1of2-p99 -->
-15.24%
  <!-- END GENERATED: sim-val-06b-lru1of2-p99 -->
- 0.6B `lru-1of2` device bytes and hit/miss mismatches:
  <!-- BEGIN GENERATED: sim-val-06b-lru1of2-bytes -->
+0.00%
  <!-- END GENERATED: sim-val-06b-lru1of2-bytes -->
  <!-- BEGIN GENERATED: sim-val-06b-lru1of2-mismatches -->
0
  <!-- END GENERATED: sim-val-06b-lru1of2-mismatches -->
- 0.6B `least-stale-1of4` p50 / p95 / p99:
  <!-- BEGIN GENERATED: sim-val-06b-leaststale-p50 -->
-2.20%
  <!-- END GENERATED: sim-val-06b-leaststale-p50 -->
  <!-- BEGIN GENERATED: sim-val-06b-leaststale-p95 -->
-0.74%
  <!-- END GENERATED: sim-val-06b-leaststale-p95 -->
  <!-- BEGIN GENERATED: sim-val-06b-leaststale-p99 -->
+1.38%
  <!-- END GENERATED: sim-val-06b-leaststale-p99 -->
- 0.6B `none-cold` p50 / p95 / p99:
  <!-- BEGIN GENERATED: sim-val-06b-none-p50 -->
-0.78%
  <!-- END GENERATED: sim-val-06b-none-p50 -->
  <!-- BEGIN GENERATED: sim-val-06b-none-p95 -->
-6.37%
  <!-- END GENERATED: sim-val-06b-none-p95 -->
  <!-- BEGIN GENERATED: sim-val-06b-none-p99 -->
-9.88%
  <!-- END GENERATED: sim-val-06b-none-p99 -->
- 1.7B `lru-1of2` p50 / p95 / p99:
  <!-- BEGIN GENERATED: sim-val-17b-lru1of2-p50 -->
-3.27%
  <!-- END GENERATED: sim-val-17b-lru1of2-p50 -->
  <!-- BEGIN GENERATED: sim-val-17b-lru1of2-p95 -->
-2.45%
  <!-- END GENERATED: sim-val-17b-lru1of2-p95 -->
  <!-- BEGIN GENERATED: sim-val-17b-lru1of2-p99 -->
+15.22%
  <!-- END GENERATED: sim-val-17b-lru1of2-p99 -->

Live store comparison (`results/sim/compare-store-06b.json`):

- cold absolute p50 discrepancy (percent):
  <!-- BEGIN GENERATED: sim-compare-cold-abs-p50 -->
-90.19%
  <!-- END GENERATED: sim-compare-cold-abs-p50 -->
- cold `p95/p50` shape discrepancy (percent):
  <!-- BEGIN GENERATED: sim-compare-cold-shape-p95 -->
-0.61%
  <!-- END GENERATED: sim-compare-cold-shape-p95 -->
- cold device-byte discrepancy (percent):
  <!-- BEGIN GENERATED: sim-compare-cold-bytes -->
+0.00%
  <!-- END GENERATED: sim-compare-cold-bytes -->
- warm `p95/p50` shape discrepancy (percent):
  <!-- BEGIN GENERATED: sim-compare-warm-shape-p95 -->
+3.91%
  <!-- END GENERATED: sim-compare-warm-shape-p95 -->

Pareto, reconstructed 1.7B M2 trace (`results/sim/pareto-m2-17b.json`):

- no-cache device bytes per record:
  <!-- BEGIN GENERATED: sim-pareto-none-bytes -->
47185920 B
  <!-- END GENERATED: sim-pareto-none-bytes -->
- LRU at half the unique payload, device bytes per record:
  <!-- BEGIN GENERATED: sim-pareto-lru05-bytes -->
23359488 B
  <!-- END GENERATED: sim-pareto-lru05-bytes -->
- LRU at half the unique payload, p50 stall (ns):
  <!-- BEGIN GENERATED: sim-pareto-lru05-p50 -->
30269233
  <!-- END GENERATED: sim-pareto-lru05-p50 -->
- LRU at half the unique payload, hit rate:
  <!-- BEGIN GENERATED: sim-pareto-lru05-hitrate -->
0.5049
  <!-- END GENERATED: sim-pareto-lru05-hitrate -->
- LRU at full residency, p50 stall (ns):
  <!-- BEGIN GENERATED: sim-pareto-lru10-p50 -->
14451972
  <!-- END GENERATED: sim-pareto-lru10-p50 -->
- LRU at full residency, p95 stall (ns):
  <!-- BEGIN GENERATED: sim-pareto-lru10-p95 -->
67035390
  <!-- END GENERATED: sim-pareto-lru10-p95 -->

Placement, synthetic alternating trace
(`results/sim/placement-synthetic-alternating.json`):

- sequential layout p95 stall (ns):
  <!-- BEGIN GENERATED: sim-placement-seq-p95 -->
4111333
  <!-- END GENERATED: sim-placement-seq-p95 -->
- co-access-blocked layout p95 stall (ns):
  <!-- BEGIN GENERATED: sim-placement-coaccess-p95 -->
4062435
  <!-- END GENERATED: sim-placement-coaccess-p95 -->
- replicated-hot layout p95 stall (ns):
  <!-- BEGIN GENERATED: sim-placement-replicated-p95 -->
4052190
  <!-- END GENERATED: sim-placement-replicated-p95 -->
- replicated-hot storage amplification:
  <!-- BEGIN GENERATED: sim-placement-replicated-amp -->
1.0179
  <!-- END GENERATED: sim-placement-replicated-amp -->

Envelopes, synthetic alternating trace
(`results/sim/envelope-utility-synthetic-alternating.json`):

- bytes not transferred:
  <!-- BEGIN GENERATED: sim-envelope-bytes-saved -->
495452160
  <!-- END GENERATED: sim-envelope-bytes-saved -->
- exposed p95 time saved (ns):
  <!-- BEGIN GENERATED: sim-envelope-time-saved -->
312796
  <!-- END GENERATED: sim-envelope-time-saved -->
- cold bytes charged to phase warm-up:
  <!-- BEGIN GENERATED: sim-envelope-cold-charged -->
475004928
  <!-- END GENERATED: sim-envelope-cold-charged -->
- unused-prefetch bytes:
  <!-- BEGIN GENERATED: sim-envelope-unused -->
0
  <!-- END GENERATED: sim-envelope-unused -->
- invalidations:
  <!-- BEGIN GENERATED: sim-envelope-invalidations -->
4
  <!-- END GENERATED: sim-envelope-invalidations -->

Single simulation, reconstructed 1.7B M2 trace, LRU at a quarter of the payload
(`results/sim/simulate-m2-17b-lru025.json`):

- p95 stall (ns):
  <!-- BEGIN GENERATED: sim-simulate-m2-p95 -->
72857039
  <!-- END GENERATED: sim-simulate-m2-p95 -->
- bytes served from host RAM:
  <!-- BEGIN GENERATED: sim-simulate-m2-hit-bytes -->
3321888768
  <!-- END GENERATED: sim-simulate-m2-hit-bytes -->

## 11. Assumptions and threats to validity

- **Empirical grid, finite sample.** The transfer distribution is the measured
  sample; its tails stop where the sample stops. A p99 beyond the sample's
  largest observation is an interpolation between the last two grid points.
- **Size extrapolation.** Reads larger or smaller than the measured sizes use the
  log-log size law. This is the only fabricated curve in the model, and it is
  labelled.
- **Serialised calibration.** The measured engine issues one synchronous read at
  a time, so the calibration describes a serialised path. Simulated queue depth
  above one is a *what-if*, not a measured configuration.
- **Device bimodality.** The same arm measured 2–3× differently across repeats on
  this workstation; the calibration uses repeat 1 and the held-out validations
  use different arms from the same repeat set, which is a real but limited
  transfer test.
- **Host contention.** The live comparison ran while the T2 pilot saturated the
  host; §5 states what that does and does not establish.
- **No GPU tier.** GPU staging, PCIe and pinned-buffer effects are absent. Any
  GPU-latency claim requires the missing measurement.
- **Compute window.** `--compute-ns` is supplied by the caller. Envelope overlap
  and prefetch benefit scale with it; it is never measured in this session.
- **Envelope build coverage.** Coverage is a property of the build prefix, not a
  prediction score for the evaluation suffix.
- **Placement surrogate.** `lift × marginal` is a local estimate of saved stall.
  The bootstrap interval covers only the strongest edges, and the coalescing
  saving comes from the size law rather than from a direct pair measurement.

## 12. What remains unvalidated

- Absolute latency of the simulator against the store **on an idle host**.
- Any GPU or PCIe transfer number.
- A trained-router execution trace; P8.3/P8.4's cold-bytes-versus-capacity curve
  is therefore a mechanism study, not a model claim.
- Deadline-aware prefetch under real scheduler pressure (the deadline field is
  implemented and tested, but no workload has a measured deadline).
- Whether the placement surrogate ranks layouts correctly beyond the four
  candidates tested here.

## 13. Commands

```sh
# calibrate from measured receipts (CPU only)
selvedge sim calibrate --metrics transfer+verify=metrics/plan-order.jsonl \
    --metrics hit_copy=metrics/lru-1of4.jsonl --io results/io/io-scan-06b-r1.json \
    --out results/sim/calibration-06b.json

# held-out replay validation
selvedge sim validate --calibration results/sim/calibration-06b.json \
    --metrics metrics/lru-1of2.jsonl --arm lru-1of2 \
    --bench results/store/store-bench-06b-r1.json --policy lru \
    --out results/sim/validation-06b-lru-1of2.json

# traces
selvedge trace export --bundle <bundle> --whole-bundle --out t.jsonl
selvedge trace export --bundle <bundle> --metrics metrics/plan-order.jsonl --out t.jsonl
selvedge trace export --bundle <1.7B bundle> --fixtures selector-inputs.json \
    --selector magnitude --out m2.jsonl
selvedge trace generate --preset alternating --out synthetic.jsonl
selvedge trace inspect --trace t.jsonl

# simulation, envelopes, comparisons
selvedge simulate --trace t.jsonl --calibration c.json --policy lru \
    --cache-fraction 0.25 --out report.json --json
selvedge envelope build --trace t.jsonl --out env.json
selvedge envelope apply --trace t.jsonl --envelopes env.json --calibration c.json
selvedge sim compare-store --trace t.jsonl --bundle <bundle> --calibration c.json
selvedge sim pareto --trace t.jsonl --calibration c.json --policies none,lru,lfu
selvedge sim placement --trace t.jsonl --calibration c.json --cache-fraction 0.25
```

All of the receipts above are regenerated by
`scripts/court rust -- sh scripts/sim-receipts.sh`, and the plots by
`scripts/court python -- scripts/sim_plots.py ...`.
