//! Engine, trace-replay and property tests for the simulator.
//!
//! The fixture calibration is **constant**: every component is a fixed value, so
//! the expected schedule can be computed by hand and compared exactly. The
//! measured calibration is exercised by the validation receipts, not here.

#![allow(clippy::unwrap_used)]

use selvedge_core::{SelectionMode, Sha256Digest, ShardId};
use selvedge_sim::calibration::{CALIBRATION_SCHEMA, Calibration, SampleStats, SizeLaw, SizePoint};
use selvedge_sim::engine::{BackpressureMode, SimConfig, TwoTimescaleConfig, WorkingSetSource};
use selvedge_sim::envelope::{self, EnvelopeParams};
use selvedge_sim::error::SimError;
use selvedge_sim::policy::{LastValuePrefetch, NoPrefetch, OracleNextRecordPrefetch, Predictor};
use selvedge_sim::trace::{
    TRACE_MANIFEST_SCHEMA, TRACE_SCHEMA, Trace, TraceBundle, TraceManifest, TraceProvenance,
    TraceRecord, TraceShard, TraceSource,
};
use selvedge_sim::{measure_envelope, simulate};
use selvedge_store::PolicyKind;

const SHARD_BYTES: u64 = 1000;

fn id(tag: u8) -> ShardId {
    ShardId::from_digest(Sha256Digest::of(&[tag]))
}

fn stats(value: u64) -> SampleStats {
    SampleStats::of(&vec![value; 16]).unwrap()
}

fn calibration() -> Calibration {
    Calibration {
        schema: CALIBRATION_SCHEMA.to_owned(),
        created_utc: "20261116T000000Z".to_owned(),
        revision: "test".to_owned(),
        inputs: Vec::new(),
        shard_bytes: SHARD_BYTES,
        transfer: stats(1000),
        verify: stats(500),
        queue_wait: stats(0),
        insert: stats(100),
        eviction: stats(10),
        hit_copy: stats(50),
        size_law: SizeLaw::fit(
            vec![
                SizePoint {
                    bytes: 1000,
                    p50_ns: 1000,
                },
                SizePoint {
                    bytes: 2000,
                    p50_ns: 1500,
                },
                SizePoint {
                    bytes: 4000,
                    p50_ns: 2000,
                },
            ],
            "fixture",
        )
        .unwrap(),
        fits: Vec::new(),
        empirical_is_model_of_record: true,
        notes: Vec::new(),
    }
}

/// Build a trace with the given shard offsets and record selections.
fn trace_with(offsets: &[(u8, u64)], records: &[(u64, &str, Vec<u8>)]) -> Trace {
    let shards: Vec<TraceShard> = offsets
        .iter()
        .map(|(tag, offset)| TraceShard {
            id: id(*tag),
            bytes: SHARD_BYTES,
            offset: *offset,
        })
        .collect();
    let built: Vec<TraceRecord> = records
        .iter()
        .map(|(token, phase, tags)| {
            let mut selected: Vec<ShardId> = tags.iter().map(|tag| id(*tag)).collect();
            selected.sort_unstable();
            selected.dedup();
            TraceRecord {
                schema: TRACE_SCHEMA.to_owned(),
                request_id: "r0".to_owned(),
                phase_id: (*phase).to_owned(),
                token_index: *token,
                layer: 0,
                selected,
                selector_id: "fixture".to_owned(),
                mode: SelectionMode::Approximate,
                task: Some("code".to_owned()),
                timestamps: None,
                observed: None,
            }
        })
        .collect();
    let manifest = TraceManifest {
        schema: TRACE_MANIFEST_SCHEMA.to_owned(),
        trace_id: "fixture".to_owned(),
        source: TraceSource::Synthetic,
        provenance: TraceProvenance {
            origin: "engine-test".to_owned(),
            receipt: None,
            receipt_sha256: None,
            revision: "test".to_owned(),
            stamp: "20261116T000000Z".to_owned(),
            notes: Vec::new(),
        },
        bundle: TraceBundle {
            bundle_id: "fixture".to_owned(),
            manifest_sha256: Sha256Digest::of(b"fixture"),
        },
        mode: SelectionMode::Approximate,
        exact_basis: None,
        shards,
        records: 0,
        records_sha256: Sha256Digest::of(b""),
        generator: None,
        created_utc: "20261116T000000Z".to_owned(),
    };
    Trace::new(manifest, built).unwrap()
}

/// Four shards packed back-to-back.
fn contiguous(records: &[(u64, &str, Vec<u8>)]) -> Trace {
    trace_with(
        &[
            (1, 0),
            (2, SHARD_BYTES),
            (3, 2 * SHARD_BYTES),
            (4, 3 * SHARD_BYTES),
        ],
        records,
    )
}

fn base_config(policy: PolicyKind, cache_bytes: u64) -> SimConfig {
    SimConfig {
        policy,
        cache_bytes,
        queue_depth: 1,
        max_in_flight_bytes: 64 * 1024 * 1024,
        coalesce_gap_bytes: 0,
        max_run_bytes: 1024 * 1024,
        verify: true,
        compute_ns_per_record: 0,
        deadline_ns: None,
        seed: 7,
        backpressure: BackpressureMode::Wait,
        placement: None,
        two_timescale: None,
        cancel_prefetch_on_phase_change: true,
        preempt_prefetch_for_required: true,
        start_record: 0,
        max_prefetch_shards: 64,
        inject_checksum_failure: None,
    }
}

#[test]
fn replay_reproduces_hand_computed_schedule() {
    // One shard, three records, LRU cache of exactly one shard.
    //   record 0: miss  1000 transfer + 500 verify + 100 insert = 1600
    //   record 1: hit    50 copy
    //   record 2: hit    50 copy
    let trace = contiguous(&[(0, "p0", vec![1]), (1, "p0", vec![1]), (2, "p0", vec![1])]);
    let calibration = calibration();
    let config = base_config(PolicyKind::Lru, SHARD_BYTES);
    let mut predictor = NoPrefetch;
    let simulation = simulate(&trace, &calibration, config, &mut predictor, None).unwrap();
    let stalls: Vec<u64> = simulation.outcomes.iter().map(|o| o.stall_ns).collect();
    assert_eq!(stalls, vec![1600, 50, 50]);
    assert_eq!(simulation.outcomes[0].start_ns, 0);
    assert_eq!(simulation.outcomes[0].end_ns, 1600);
    assert_eq!(simulation.outcomes[1].start_ns, 1600);
    assert_eq!(simulation.outcomes[2].end_ns, 1700);
    assert_eq!(simulation.summary.bytes_needed, 3000);
    assert_eq!(simulation.summary.bytes_hit, 2000);
    assert_eq!(simulation.summary.bytes_required_miss, 1000);
    assert_eq!(simulation.summary.bytes_device_read, 1000);
    simulation.summary.check().unwrap();
}

#[test]
fn capacity_boundary_and_eviction_ordering() {
    // Two shards, capacity for one, LRU. A then B then A: the second A misses.
    let trace = contiguous(&[(0, "p0", vec![1]), (1, "p0", vec![2]), (2, "p0", vec![1])]);
    let calibration = calibration();
    let config = base_config(PolicyKind::Lru, SHARD_BYTES);
    let mut predictor = NoPrefetch;
    let simulation = simulate(&trace, &calibration, config, &mut predictor, None).unwrap();
    assert_eq!(simulation.summary.misses, 3);
    assert_eq!(simulation.summary.hits, 0);
    assert_eq!(simulation.summary.evictions, 2);
    // Capacity for two shards: the third record is a hit.
    let config = base_config(PolicyKind::Lru, 2 * SHARD_BYTES);
    let simulation = simulate(&trace, &calibration, config, &mut predictor, None).unwrap();
    assert_eq!(simulation.summary.hits, 1);
    assert_eq!(simulation.outcomes[2].stall_ns, 50);
}

#[test]
fn coalescing_merges_adjacent_extents() {
    // One record needing two adjacent shards, one device server, no cache.
    //   separate (run cap = one shard): 2 * (1000 + 500) = 3000
    //   coalesced: 1500 (size law at 2000 bytes) + 1000 (verify, linear) = 2500
    let trace = contiguous(&[(0, "p0", vec![1, 2])]);
    let calibration = calibration();
    let mut config = base_config(PolicyKind::None, 0);
    config.max_run_bytes = SHARD_BYTES;
    let mut predictor = NoPrefetch;
    let split = simulate(&trace, &calibration, config.clone(), &mut predictor, None).unwrap();
    assert_eq!(split.outcomes[0].stall_ns, 3000);
    assert_eq!(split.summary.bytes_overread, 0);
    config.max_run_bytes = 1024 * 1024;
    let joined = simulate(&trace, &calibration, config, &mut predictor, None).unwrap();
    assert_eq!(joined.outcomes[0].stall_ns, 2500);
    assert_eq!(joined.summary.bytes_device_read, 2000);
    assert_eq!(joined.summary.bytes_overread, 0);
}

#[test]
fn coalescing_never_bridges_beyond_gap() {
    // Shard 1 at 0, shard 2 at 1500: a 500-byte hole.
    let trace = trace_with(&[(1, 0), (2, 1500)], &[(0, "p0", vec![1, 2])]);
    let calibration = calibration();
    let mut config = base_config(PolicyKind::None, 0);
    config.coalesce_gap_bytes = 499;
    let mut predictor = NoPrefetch;
    let narrow = simulate(&trace, &calibration, config, &mut predictor, None).unwrap();
    assert_eq!(narrow.summary.bytes_overread, 0);
    // The two runs are separate, so the transfer cost is paid twice.
    assert!(narrow.outcomes[0].stall_ns > 2600);
    let mut wide = base_config(PolicyKind::None, 0);
    wide.coalesce_gap_bytes = 500;
    let joined = simulate(&trace, &calibration, wide, &mut predictor, None).unwrap();
    assert_eq!(joined.summary.bytes_overread, 500);
    assert_eq!(joined.summary.bytes_device_read, 2500);
    joined.summary.check().unwrap();
}

#[test]
fn checksum_failure_fails_closed() {
    let trace = contiguous(&[(0, "p0", vec![1])]);
    let calibration = calibration();
    let mut config = base_config(PolicyKind::None, 0);
    config.inject_checksum_failure = Some(id(1));
    let mut predictor = NoPrefetch;
    let error = simulate(&trace, &calibration, config, &mut predictor, None).unwrap_err();
    assert!(error.to_string().contains("verification failure"));
}

#[test]
fn backpressure_wait_and_fail_modes() {
    // Two shards in one record with room for one in flight.
    let trace = contiguous(&[(0, "p0", vec![1, 2])]);
    let calibration = calibration();
    let mut config = base_config(PolicyKind::None, 0);
    config.max_in_flight_bytes = SHARD_BYTES;
    config.max_run_bytes = SHARD_BYTES;
    let mut predictor = NoPrefetch;
    let waited = simulate(&trace, &calibration, config.clone(), &mut predictor, None).unwrap();
    assert!(waited.summary.backpressure_waits >= 1);
    assert!(waited.outcomes[0].stall_ns >= 3000);
    config.backpressure = BackpressureMode::Fail;
    let error = simulate(&trace, &calibration, config, &mut predictor, None).unwrap_err();
    assert!(error.to_string().contains("would wait"));
}

#[test]
fn oversized_transfer_is_rejected() {
    let trace = contiguous(&[(0, "p0", vec![1])]);
    let calibration = calibration();
    let mut config = base_config(PolicyKind::None, 0);
    config.max_in_flight_bytes = 10;
    let mut predictor = NoPrefetch;
    let error = simulate(&trace, &calibration, config, &mut predictor, None).unwrap_err();
    assert!(error.to_string().contains("can never fit"));
}

#[test]
fn cancellation_returns_the_server() {
    // A speculative transfer holds the only server; the next record needs two
    // real shards and must cancel it instead of waiting for it.
    let trace = contiguous(&[(0, "p0", vec![1]), (1, "p0", vec![2, 3])]);
    let calibration = calibration();
    let mut config = base_config(PolicyKind::Lru, 2 * SHARD_BYTES);
    config.max_run_bytes = SHARD_BYTES;
    let mut predictor = FixedPrefetch(vec![4]);
    let simulation = simulate(&trace, &calibration, config, &mut predictor, None).unwrap();
    assert_eq!(simulation.summary.bytes_prefetch_cancelled, SHARD_BYTES);
    // Without preemption the record would wait for the speculative transfer
    // first; with it, both cold reads start immediately.
    assert!(
        simulation.outcomes[1].stall_ns < 4000,
        "stall {}",
        simulation.outcomes[1].stall_ns
    );
    simulation.summary.check().unwrap();
}

/// A predictor that proposes a fixed set, so a prefetch can be observed
/// without a hypothesis about the next record.
#[derive(Debug)]
struct FixedPrefetch(Vec<u8>);

impl Predictor for FixedPrefetch {
    fn name(&self) -> &'static str {
        "fixed"
    }

    fn predict(&mut self, _context: &selvedge_sim::policy::PredictContext<'_>) -> Vec<ShardId> {
        self.0.iter().map(|tag| id(*tag)).collect()
    }
}

#[test]
fn prefetch_cancelled_at_the_end_is_not_counted_as_read() {
    // A prefetch that is never consumed and has not completed is cancelled.
    let trace = contiguous(&[(0, "p0", vec![1])]);
    let calibration = calibration();
    let config = base_config(PolicyKind::Lru, 2 * SHARD_BYTES);
    let mut predictor = FixedPrefetch(vec![4]);
    let simulation = simulate(&trace, &calibration, config, &mut predictor, None).unwrap();
    assert_eq!(simulation.summary.prefetch_accepted, 1);
    assert_eq!(simulation.summary.bytes_prefetch, 0);
    assert_eq!(simulation.summary.bytes_prefetch_wasted, 0);
    assert_eq!(simulation.summary.bytes_prefetch_cancelled, SHARD_BYTES);
    simulation.summary.check().unwrap();
}

#[test]
fn unused_prefetch_that_completed_is_waste() {
    let trace = contiguous(&[
        (0, "p0", vec![1]),
        (1, "p0", vec![1]),
        (2, "p0", vec![1]),
        (3, "p0", vec![1]),
    ]);
    let calibration = calibration();
    let mut config = base_config(PolicyKind::Lru, 2 * SHARD_BYTES);
    config.compute_ns_per_record = 5000;
    let mut predictor = FixedPrefetch(vec![4]);
    let simulation = simulate(&trace, &calibration, config, &mut predictor, None).unwrap();
    assert_eq!(simulation.summary.bytes_prefetch, SHARD_BYTES);
    assert_eq!(simulation.summary.bytes_prefetch_used, 0);
    assert_eq!(simulation.summary.bytes_prefetch_wasted, SHARD_BYTES);
    simulation.summary.check().unwrap();
}

#[test]
fn prefetch_saves_stall_when_it_completes_in_time() {
    // The oracle predictor loads record 1's shard during record 0's compute
    // window, so record 1 does not block on the device.
    let trace = contiguous(&[(0, "p0", vec![1]), (1, "p0", vec![2])]);
    let calibration = calibration();
    let mut config = base_config(PolicyKind::Lru, 2 * SHARD_BYTES);
    config.compute_ns_per_record = 5000;
    let mut predictor = OracleNextRecordPrefetch;
    let simulation = simulate(&trace, &calibration, config, &mut predictor, None).unwrap();
    assert_eq!(simulation.outcomes[0].stall_ns, 1600);
    assert_eq!(simulation.outcomes[1].stall_ns, 50);
    assert_eq!(simulation.summary.bytes_prefetch_used, SHARD_BYTES);
    simulation.summary.check().unwrap();
}

#[test]
fn prefetch_never_changes_the_selected_set() {
    let trace = contiguous(&[
        (0, "p0", vec![1, 2]),
        (1, "p0", vec![2, 3]),
        (2, "p0", vec![3, 4]),
    ]);
    let calibration = calibration();
    let config = base_config(PolicyKind::Lru, 3 * SHARD_BYTES);
    let mut none = NoPrefetch;
    let mut last = LastValuePrefetch;
    let mut oracle = OracleNextRecordPrefetch;
    let a = simulate(&trace, &calibration, config.clone(), &mut none, None).unwrap();
    let b = simulate(&trace, &calibration, config.clone(), &mut last, None).unwrap();
    let c = simulate(&trace, &calibration, config, &mut oracle, None).unwrap();
    let needs = |sim: &selvedge_sim::engine::Simulation| -> Vec<u64> {
        sim.outcomes
            .iter()
            .map(|outcome| outcome.bytes_needed)
            .collect()
    };
    assert_eq!(needs(&a), needs(&b));
    assert_eq!(needs(&a), needs(&c));
    assert!(oracle.uses_future());
    assert!(!last.uses_future());
}

#[test]
fn deadline_misses_are_counted() {
    let trace = contiguous(&[(0, "p0", vec![1]), (1, "p0", vec![1])]);
    let calibration = calibration();
    let mut config = base_config(PolicyKind::Lru, SHARD_BYTES);
    config.deadline_ns = Some(100);
    let mut predictor = NoPrefetch;
    let simulation = simulate(&trace, &calibration, config, &mut predictor, None).unwrap();
    assert_eq!(simulation.summary.deadline_misses, 1);
    assert!(simulation.outcomes[0].deadline_missed);
    assert!(!simulation.outcomes[1].deadline_missed);
}

#[test]
fn two_timescale_cold_change_is_charged_to_the_phase() {
    // Phase p0 uses shard 1; phase p1 uses shard 2. With a history working set,
    // p1's warm-up refetches shard 1 and that is charged as phase-cold.
    let trace = contiguous(&[
        (0, "p0", vec![1]),
        (1, "p0", vec![2]),
        (2, "p1", vec![3]),
        (3, "p1", vec![3]),
    ]);
    let calibration = calibration();
    let mut config = base_config(PolicyKind::Lru, SHARD_BYTES);
    config.two_timescale = Some(TwoTimescaleConfig {
        source: WorkingSetSource::History,
        capacity_fraction: 1.0,
        min_usage: 1,
    });
    let mut predictor = NoPrefetch;
    let simulation = simulate(&trace, &calibration, config, &mut predictor, None).unwrap();
    // The first record of p0 has no past, so it charges nothing.
    assert_eq!(simulation.phases[0].bytes_phase_cold, 0);
    // The first record of p1 charges the history working set.
    assert!(simulation.phases[1].bytes_phase_cold >= SHARD_BYTES);
    assert!(simulation.summary.bytes_phase_cold >= SHARD_BYTES);
    simulation.summary.check().unwrap();
}

#[test]
fn envelope_invalidation_on_excursion() {
    // Build envelopes from p0 records that use shard 1; evaluate p0 records
    // that use shard 2. The miss-rate rule must invalidate.
    let trace = contiguous(&[
        (0, "p0", vec![1]),
        (1, "p0", vec![1]),
        (2, "p0", vec![2]),
        (3, "p0", vec![2]),
    ]);
    let calibration = calibration();
    let params = EnvelopeParams {
        build_fraction: 0.5,
        min_coverage: 0.5,
        miss_rate_threshold: 0.5,
        miss_rate_min_records: 1,
        ..EnvelopeParams::default()
    };
    let set = envelope::build(&trace, params, "test", "stamp").unwrap();
    let mut config = base_config(PolicyKind::None, 0);
    config.start_record = set.build_records;
    config.two_timescale = Some(TwoTimescaleConfig {
        source: WorkingSetSource::Envelope,
        capacity_fraction: 1.0,
        min_usage: 1,
    });
    let mut predictor = NoPrefetch;
    let simulation = simulate(&trace, &calibration, config, &mut predictor, Some(&set)).unwrap();
    assert!(simulation.invalidations >= 1);
    assert!(simulation.summary.envelope_invalidations >= 1);
}

#[test]
fn envelope_utility_is_bytes_and_time() {
    // Alternating phases whose hot sets differ; the envelope is built from the
    // first half and evaluated on the second.
    let mut records = Vec::new();
    for token in 0..8_u64 {
        let phase = if token % 2 == 0 { "p0" } else { "p1" };
        let tag = if token % 2 == 0 { 1 } else { 2 };
        records.push((token, phase, vec![tag]));
    }
    let trace = contiguous(&records);
    let calibration = calibration();
    let params = EnvelopeParams {
        build_fraction: 0.5,
        min_coverage: 0.5,
        ..EnvelopeParams::default()
    };
    let set = envelope::build(&trace, params, "test", "stamp").unwrap();
    let base = base_config(PolicyKind::Lru, 2 * SHARD_BYTES);
    let (baseline, envelope_run, utility) =
        measure_envelope(&trace, &calibration, &base, &set).unwrap();
    assert_eq!(utility.baseline_bytes, baseline.summary.bytes_device_read);
    assert_eq!(
        utility.envelope_bytes,
        envelope_run.summary.bytes_device_read
    );
    assert_eq!(
        utility.bytes_not_transferred,
        utility.baseline_bytes as i64 - utility.envelope_bytes as i64
    );
    assert!(set.split_is_disjoint());
    // Both runs start at the evaluation boundary, never in the build prefix.
    assert_eq!(
        baseline.outcomes[0].index as usize,
        set.build_records as usize
    );
}

#[test]
fn determinism_under_a_fixed_seed() {
    let trace = contiguous(&[
        (0, "p0", vec![1, 2, 3]),
        (1, "p0", vec![2, 3]),
        (2, "p1", vec![3, 4]),
    ]);
    let calibration = calibration();
    let config = base_config(PolicyKind::Lru, 2 * SHARD_BYTES);
    let mut predictor = LastValuePrefetch;
    let a = simulate(&trace, &calibration, config.clone(), &mut predictor, None).unwrap();
    let b = simulate(&trace, &calibration, config, &mut predictor, None).unwrap();
    assert_eq!(a.summary, b.summary);
    assert_eq!(a.outcomes, b.outcomes);
}

#[test]
fn start_record_outside_the_trace_is_rejected() {
    let trace = contiguous(&[(0, "p0", vec![1])]);
    let calibration = calibration();
    let mut config = base_config(PolicyKind::None, 0);
    config.start_record = 5;
    let mut predictor = NoPrefetch;
    assert!(simulate(&trace, &calibration, config, &mut predictor, None).is_err());
}

#[test]
fn divergence_is_reported_as_a_percentage() {
    use selvedge_sim::compare::validate_metrics;
    let dir = selvedge_core::testutil::TempDir::new("sim-replay").unwrap();
    let metrics = dir.join("m.jsonl");
    let mut text = String::new();
    for (tag, transfer, verify, hit) in [
        (1_u8, 1000_u64, 500_u64, false),
        (2, 1000, 500, false),
        (1, 0, 0, true),
    ] {
        text.push_str(&metrics_line(tag, transfer, verify, hit));
        text.push('\n');
    }
    std::fs::write(&metrics, &text).unwrap();
    let calibration = calibration();
    let validation = validate_metrics(
        &calibration,
        &metrics,
        "handmade",
        PolicyKind::Lru,
        2 * SHARD_BYTES,
        7,
        "test",
        "stamp",
    )
    .unwrap();
    assert_eq!(validation.reads, 3);
    assert_eq!(validation.hit_mismatches, 0);
    assert_eq!(validation.hits_recorded, 1);
    assert_eq!(validation.hits_replayed, 1);
    assert_eq!(validation.bytes.measured_device_bytes, 2000);
    assert!(validation.stall_discrepancy_pct.contains_key("p95"));
    // The synthetic record's stall is 1600 and the model predicts
    // 1000 + 500 + 100 = 1600, so the p50 discrepancy is zero.
    assert!(validation.stall_discrepancy_pct["p50"].abs() < 1.0);
    // A deliberate mismatch is caught and the verdict is `defect`.
    let mismatched = dir.join("bad.jsonl");
    let mut text = String::new();
    text.push_str(&metrics_line(1, 1000, 500, true));
    text.push('\n');
    std::fs::write(&mismatched, text).unwrap();
    let bad = validate_metrics(
        &calibration,
        &mismatched,
        "bad",
        PolicyKind::Lru,
        SHARD_BYTES,
        7,
        "test",
        "stamp",
    )
    .unwrap();
    assert_eq!(bad.hit_mismatches, 1);
    assert_eq!(bad.verdict, "defect");
    assert!(bad.first_mismatch.is_some());
}

fn metrics_line(tag: u8, transfer: u64, verify: u64, hit: bool) -> String {
    let shard = format!(
        "sha256:{}",
        "ab".repeat(32).replacen("ab", &format!("{tag:02x}"), 1)
    );
    format!(
        "{{\"schema\":\"selvedge/store-metrics/v1\",\"event\":\"read\",\
         \"wall_unix_ns\":1,\"mono_ns\":1,\"shard\":\"{shard}\",\
         \"source\":\"{}\",\"bytes_requested\":{SHARD_BYTES},\
         \"bytes_completed\":{SHARD_BYTES},\"bytes_used\":{SHARD_BYTES},\
         \"bytes_wasted\":0,\"device_bytes\":{},\"queue_wait_ns\":0,\
         \"transfer_ns\":{transfer},\"verify_ns\":{verify},\"cache_insert_ns\":100,\
         \"eviction_ns\":0,\"stall_ns\":{},\"read_amplification\":1.0,\
         \"hit\":{hit},\"verified\":true,\"verification\":\"sha256_read\",\
         \"evicted_entries\":0}}",
        if hit { "host_ram" } else { "nvme" },
        if hit { 0 } else { SHARD_BYTES },
        if hit { 50 } else { transfer + verify + 100 },
    )
}

#[test]
fn byte_accounting_is_conserved_under_random_configurations() {
    use proptest::prelude::*;
    let mut runner = proptest::test_runner::TestRunner::deterministic();
    let strategy = (
        proptest::collection::vec(proptest::collection::vec(1_u8..5, 1..4), 1..8),
        prop_oneof![
            Just(PolicyKind::None),
            Just(PolicyKind::Lru),
            Just(PolicyKind::Lfu),
            Just(PolicyKind::LeastStale),
        ],
        1_u64..4,
        0_u64..3,
    );
    let calibration = calibration();
    runner
        .run(
            &strategy,
            |(records, policy, shard_span, capacity_shards)| {
                let tags: Vec<u8> = (1..=shard_span as u8).collect();
                let offsets: Vec<(u8, u64)> = tags
                    .iter()
                    .enumerate()
                    .map(|(index, tag)| (*tag, index as u64 * SHARD_BYTES))
                    .collect();
                // Clamp selections to the shard span.
                let records: Vec<(u64, &str, Vec<u8>)> = records
                    .iter()
                    .enumerate()
                    .map(|(index, selection)| {
                        (
                            index as u64,
                            if index % 3 == 0 { "p0" } else { "p1" },
                            selection
                                .iter()
                                .map(|tag| 1 + ((*tag - 1) % shard_span as u8))
                                .collect(),
                        )
                    })
                    .collect();
                let trace = trace_with(&offsets, &records);
                let cache_bytes = if policy == PolicyKind::None {
                    0
                } else {
                    capacity_shards.max(1) * SHARD_BYTES
                };
                let config = base_config(policy, cache_bytes);
                let mut predictor = NoPrefetch;
                let simulation =
                    simulate(&trace, &calibration, config, &mut predictor, None).unwrap();
                simulation.summary.check().unwrap();
                // No double counting: the device total is exactly the sum of its
                // named components.
                assert_eq!(
                    simulation.summary.bytes_device_read,
                    simulation.summary.bytes_required_miss
                        + simulation.summary.bytes_phase_cold
                        + simulation.summary.bytes_overread
                        + simulation.summary.bytes_prefetch
                );
                // Monotonic time: each record starts no earlier than the
                // previous one ended.
                let mut previous_end = 0_u64;
                for outcome in &simulation.outcomes {
                    assert!(outcome.start_ns >= previous_end);
                    assert!(outcome.end_ns >= outcome.start_ns);
                    previous_end = outcome.end_ns;
                }
                // Changing the policy never changes the selected set.
                let mut other = base_config(PolicyKind::Lfu, 2 * SHARD_BYTES);
                other.seed = 11;
                let mut predictor = NoPrefetch;
                let other_sim =
                    simulate(&trace, &calibration, other, &mut predictor, None).unwrap();
                let needs: Vec<u64> = simulation
                    .outcomes
                    .iter()
                    .map(|outcome| outcome.bytes_needed)
                    .collect();
                let other_needs: Vec<u64> = other_sim
                    .outcomes
                    .iter()
                    .map(|outcome| outcome.bytes_needed)
                    .collect();
                assert_eq!(needs, other_needs);
                Ok(())
            },
        )
        .unwrap();
}

#[test]
fn simulator_rejects_inconsistent_cache_configuration() {
    let trace = contiguous(&[(0, "p0", vec![1])]);
    let calibration = calibration();
    let mut config = base_config(PolicyKind::None, 0);
    config.cache_bytes = 100;
    let mut predictor = NoPrefetch;
    assert!(matches!(
        simulate(&trace, &calibration, config, &mut predictor, None),
        Err(SimError::Invalid { .. })
    ));
}
