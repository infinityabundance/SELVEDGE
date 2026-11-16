//! Validation of the simulator against the measured engine.
//!
//! Two independent checks, both CPU-only:
//!
//! 1. **Recorded-metrics replay.** Read the per-request JSONL the measured
//!    engine wrote, replay the cache policy over the same read sequence, and
//!    compare (a) the hit/miss decision of every read and (b) the stall
//!    distribution predicted by the calibrated model against the measured one.
//!    The hit/miss check tests the policy implementation exactly. The latency
//!    check is out of sample when the calibration and the target arm are
//!    different files, which is how the court invokes it.
//! 2. **Live store comparison.** Run the same selected sets through the real
//!    `selvedge-store` read path and through the simulator and compare p50/p95/
//!    p99 stall, device bytes and waste. The store path is synchronous, so the
//!    simulator is configured with one server and no overlap for this check.
//!
//! A large divergence is a simulator defect, not a result. The thresholds are
//! fixed here, before the numbers: p50 within 25%, p95 within 35%, p99 within
//! 50%, device bytes within 1%, and zero hit/miss mismatches.

use std::collections::BTreeMap;
use std::path::Path;

use selvedge_core::ShardId;
use selvedge_store::{
    ExecutionScope, PageCache, PolicyKind, QueueLimits, ShardStore, StoreConfig, VerificationMode,
};
use serde::{Deserialize, Serialize};

use crate::calibration::{Calibration, MetricsRecord, ReceiptRef, read_metrics};
use crate::engine::{SimConfig, simulate};
use crate::error::{Result, SimError};
use crate::policy::{NoPrefetch, SimCache};
use crate::prng::SplitMix64;
use crate::report::{CalibrationRef, LatencySummary};
use crate::trace::{Trace, TraceLimits};

/// Schema of a validation receipt.
pub const VALIDATION_SCHEMA: &str = "selvedge/sim-validation/v1";
/// Schema of a store comparison receipt.
pub const COMPARISON_SCHEMA: &str = "selvedge/sim-store-comparison/v1";
/// Schema of a Pareto receipt.
pub const PARETO_SCHEMA: &str = "selvedge/sim-pareto/v1";

/// Pre-registered tolerance for the stall p50, in percent.
pub const TOLERANCE_P50_PCT: f64 = 25.0;
/// Pre-registered tolerance for the stall p95, in percent.
pub const TOLERANCE_P95_PCT: f64 = 35.0;
/// Pre-registered tolerance for the stall p99, in percent.
pub const TOLERANCE_P99_PCT: f64 = 50.0;
/// Pre-registered tolerance for device bytes, in percent.
pub const TOLERANCE_BYTES_PCT: f64 = 1.0;

/// One hit/miss disagreement.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HitMismatch {
    /// Read index in the recorded sequence.
    pub index: u64,
    /// Shard.
    pub shard: ShardId,
    /// What the store recorded.
    pub recorded_hit: bool,
    /// What the replayed policy decided.
    pub replayed_hit: bool,
}

/// Byte comparison for one validation.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ByteComparison {
    /// Recorded device bytes.
    pub measured_device_bytes: u64,
    /// Predicted device bytes.
    pub predicted_device_bytes: u64,
    /// Signed relative discrepancy, percent.
    pub discrepancy_pct: f64,
}

/// Result of replaying a recorded read sequence.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MetricsValidation {
    /// Always [`VALIDATION_SCHEMA`].
    pub schema: String,
    /// Creation stamp.
    pub created_utc: String,
    /// Revision.
    pub revision: String,
    /// Calibration used.
    pub calibration: CalibrationRef,
    /// Target metrics file, hashed.
    pub target: ReceiptRef,
    /// Arm name, for the receipt reader.
    pub arm: String,
    /// Policy replayed.
    pub policy: String,
    /// Cache capacity.
    pub capacity_bytes: u64,
    /// Whether the calibration and the target are different files.
    pub out_of_sample: bool,
    /// Seed.
    pub seed: u64,
    /// Reads replayed.
    pub reads: u64,
    /// Hits the store recorded.
    pub hits_recorded: u64,
    /// Hits the replay produced.
    pub hits_replayed: u64,
    /// Disagreements.
    pub hit_mismatches: u64,
    /// First disagreement, if any.
    pub first_mismatch: Option<HitMismatch>,
    /// Recorded stall distribution.
    pub measured_stall: LatencySummary,
    /// Predicted stall distribution.
    pub predicted_stall: LatencySummary,
    /// Signed discrepancy per percentile, percent.
    pub stall_discrepancy_pct: BTreeMap<String, f64>,
    /// Device byte comparison.
    pub bytes: ByteComparison,
    /// `pass` or `defect`.
    pub verdict: String,
    /// Why the verdict was reached.
    pub verdict_reason: String,
    /// Caveats.
    pub notes: Vec<String>,
}

/// Replay a recorded metrics file through the simulator's policy and latency
/// model.
///
/// # Errors
///
/// Rejects a capacity inconsistent with the policy, malformed metrics and
/// sampling failures.
#[allow(clippy::too_many_arguments)]
pub fn validate_metrics(
    calibration: &Calibration,
    target: &Path,
    arm: &str,
    policy: PolicyKind,
    capacity_bytes: u64,
    seed: u64,
    revision: &str,
    stamp: &str,
) -> Result<MetricsValidation> {
    if policy != PolicyKind::None && capacity_bytes == 0 {
        return Err(SimError::invalid(
            "capacity_bytes",
            "a policy-backed replay needs a capacity",
        ));
    }
    let sequence = read_metrics(target, TraceLimits::default())?;
    let mut cache = SimCache::new(policy, capacity_bytes)?;
    let mut rng = SplitMix64::new(seed);
    let mut measured = Vec::with_capacity(sequence.reads.len());
    let mut predicted = Vec::with_capacity(sequence.reads.len());
    let mut measured_device = 0_u64;
    let mut predicted_device = 0_u64;
    let mut hits_recorded = 0_u64;
    let mut hits_replayed = 0_u64;
    let mut mismatches = 0_u64;
    let mut first_mismatch: Option<HitMismatch> = None;
    for (index, read) in sequence.reads.iter().enumerate() {
        let replayed_hit = cache.get(&read.shard);
        if read.hit {
            hits_recorded = hits_recorded.saturating_add(1);
        }
        if replayed_hit {
            hits_replayed = hits_replayed.saturating_add(1);
        }
        if replayed_hit != read.hit {
            mismatches = mismatches.saturating_add(1);
            if first_mismatch.is_none() {
                first_mismatch = Some(HitMismatch {
                    index: index as u64,
                    shard: read.shard,
                    recorded_hit: read.hit,
                    replayed_hit,
                });
            }
        }
        measured.push(read.stall_ns);
        measured_device = measured_device.saturating_add(read.device_bytes);
        if replayed_hit {
            predicted.push(cache_hit_cost(calibration, &mut rng, read)?);
        } else {
            let bytes = read.bytes_requested;
            let transfer = calibration.sample_transfer_ns(&mut rng, bytes)?;
            let verify = calibration.sample_verify_ns(&mut rng, bytes)?;
            let insert = calibration.sample_insert_ns(&mut rng)?;
            let queue = calibration.sample_queue_wait_ns(&mut rng)?;
            let outcome = cache.insert(read.shard, bytes)?;
            let eviction = if outcome.evictions > 0 {
                calibration.sample_eviction_ns(&mut rng, outcome.evictions)?
            } else {
                0.0
            };
            predicted.push(
                (queue + transfer + verify + insert + eviction)
                    .round()
                    .max(0.0) as u64,
            );
            predicted_device = predicted_device.saturating_add(bytes);
        }
    }
    let measured_stall = LatencySummary::of(&measured);
    let predicted_stall = LatencySummary::of(&predicted);
    let mut stall_discrepancy_pct = BTreeMap::new();
    for (name, measured_value, predicted_value) in [
        (
            "p50",
            measured_stall.p50_ns as f64,
            predicted_stall.p50_ns as f64,
        ),
        (
            "p95",
            measured_stall.p95_ns as f64,
            predicted_stall.p95_ns as f64,
        ),
        (
            "p99",
            measured_stall.p99_ns as f64,
            predicted_stall.p99_ns as f64,
        ),
        ("mean", measured_stall.mean_ns, predicted_stall.mean_ns),
    ] {
        stall_discrepancy_pct.insert(
            name.to_owned(),
            discrepancy_pct(measured_value, predicted_value),
        );
    }
    let bytes = ByteComparison {
        measured_device_bytes: measured_device,
        predicted_device_bytes: predicted_device,
        discrepancy_pct: discrepancy_pct(measured_device as f64, predicted_device as f64),
    };
    let p50 = stall_discrepancy_pct.get("p50").copied().unwrap_or(0.0);
    let p95 = stall_discrepancy_pct.get("p95").copied().unwrap_or(0.0);
    let p99 = stall_discrepancy_pct.get("p99").copied().unwrap_or(0.0);
    let mut reasons = Vec::new();
    if mismatches > 0 {
        reasons.push(format!("{mismatches} hit/miss mismatches"));
    }
    if p50.abs() > TOLERANCE_P50_PCT {
        reasons.push(format!("p50 off by {p50:.1}%"));
    }
    if p95.abs() > TOLERANCE_P95_PCT {
        reasons.push(format!("p95 off by {p95:.1}%"));
    }
    if p99.abs() > TOLERANCE_P99_PCT {
        reasons.push(format!("p99 off by {p99:.1}%"));
    }
    if bytes.discrepancy_pct.abs() > TOLERANCE_BYTES_PCT {
        reasons.push(format!("device bytes off by {:.1}%", bytes.discrepancy_pct));
    }
    let verdict = if reasons.is_empty() {
        "pass".to_owned()
    } else {
        "defect".to_owned()
    };
    let target_ref = ReceiptRef::of(target, "target_metrics")?;
    let out_of_sample = calibration
        .inputs
        .iter()
        .all(|input| input.sha256 != target_ref.sha256);
    Ok(MetricsValidation {
        schema: VALIDATION_SCHEMA.to_owned(),
        created_utc: stamp.to_owned(),
        revision: revision.to_owned(),
        calibration: CalibrationRef::of(calibration),
        target: target_ref,
        arm: arm.to_owned(),
        policy: policy.name().to_owned(),
        capacity_bytes,
        out_of_sample,
        seed,
        reads: sequence.reads.len() as u64,
        hits_recorded,
        hits_replayed,
        hit_mismatches: mismatches,
        first_mismatch,
        measured_stall,
        predicted_stall,
        stall_discrepancy_pct,
        bytes,
        verdict: verdict.clone(),
        verdict_reason: if reasons.is_empty() {
            "within the pre-registered tolerances".to_owned()
        } else {
            reasons.join("; ")
        },
        notes: vec![
            "the hit/miss check is exact: it replays the policy over the recorded read \
             order and compares every decision with the engine's own record"
                .to_owned(),
            format!(
                "out_of_sample={out_of_sample}: a false value means the latency comparison \
                 resamples the same file it was calibrated on and is not a prediction"
            ),
            "prefetch events are excluded from the replay; the measured prefetch arm is a \
             batch event, not a sequence of required reads"
                .to_owned(),
        ],
    })
}

fn cache_hit_cost(
    calibration: &Calibration,
    rng: &mut SplitMix64,
    read: &MetricsRecord,
) -> Result<u64> {
    let copy = calibration.sample_hit_ns(rng, read.bytes_requested)?;
    Ok(copy.round().max(0.0) as u64)
}

fn discrepancy_pct(measured: f64, predicted: f64) -> f64 {
    if measured <= 0.0 {
        return 0.0;
    }
    (predicted - measured) / measured * 100.0
}

/// Read the configured cache capacity for one arm of a store-bench receipt.
///
/// # Errors
///
/// Rejects a missing receipt, an unknown arm and malformed JSON.
pub fn arm_capacity(bench_path: &Path, arm: &str) -> Result<u64> {
    #[derive(Deserialize)]
    struct Receipt {
        arms: Vec<Arm>,
    }
    #[derive(Deserialize)]
    struct Arm {
        arm: String,
        cache_configured_bytes: u64,
    }
    let bytes = std::fs::read(bench_path)?;
    let receipt: Receipt = serde_json::from_slice(&bytes)?;
    receipt
        .arms
        .into_iter()
        .find(|candidate| candidate.arm == arm)
        .map(|candidate| candidate.cache_configured_bytes)
        .ok_or_else(|| SimError::invalid("arm", format!("arm {arm} is not in the receipt")))
}

/// One pass of the real engine over a trace.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StorePass {
    /// `cold` (page cache dropped first) or `warm`.
    pub pass: String,
    /// Records read.
    pub records: u64,
    /// Per-record stall summary.
    pub stall: LatencySummary,
    /// Bytes read from the backing tier.
    pub device_bytes: u64,
    /// Bytes delivered for use.
    pub bytes_used: u64,
    /// Cache hits.
    pub hits: u64,
    /// Wall time.
    pub elapsed_ns: u64,
}

/// One simulated pass over the same trace.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SimPass {
    /// `cold` or `warm`.
    pub pass: String,
    /// Per-record stall summary.
    pub stall: LatencySummary,
    /// Bytes read from the backing tier.
    pub device_bytes: u64,
    /// Bytes needed.
    pub bytes_needed: u64,
    /// Bytes wasted.
    pub bytes_wasted: u64,
}

/// Comparison for one pass.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PassComparison {
    /// Measured pass.
    pub store: StorePass,
    /// Simulated pass.
    pub sim: SimPass,
    /// Signed p50 discrepancy, percent.
    pub p50_discrepancy_pct: f64,
    /// Signed p95 discrepancy, percent.
    pub p95_discrepancy_pct: f64,
    /// Signed p99 discrepancy, percent.
    pub p99_discrepancy_pct: f64,
    /// Signed device-byte discrepancy, percent.
    pub bytes_discrepancy_pct: f64,
    /// `pass` or `defect`.
    pub verdict: String,
}

/// Live comparison receipt.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StoreComparison {
    /// Always [`COMPARISON_SCHEMA`].
    pub schema: String,
    /// Creation stamp.
    pub created_utc: String,
    /// Revision.
    pub revision: String,
    /// Bundle path read.
    pub bundle: String,
    /// Trace identifier.
    pub trace_id: String,
    /// Trace digest.
    pub trace_sha256: String,
    /// Policy.
    pub policy: String,
    /// Cache capacity.
    pub cache_bytes: u64,
    /// Seed.
    pub seed: u64,
    /// Cold pass comparison.
    pub cold: PassComparison,
    /// Warm pass comparison.
    pub warm: PassComparison,
    /// Notes.
    pub notes: Vec<String>,
}

/// Run the real store and the simulator over the same trace and compare.
///
/// The store's read path is synchronous, so `config.queue_depth` must be 1 and
/// `config.compute_ns_per_record` must be zero; anything else is a configuration
/// error rather than a silently different comparison.
///
/// # Errors
///
/// Rejects a mismatched configuration, an empty trace, store failures and
/// simulator failures.
pub fn compare_with_store(
    trace: &Trace,
    bundle_path: &Path,
    config: &SimConfig,
    calibration: &Calibration,
    revision: &str,
    stamp: &str,
) -> Result<StoreComparison> {
    if config.queue_depth != 1 {
        return Err(SimError::invalid(
            "queue_depth",
            "the measured engine is synchronous; queue_depth must be 1",
        ));
    }
    if config.compute_ns_per_record != 0 {
        return Err(SimError::invalid(
            "compute_ns_per_record",
            "the measured engine has no overlap; must be zero",
        ));
    }
    if config.placement.is_some() {
        return Err(SimError::invalid(
            "placement",
            "a live comparison must use the bundle's own physical order",
        ));
    }
    let cold = run_store(trace, bundle_path, config, true, "cold")?;
    let warm = run_store(trace, bundle_path, config, false, "warm")?;
    let sim_cold = run_sim(trace, calibration, config, "cold")?;
    let sim_warm = run_sim(trace, calibration, config, "warm")?;
    let cold_comparison = compare_pass(cold, sim_cold);
    let warm_comparison = compare_pass(warm, sim_warm);
    Ok(StoreComparison {
        schema: COMPARISON_SCHEMA.to_owned(),
        created_utc: stamp.to_owned(),
        revision: revision.to_owned(),
        bundle: bundle_path.display().to_string(),
        trace_id: trace.manifest.trace_id.clone(),
        trace_sha256: trace.manifest.records_sha256.to_hex(),
        policy: config.policy.name().to_owned(),
        cache_bytes: config.cache_bytes,
        seed: config.seed,
        cold: cold_comparison,
        warm: warm_comparison,
        notes: vec![
            "the store's page-cache drop is an advisory posix_fadvise; the cold pass is \
             reported as advisory-cold, not as proof of a device read"
                .to_owned(),
            "the warm pass reuses whatever the first pass left in the page cache; it is \
             a page-cache measurement, not a cache-policy win"
                .to_owned(),
            "the simulator is charged the measured insertion, eviction and hit-copy costs \
             on the same path as the store, so the comparison is of scheduling, not of \
             components it does not model"
                .to_owned(),
            "the executed set is the union of the trace's selections in both runs; exact \
             mode semantics are unchanged by construction"
                .to_owned(),
        ],
    })
}

fn run_store(
    trace: &Trace,
    bundle_path: &Path,
    config: &SimConfig,
    drop_cache: bool,
    label: &str,
) -> Result<StorePass> {
    let store_config = StoreConfig {
        cache_bytes: config.cache_bytes,
        policy: config.policy,
        verification: VerificationMode::Always,
        queue: QueueLimits {
            max_pending: 64,
            max_in_flight: 16,
            max_in_flight_bytes: config.max_in_flight_bytes,
        },
        max_shard_bytes: 64 * 1024 * 1024,
        max_run_bytes: config.max_run_bytes.max(1),
        coalesce_gap_bytes: 0,
        metrics: None,
    };
    store_config.validate()?;
    let store = ShardStore::open(bundle_path, store_config)?;
    let scope = ExecutionScope::exact(trace.shards().iter().map(|shard| shard.id))?;
    if drop_cache {
        PageCache::drop_file(bundle_path)?;
    }
    let max_extent = trace
        .shards()
        .iter()
        .map(|shard| shard.bytes)
        .max()
        .unwrap_or(0);
    if max_extent == 0 {
        return Err(SimError::invalid("shards", "no bytes in the trace table"));
    }
    let mut buffer = vec![0_u8; max_extent as usize];
    let start = std::time::Instant::now();
    let mut stalls = Vec::new();
    let mut device_bytes = 0_u64;
    let mut bytes_used = 0_u64;
    let mut hits = 0_u64;
    for record in &trace.records {
        let mut order: Vec<ShardId> = record.selected.clone();
        order.sort_unstable_by_key(|id| {
            trace
                .shards()
                .iter()
                .find(|shard| shard.id == *id)
                .map_or(u64::MAX, |shard| shard.offset)
        });
        let mut stall = 0_u64;
        for shard in order {
            let report = store.read_required(&scope, &shard, &mut buffer)?;
            stall = stall.saturating_add(report.stall_ns);
            device_bytes = device_bytes.saturating_add(report.device_bytes);
            bytes_used = bytes_used.saturating_add(report.bytes_used);
            if report.hit {
                hits = hits.saturating_add(1);
            }
        }
        stalls.push(stall);
    }
    let elapsed_ns = u64::try_from(start.elapsed().as_nanos()).unwrap_or(u64::MAX);
    Ok(StorePass {
        pass: label.to_owned(),
        records: stalls.len() as u64,
        stall: LatencySummary::of(&stalls),
        device_bytes,
        bytes_used,
        hits,
        elapsed_ns,
    })
}

fn run_sim(
    trace: &Trace,
    calibration: &Calibration,
    config: &SimConfig,
    label: &str,
) -> Result<SimPass> {
    let mut sim_config = config.clone();
    sim_config.start_record = 0;
    let mut predictor = NoPrefetch;
    let simulation = simulate(trace, calibration, sim_config, &mut predictor, None)?;
    Ok(SimPass {
        pass: label.to_owned(),
        stall: simulation.summary.stall,
        device_bytes: simulation.summary.bytes_device_read,
        bytes_needed: simulation.summary.bytes_needed,
        bytes_wasted: simulation.summary.bytes_wasted,
    })
}

fn compare_pass(store: StorePass, sim: SimPass) -> PassComparison {
    let p50 = discrepancy_pct(store.stall.p50_ns as f64, sim.stall.p50_ns as f64);
    let p95 = discrepancy_pct(store.stall.p95_ns as f64, sim.stall.p95_ns as f64);
    let p99 = discrepancy_pct(store.stall.p99_ns as f64, sim.stall.p99_ns as f64);
    let bytes = discrepancy_pct(store.device_bytes as f64, sim.device_bytes as f64);
    let ok = p50.abs() <= TOLERANCE_P50_PCT
        && p95.abs() <= TOLERANCE_P95_PCT
        && p99.abs() <= TOLERANCE_P99_PCT
        && bytes.abs() <= TOLERANCE_BYTES_PCT;
    PassComparison {
        store,
        sim,
        p50_discrepancy_pct: p50,
        p95_discrepancy_pct: p95,
        p99_discrepancy_pct: p99,
        bytes_discrepancy_pct: bytes,
        verdict: if ok { "pass" } else { "defect" }.to_owned(),
    }
}

/// One row of the Pareto table.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ParetoRow {
    /// Arm label (`<policy>-<capacity-fraction>`).
    pub arm: String,
    /// Policy.
    pub policy: String,
    /// Cache capacity.
    pub cache_bytes: u64,
    /// Capacity as a fraction of the unique payload.
    pub cache_fraction: f64,
    /// Tokens (records) simulated.
    pub records: u64,
    /// Device bytes per record.
    pub bytes_per_record: f64,
    /// Device bytes per selected shard.
    pub bytes_per_shard: f64,
    /// p50 stall.
    pub p50_stall_ns: u64,
    /// p95 stall.
    pub p95_stall_ns: u64,
    /// p99 stall.
    pub p99_stall_ns: u64,
    /// Hit bytes over needed bytes.
    pub hit_rate: f64,
    /// Wasted bytes.
    pub wasted_bytes: u64,
}

/// Pareto receipt.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ParetoReport {
    /// Always [`PARETO_SCHEMA`].
    pub schema: String,
    /// Creation stamp.
    pub creation: String,
    /// Revision.
    pub revision: String,
    /// Trace.
    pub trace_id: String,
    /// Trace digest.
    pub trace_sha256: String,
    /// Calibration.
    pub calibration: CalibrationRef,
    /// Quality statement.
    pub quality: String,
    /// Rows.
    pub rows: Vec<ParetoRow>,
    /// Notes.
    pub notes: Vec<String>,
}

/// Run a Pareto sweep: policies at capacity fractions of the unique payload.
///
/// Quality is fixed by construction: every arm executes the trace's selected
/// sets exactly, so the table trades bytes and tail latency, not quality.
///
/// # Errors
///
/// Rejects invalid fractions and propagates simulator failures.
pub fn run_pareto(
    trace: &Trace,
    calibration: &Calibration,
    base: &SimConfig,
    policies: &[PolicyKind],
    fractions: &[f64],
    revision: &str,
    stamp: &str,
) -> Result<ParetoReport> {
    let payload: u64 = trace
        .shards()
        .iter()
        .map(|shard| shard.bytes)
        .fold(0_u64, u64::saturating_add);
    let records_per_shard: f64 = {
        let total: u64 = trace
            .records
            .iter()
            .map(|record| record.selected.len() as u64)
            .sum();
        if total == 0 {
            0.0
        } else {
            trace.records.len() as f64 / total as f64
        }
    };
    let mut rows = Vec::new();
    for policy in policies {
        for fraction in fractions {
            if *fraction <= 0.0 || *fraction > 1.0 {
                return Err(SimError::invalid("fraction", "must be in (0, 1]"));
            }
            let capacity = (payload as f64 * fraction) as u64;
            let mut config = base.clone();
            config.policy = *policy;
            config.cache_bytes = capacity;
            config.start_record = 0;
            let mut predictor = NoPrefetch;
            let simulation = simulate(trace, calibration, config, &mut predictor, None)?;
            let records = simulation.summary.evaluated_records;
            let bytes_per_record = if records == 0 {
                0.0
            } else {
                simulation.summary.bytes_device_read as f64 / records as f64
            };
            let bytes_per_shard = bytes_per_record * records_per_shard;
            let hit_rate = if simulation.summary.bytes_needed == 0 {
                0.0
            } else {
                simulation.summary.bytes_hit as f64 / simulation.summary.bytes_needed as f64
            };
            rows.push(ParetoRow {
                arm: format!("{}-{fraction:.3}", policy.name()),
                policy: policy.name().to_owned(),
                cache_bytes: capacity,
                cache_fraction: *fraction,
                records,
                bytes_per_record,
                bytes_per_shard,
                p50_stall_ns: simulation.summary.stall.p50_ns,
                p95_stall_ns: simulation.summary.stall.p95_ns,
                p99_stall_ns: simulation.summary.stall.p99_ns,
                hit_rate,
                wasted_bytes: simulation.summary.bytes_wasted,
            });
        }
    }
    Ok(ParetoReport {
        schema: PARETO_SCHEMA.to_owned(),
        creation: stamp.to_owned(),
        revision: revision.to_owned(),
        trace_id: trace.manifest.trace_id.clone(),
        trace_sha256: trace.manifest.records_sha256.to_hex(),
        calibration: CalibrationRef::of(calibration),
        quality: "fixed: every arm executes the trace's selected sets exactly; no quality \
                  trade is present in this table"
            .to_owned(),
        rows,
        notes: vec![
            "bytes/token and p95 are simulated, not measured; the simulator is validated by \
             the metrics replay and the live store comparison"
                .to_owned(),
            "a smaller cache is not a quality change in exact mode: it costs bytes and \
             latency, never a different selection"
                .to_owned(),
        ],
    })
}
