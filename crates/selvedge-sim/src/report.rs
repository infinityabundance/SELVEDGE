//! Simulation outputs and their accounting invariants.
//!
//! Every byte counter is defined once, here, so the conservation property is a
//! statement about named fields rather than a hope:
//!
//! ```text
//! bytes_needed          = bytes_hit + bytes_required_miss
//! bytes_device_read     = bytes_required_miss + bytes_phase_cold
//!                       + bytes_overread + bytes_prefetch
//! bytes_prefetch        = bytes_prefetch_used + bytes_prefetch_wasted
//! bytes_wasted          = bytes_overread + bytes_prefetch_wasted
//! ```
//!
//! `bytes_phase_cold` is the phase warm-up: bytes fetched to establish a
//! working set. They are neither needed by the record that triggered them nor
//! wasted, so they get their own term instead of being folded into either.
//!
//! `bytes_device_read` counts completed transfers only. A prefetch cancelled
//! before completion is counted in `bytes_prefetch_cancelled` and never in
//! `bytes_device_read`, because the simulator cannot see part of a transfer.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::calibration::Calibration;
use crate::error::{Result, SimError};
use crate::placement::{ObjectiveReport, PlacementKind};
use crate::prng::percentile_u64;
use crate::trace::{Trace, TraceManifest};

/// Schema of a simulation report.
pub const SIM_REPORT_SCHEMA: &str = "selvedge/sim-report/v1";

/// Latency summary in nanoseconds.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LatencySummary {
    /// Sample count.
    pub n: u64,
    /// Mean.
    pub mean_ns: f64,
    /// p50.
    pub p50_ns: u64,
    /// p90.
    pub p90_ns: u64,
    /// p95.
    pub p95_ns: u64,
    /// p99.
    pub p99_ns: u64,
    /// Maximum.
    pub max_ns: u64,
    /// Sum.
    pub total_ns: u64,
}

impl LatencySummary {
    /// Summarise a sample (the slice is copied, not reordered in place).
    #[must_use]
    pub fn of(samples: &[u64]) -> Self {
        if samples.is_empty() {
            return Self::default();
        }
        let mut sorted = samples.to_vec();
        sorted.sort_unstable();
        let total: u64 = sorted.iter().copied().fold(0, u64::saturating_add);
        Self {
            n: sorted.len() as u64,
            mean_ns: total as f64 / sorted.len() as f64,
            p50_ns: percentile_u64(&sorted, 50.0),
            p90_ns: percentile_u64(&sorted, 90.0),
            p95_ns: percentile_u64(&sorted, 95.0),
            p99_ns: percentile_u64(&sorted, 99.0),
            max_ns: *sorted.last().unwrap_or(&0),
            total_ns: total,
        }
    }
}

/// Aggregate simulator counters.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SimSummary {
    /// Records simulated.
    pub records: u64,
    /// Records whose stall sample is in the latency summary.
    pub evaluated_records: u64,
    /// Consumer stall per record.
    pub stall: LatencySummary,
    /// Bytes the selected sets needed.
    pub bytes_needed: u64,
    /// Needed bytes served from host RAM (including prefetch-served bytes).
    pub bytes_hit: u64,
    /// Needed bytes fetched by a required transfer.
    pub bytes_required_miss: u64,
    /// Bytes completed from the backing tier, all transfers.
    pub bytes_device_read: u64,
    /// Bytes read to bridge coalescing gaps.
    pub bytes_overread: u64,
    /// Speculative bytes completed.
    pub bytes_prefetch: u64,
    /// Speculative bytes later used by a required read.
    pub bytes_prefetch_used: u64,
    /// Speculative bytes never used.
    pub bytes_prefetch_wasted: u64,
    /// Speculative bytes cancelled before completion.
    pub bytes_prefetch_cancelled: u64,
    /// `bytes_overread + bytes_prefetch_wasted`.
    pub bytes_wasted: u64,
    /// Bytes fetched to establish a phase working set or envelope.
    pub bytes_phase_cold: u64,
    /// Cache lookups.
    pub hits: u64,
    /// Cache lookups that missed.
    pub misses: u64,
    /// Evictions.
    pub evictions: u64,
    /// Evicted bytes.
    pub evicted_bytes: u64,
    /// Evicted bytes that had never been reused.
    pub evicted_unused_bytes: u64,
    /// Prefetch candidates proposed.
    pub prefetch_candidates: u64,
    /// Prefetch candidates accepted for transfer.
    pub prefetch_accepted: u64,
    /// Prefetch candidates skipped because the cache could not hold them.
    pub prefetch_rejected_capacity: u64,
    /// Envelope invalidations.
    pub envelope_invalidations: u64,
    /// Required transfers that had to wait for in-flight byte budget.
    pub backpressure_waits: u64,
    /// Records whose exposed stall exceeded the deadline.
    pub deadline_misses: u64,
    /// Deadline used, when one was configured.
    pub deadline_ns: Option<u64>,
    /// Maximum simulated timestamp.
    pub end_mono_ns: u64,
    /// Recorded reasons a run could not proceed, if any.
    pub errors: Vec<String>,
}

impl SimSummary {
    /// Check the byte-conservation invariants.
    ///
    /// # Errors
    ///
    /// Returns a typed error naming the invariant that failed.
    pub fn check(&self) -> Result<()> {
        let hit_plus_miss = self.bytes_hit.saturating_add(self.bytes_required_miss);
        if hit_plus_miss != self.bytes_needed {
            return Err(SimError::invalid(
                "bytes_needed",
                format!(
                    "{} needed, {} served from cache + {} fetched",
                    self.bytes_needed, self.bytes_hit, self.bytes_required_miss
                ),
            ));
        }
        let device_sum = self
            .bytes_required_miss
            .saturating_add(self.bytes_phase_cold)
            .saturating_add(self.bytes_overread)
            .saturating_add(self.bytes_prefetch);
        if device_sum != self.bytes_device_read {
            return Err(SimError::invalid(
                "bytes_device_read",
                format!(
                    "{} read, components sum to {}",
                    self.bytes_device_read, device_sum
                ),
            ));
        }
        let prefetch_sum = self
            .bytes_prefetch_used
            .saturating_add(self.bytes_prefetch_wasted);
        if prefetch_sum != self.bytes_prefetch {
            return Err(SimError::invalid(
                "bytes_prefetch",
                format!(
                    "{} speculative bytes, {} used + {} wasted",
                    self.bytes_prefetch, self.bytes_prefetch_used, self.bytes_prefetch_wasted
                ),
            ));
        }
        let waste_sum = self
            .bytes_overread
            .saturating_add(self.bytes_prefetch_wasted);
        if waste_sum != self.bytes_wasted {
            return Err(SimError::invalid(
                "bytes_wasted",
                format!(
                    "{} wasted, components sum to {}",
                    self.bytes_wasted, waste_sum
                ),
            ));
        }
        Ok(())
    }
}

/// Per-phase counters.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PhaseSummary {
    /// Phase identifier.
    pub phase_id: String,
    /// Records evaluated in this phase.
    pub records: u64,
    /// Bytes needed by those records.
    pub bytes_needed: u64,
    /// Bytes fetched from the backing tier for those records.
    pub bytes_device_read: u64,
    /// Bytes fetched to warm the phase working set.
    pub bytes_phase_cold: u64,
    /// Stall summary.
    pub stall: LatencySummary,
}

/// Per-task counters.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TaskSummary {
    /// Task label; `unlabelled` when the trace carries none.
    pub task: String,
    /// Records evaluated.
    pub records: u64,
    /// Bytes needed.
    pub bytes_needed: u64,
    /// Bytes fetched from the backing tier.
    pub bytes_device_read: u64,
    /// Bytes fetched to warm a working set.
    pub bytes_phase_cold: u64,
    /// p95 stall.
    pub stall_p95_ns: u64,
}

/// One record's outcome, for replay checks and debugging.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RecordOutcome {
    /// Record index in the trace.
    pub index: u64,
    /// Consumer time at which the record's data was needed.
    pub start_ns: u64,
    /// Consumer time after the record's stall and compute.
    pub end_ns: u64,
    /// Request.
    pub request_id: String,
    /// Phase.
    pub phase_id: String,
    /// Token index.
    pub token_index: u64,
    /// Layer.
    pub layer: u32,
    /// Bytes the record needed.
    pub bytes_needed: u64,
    /// Bytes served from host RAM.
    pub bytes_hit: u64,
    /// Bytes fetched for it from the backing tier.
    pub bytes_fetched: u64,
    /// Exposed consumer stall.
    pub stall_ns: u64,
    /// Whether the deadline was missed.
    pub deadline_missed: bool,
    /// Prefetch candidates proposed after the record.
    pub prefetch_candidates: u64,
    /// Speculative bytes consumed by this record.
    pub prefetch_used_bytes: u64,
}

/// Reference to the calibration a report used.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CalibrationRef {
    /// Calibration stamp.
    pub stamp: String,
    /// Revision of the calibration inputs.
    pub revision: String,
    /// Reference shard size.
    pub shard_bytes: u64,
    /// Measured p50 transfer for one shard.
    pub transfer_p50_ns: u64,
    /// Measured p50 verification for one shard.
    pub verify_p50_ns: u64,
    /// Size-law points.
    pub size_law_points: Vec<(u64, u64)>,
}

impl CalibrationRef {
    /// Build a reference from a calibration.
    #[must_use]
    pub fn of(calibration: &Calibration) -> Self {
        Self {
            stamp: calibration.stamp.clone(),
            revision: calibration.revision.clone(),
            shard_bytes: calibration.shard_bytes,
            transfer_p50_ns: calibration.transfer.p50,
            verify_p50_ns: calibration.verify.p50,
            size_law_points: calibration
                .size_law
                .points
                .iter()
                .map(|point| (point.bytes, point.p50_ns))
                .collect(),
        }
    }
}

/// The configuration a run used, as recorded.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ConfigSummary {
    /// Cache policy name.
    pub policy: String,
    /// Cache capacity.
    pub cache_bytes: u64,
    /// Concurrent device transfers.
    pub queue_depth: u32,
    /// In-flight byte budget.
    pub max_in_flight_bytes: u64,
    /// Coalescing gap bridged.
    pub coalesce_gap_bytes: u64,
    /// Largest coalesced run.
    pub max_run_bytes: u64,
    /// Whether verification was charged.
    pub verify: bool,
    /// Compute time per record available for overlap.
    pub compute_ns_per_record: u64,
    /// Deadline, when configured.
    pub deadline_ns: Option<u64>,
    /// Seed.
    pub seed: u64,
    /// Backpressure mode.
    pub backpressure: String,
    /// Predictor name.
    pub predictor: String,
    /// Whether the predictor uses future information.
    pub predictor_uses_future: bool,
    /// Two-timescale working set enabled.
    pub two_timescale: bool,
    /// Working-set source.
    pub working_set_source: String,
    /// Record index where evaluation started.
    pub start_record: u64,
}

/// Placement comparison row.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PlacementReport {
    /// Layout family.
    pub kind: PlacementKind,
    /// Unique shards.
    pub unique_shards: u64,
    /// Unique bytes.
    pub unique_bytes: u64,
    /// Placed bytes.
    pub placed_bytes: u64,
    /// Replicated bytes.
    pub replicated_bytes: u64,
    /// Storage amplification.
    pub amplification: f64,
    /// Simulated p95 stall under this layout.
    pub stall_p95_ns: u64,
    /// Simulated bytes read from the backing tier.
    pub bytes_device_read: u64,
    /// Speculative bytes never used.
    pub unused_prefetch_bytes: u64,
    /// Surrogate objective.
    pub objective: ObjectiveReport,
}

/// Envelope utility, measured in bytes and time, never in prediction accuracy.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EnvelopeUtility {
    /// Bytes the baseline read from the backing tier.
    pub baseline_bytes: u64,
    /// Bytes the envelope run read.
    pub envelope_bytes: u64,
    /// `baseline - envelope`; negative means the envelope read more.
    pub bytes_not_transferred: i64,
    /// Baseline p95 stall.
    pub baseline_stall_p95_ns: u64,
    /// Envelope p95 stall.
    pub envelope_stall_p95_ns: u64,
    /// `baseline - envelope` p95, signed.
    pub exposed_time_saved_ns: i64,
    /// Speculative bytes the envelope fetched and nothing used.
    pub unused_prefetch_bytes: u64,
    /// Bytes fetched to establish the envelope at phase start.
    pub cold_bytes_charged: u64,
    /// Envelopes invalidated during evaluation.
    pub invalidations: u64,
    /// Phases seen in evaluation with no envelope.
    pub uncovered_phases: Vec<String>,
    /// Interpretation statement.
    pub interpretation: String,
}

/// The full simulation report.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SimReport {
    /// Always [`SIM_REPORT_SCHEMA`].
    pub schema: String,
    /// Creation stamp.
    pub stamp: String,
    /// Repository revision.
    pub revision: String,
    /// Trace identifier.
    pub trace_id: String,
    /// Trace records digest.
    pub trace_sha256: String,
    /// Trace source (`measured`, `reconstructed`, `synthetic`).
    pub trace_source: String,
    /// Calibration reference.
    pub calibration: CalibrationRef,
    /// Configuration.
    pub config: ConfigSummary,
    /// Aggregate counters.
    pub summary: SimSummary,
    /// Per-phase rows.
    pub phases: Vec<PhaseSummary>,
    /// Per-task rows.
    pub tasks: Vec<TaskSummary>,
    /// Placement row, when a layout was supplied.
    pub placement: Option<PlacementReport>,
    /// Envelope utility, when envelopes were applied.
    pub envelope: Option<EnvelopeUtility>,
    /// Caveats.
    pub notes: Vec<String>,
}

impl SimReport {
    /// Write the report.
    ///
    /// # Errors
    ///
    /// Propagates I/O and serialization failures.
    pub fn write(&self, path: &std::path::Path) -> Result<()> {
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(path, serde_json::to_vec_pretty(self)?)?;
        Ok(())
    }

    /// Load a report.
    ///
    /// # Errors
    ///
    /// Rejects a wrong schema and propagates I/O and JSON failures.
    pub fn load(path: &std::path::Path) -> Result<Self> {
        let bytes = std::fs::read(path)?;
        let report: Self = serde_json::from_slice(&bytes)?;
        if report.schema != SIM_REPORT_SCHEMA {
            return Err(SimError::invalid(
                "schema",
                format!("sim report schema is not {SIM_REPORT_SCHEMA}"),
            ));
        }
        Ok(report)
    }

    /// Check the summary invariants.
    ///
    /// # Errors
    ///
    /// Propagates a failed conservation check.
    pub fn check(&self) -> Result<()> {
        self.summary.check()
    }
}

/// Build a report from a finished simulation.
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn report(
    trace: &Trace,
    calibration: &Calibration,
    config: ConfigSummary,
    summary: SimSummary,
    phases: Vec<PhaseSummary>,
    tasks: Vec<TaskSummary>,
    placement: Option<PlacementReport>,
    envelope: Option<EnvelopeUtility>,
    stamp: &str,
    revision: &str,
    notes: Vec<String>,
) -> SimReport {
    SimReport {
        schema: SIM_REPORT_SCHEMA.to_owned(),
        stamp: stamp.to_owned(),
        revision: revision.to_owned(),
        trace_id: trace.manifest.trace_id.clone(),
        trace_sha256: trace.manifest.records_sha256.to_hex(),
        trace_source: trace.manifest.source.name().to_owned(),
        calibration: CalibrationRef::of(calibration),
        config,
        summary,
        phases,
        tasks,
        placement,
        envelope,
        notes,
    }
}

/// Trace facts a report repeats so it can be read without the trace file.
#[must_use]
pub fn trace_facts(manifest: &TraceManifest) -> BTreeMap<String, String> {
    BTreeMap::from([
        ("trace_id".to_owned(), manifest.trace_id.clone()),
        ("source".to_owned(), manifest.source.name().to_owned()),
        ("mode".to_owned(), manifest.mode.to_string()),
        ("records".to_owned(), manifest.records.to_string()),
        ("shards".to_owned(), manifest.shards.len().to_string()),
    ])
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn latency_summary_uses_nearest_rank() {
        let samples: Vec<u64> = (1..=100).collect();
        let summary = LatencySummary::of(&samples);
        assert_eq!(summary.n, 100);
        assert_eq!(summary.p50_ns, 50);
        assert_eq!(summary.p95_ns, 95);
        assert_eq!(summary.p99_ns, 99);
        assert_eq!(summary.max_ns, 100);
        assert_eq!(summary.total_ns, 5050);
        assert!((summary.mean_ns - 50.5).abs() < 1e-9);
    }

    #[test]
    fn conservation_checks_reject_a_bad_summary() {
        let good = SimSummary {
            bytes_needed: 100,
            bytes_hit: 40,
            bytes_required_miss: 60,
            bytes_device_read: 70,
            bytes_overread: 10,
            bytes_prefetch: 0,
            bytes_prefetch_used: 0,
            bytes_prefetch_wasted: 0,
            bytes_wasted: 10,
            ..SimSummary::default()
        };
        good.check().unwrap();
        let mut bad = good.clone();
        bad.bytes_device_read = 71;
        assert!(bad.check().is_err());
        let mut bad = good.clone();
        bad.bytes_hit = 39;
        assert!(bad.check().is_err());
        let mut bad = good;
        bad.bytes_wasted = 11;
        assert!(bad.check().is_err());
    }
}
