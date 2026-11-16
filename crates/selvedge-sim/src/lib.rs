//! `selvedge-sim` — trace-driven execution and deadline simulation.
//!
//! This crate owns the M3 part-2 boundary: a versioned execution trace, a
//! calibrated discrete-event simulator, cache and prefetch policies behind
//! traits, execution envelopes, placement optimisation and validation against
//! the measured engine.
//!
//! # The honesty boundary
//!
//! - **A trace says what was selected, never what should have been.** There is
//!   no field for an expected set and every record struct rejects unknown
//!   fields.
//! - **A prefetch is advisory.** A wrong prediction is a stall or a wasted
//!   byte; the executed set is the trace's.
//! - **A simulation is not a measurement.** Every report names its calibration,
//!   the seed, the trace digest and the measured receipts the calibration came
//!   from, and the validation receipts compare the simulator with the real
//!   engine on the same workload.
//! - **Envelope utility is bytes and time**, never prediction accuracy.
//!
//! # Entry points
//!
//! - [`trace`] — the schema, its validation and the record digest.
//! - [`calibration`] — empirical distributions, parametric fits and residuals.
//! - [`engine`] — the event simulation and the policies it drives.
//! - [`envelope`] — phase candidate supersets built from a disjoint prefix.
//! - [`placement`] — candidate layouts and the weighted co-access surrogate.
//! - [`synthetic`] — controlled hot/cold, phase-switch and chunk-union studies.
//! - [`compare`] — replay and live comparison against `selvedge-store`.
//! - [`report`] — report schema and the byte-conservation invariants.

#![forbid(unsafe_code)]
#![deny(missing_docs)]

pub mod calibration;
pub mod compare;
pub mod engine;
pub mod envelope;
pub mod error;
pub mod placement;
pub mod policy;
pub mod prng;
pub mod report;
pub mod synthetic;
pub mod trace;

pub use calibration::{
    CALIBRATION_SCHEMA, Calibration, CalibrationInput, FitReport, QuantileResidual, ReceiptRef,
    SampleStats, SizeLaw, SizePoint, calibrate, fit_reports, normal_quantile,
};
pub use compare::{
    COMPARISON_SCHEMA, MetricsValidation, PARETO_SCHEMA, ParetoReport, ParetoRow, StoreComparison,
    VALIDATION_SCHEMA, arm_capacity, compare_with_store, run_pareto, validate_metrics,
};
pub use engine::{
    BackpressureMode, SimConfig, Simulation, TwoTimescaleConfig, WorkingSetSource, simulate,
};
pub use envelope::{
    ENVELOPE_SCHEMA, Envelope, EnvelopeParams, EnvelopeSet, Invalidation, build as build_envelopes,
    evaluation_start,
};
pub use error::{Result, SimError};
pub use placement::{
    CoAccessWeights, ObjectiveReport, PLACEMENT_SCHEMA, PlacedExtent, Placement, PlacementKind,
    PlacementParams, WeightedEdge, build as build_placement, objective,
    weights as coaccess_weights,
};
pub use policy::{
    EnvelopePrefetch, LastValuePrefetch, NoPrefetch, OracleNextRecordPrefetch, PredictContext,
    Predictor, SimCache,
};
pub use prng::SplitMix64;
pub use report::{
    CalibrationRef, ConfigSummary, EnvelopeUtility, LatencySummary, PhaseSummary, PlacementReport,
    RecordOutcome, SIM_REPORT_SCHEMA, SimReport, SimSummary, TaskSummary,
};
pub use synthetic::{
    MAX_SYNTHETIC_RECORDS, SyntheticSpec, SyntheticTrace, generate as generate_trace,
};
pub use trace::{
    DEFAULT_MAX_LINE_BYTES, DEFAULT_MAX_RECORDS, DEFAULT_MAX_SHARDS, ExactBasis, ExactBasisKind,
    ObservedStep, TRACE_MANIFEST_SCHEMA, TRACE_SCHEMA, Trace, TraceBundle, TraceLimits,
    TraceManifest, TraceProvenance, TraceRecord, TraceShard, TraceSource, TraceTimestamps,
};

/// Run a simulation and assemble a report.
///
/// # Errors
///
/// Propagates simulator failures and sample-table inconsistencies.
#[allow(clippy::too_many_arguments)]
pub fn simulate_report(
    trace: &Trace,
    calibration: &Calibration,
    config: SimConfig,
    predictor: &mut dyn Predictor,
    envelopes: Option<&EnvelopeSet>,
    weights: Option<&CoAccessWeights>,
    lambda_bytes: f64,
    lambda_eviction: f64,
    stamp: &str,
    revision: &str,
    mut notes: Vec<String>,
) -> Result<SimReport> {
    let start_record = config.start_record;
    let config_summary = config.summary(predictor, start_record);
    let placement_input = config.placement.clone();
    let simulation = simulate(trace, calibration, config, predictor, envelopes)?;
    simulation.summary.check()?;
    if predictor.uses_future() {
        notes.push(format!(
            "predictor '{}' reads the next record: its numbers are an upper bound and are \
             not deployable",
            predictor.name()
        ));
    }
    if let Some(set) = envelopes
        && start_record < set.build_records
    {
        notes.push(
            "start_record is inside the envelope build prefix; the run is therefore \
             contaminated and must not be reported as a held-out result"
                .to_owned(),
        );
    }
    let placement = match (&placement_input, weights) {
        (Some(placement), Some(weights)) => Some(PlacementReport {
            kind: placement.kind,
            unique_shards: placement.unique_shards,
            unique_bytes: placement.unique_bytes,
            placed_bytes: placement.placed_bytes,
            replicated_bytes: placement.replicated_bytes,
            amplification: placement.amplification,
            stall_p95_ns: simulation.summary.stall.p95_ns,
            bytes_device_read: simulation.summary.bytes_device_read,
            unused_prefetch_bytes: simulation.summary.bytes_prefetch_wasted,
            objective: objective(placement, weights, lambda_bytes, lambda_eviction),
        }),
        (Some(placement), None) => Some(PlacementReport {
            kind: placement.kind,
            unique_shards: placement.unique_shards,
            unique_bytes: placement.unique_bytes,
            placed_bytes: placement.placed_bytes,
            replicated_bytes: placement.replicated_bytes,
            amplification: placement.amplification,
            stall_p95_ns: simulation.summary.stall.p95_ns,
            bytes_device_read: simulation.summary.bytes_device_read,
            unused_prefetch_bytes: simulation.summary.bytes_prefetch_wasted,
            objective: ObjectiveReport {
                kind: placement.kind,
                adjacent_coaccessed_pairs: 0,
                estimated_stall_saved_ns: 0.0,
                unique_bytes: placement.unique_bytes,
                placed_bytes: placement.placed_bytes,
                replicated_bytes: placement.replicated_bytes,
                objective: f64::NAN,
                lambda_bytes,
                lambda_eviction,
                interpretation: "no co-access weights were supplied, so the surrogate \
                                  objective is undefined; the simulated row is still valid"
                    .to_owned(),
            },
        }),
        (None, _) => None,
    };
    Ok(report::report(
        trace,
        calibration,
        config_summary,
        simulation.summary,
        simulation.phases,
        simulation.tasks,
        placement,
        None,
        stamp,
        revision,
        notes,
    ))
}

/// Measure envelope utility against a no-envelope baseline.
///
/// Both runs start at the envelope set's evaluation boundary, so the build
/// prefix is never evaluated. The baseline uses the same cache capacity and
/// policy with no working set and no prefetch.
///
/// # Errors
///
/// Rejects a configuration whose start record precedes the evaluation boundary
/// and propagates simulator failures.
pub fn measure_envelope(
    trace: &Trace,
    calibration: &Calibration,
    base: &SimConfig,
    envelopes: &EnvelopeSet,
) -> Result<(Simulation, Simulation, EnvelopeUtility)> {
    let start = evaluation_start(envelopes) as u64;
    let mut baseline_config = base.clone();
    baseline_config.start_record = start;
    baseline_config.two_timescale = None;
    let mut none = NoPrefetch;
    let baseline = simulate(trace, calibration, baseline_config, &mut none, None)?;
    let mut envelope_config = base.clone();
    envelope_config.start_record = start;
    envelope_config.two_timescale = Some(TwoTimescaleConfig {
        source: WorkingSetSource::Envelope,
        capacity_fraction: 1.0,
        min_usage: 1,
    });
    let mut predictor = EnvelopePrefetch;
    let envelope_run = simulate(
        trace,
        calibration,
        envelope_config,
        &mut predictor,
        Some(envelopes),
    )?;
    let baseline_bytes = baseline.summary.bytes_device_read;
    let envelope_bytes = envelope_run.summary.bytes_device_read;
    let utility = EnvelopeUtility {
        baseline_bytes,
        envelope_bytes,
        bytes_not_transferred: i64::try_from(baseline_bytes).unwrap_or(i64::MAX)
            - i64::try_from(envelope_bytes).unwrap_or(i64::MAX),
        baseline_stall_p95_ns: baseline.summary.stall.p95_ns,
        envelope_stall_p95_ns: envelope_run.summary.stall.p95_ns,
        exposed_time_saved_ns: i64::try_from(baseline.summary.stall.p95_ns).unwrap_or(i64::MAX)
            - i64::try_from(envelope_run.summary.stall.p95_ns).unwrap_or(i64::MAX),
        unused_prefetch_bytes: envelope_run.summary.bytes_prefetch_wasted,
        cold_bytes_charged: envelope_run.summary.bytes_phase_cold,
        invalidations: envelope_run.invalidations,
        uncovered_phases: envelopes.uncovered_phases.clone(),
        interpretation: "utility is bytes not transferred and exposed p95 time saved \
                          against the same workload with no envelope; a negative value means \
                          the envelope cost more than it saved"
            .to_owned(),
    };
    Ok((baseline, envelope_run, utility))
}
