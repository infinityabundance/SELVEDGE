//! `selvedge trace`, `selvedge simulate`, `selvedge envelope` and
//! `selvedge sim`: the trace-driven execution and deadline simulator.
//!
//! Every command writes a typed receipt when `--out` is given and prints a
//! compact summary otherwise. Invalid input exits non-zero with a typed error;
//! no command guesses a default that would change a reported number.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Args, Subcommand};
use selvedge_core::{SelectionMode, Sha256Digest};
use selvedge_format::BundleFile;
use selvedge_sim::SimError;
use selvedge_sim::calibration::{
    Calibration, CalibrationInput, ReceiptRef, calibrate, read_metrics,
};
use selvedge_sim::compare::{arm_capacity, compare_with_store, run_pareto, validate_metrics};
use selvedge_sim::engine::{BackpressureMode, SimConfig, TwoTimescaleConfig, WorkingSetSource};
use selvedge_sim::envelope::{self, EnvelopeParams, EnvelopeSet};
use selvedge_sim::placement::{self, CoAccessWeights, PlacementKind, PlacementParams};
use selvedge_sim::policy::{
    EnvelopePrefetch, LastValuePrefetch, NoPrefetch, OracleNextRecordPrefetch, Predictor,
};
use selvedge_sim::report::SimReport;
use selvedge_sim::synthetic::{self, SyntheticSpec};
use selvedge_sim::trace::{
    ExactBasis, ExactBasisKind, ObservedStep, TRACE_MANIFEST_SCHEMA, TRACE_SCHEMA, Trace,
    TraceBundle, TraceLimits, TraceManifest, TraceProvenance, TraceRecord, TraceShard, TraceSource,
    TraceTimestamps,
};
use selvedge_store::PolicyKind;
use serde::Serialize;

use crate::CliResult;

/// Arguments for `selvedge trace`.
#[derive(Debug, Args)]
pub(crate) struct TraceArgs {
    #[command(subcommand)]
    pub(crate) command: TraceCommand,
}

/// `selvedge trace` subcommands.
#[derive(Debug, Subcommand)]
pub(crate) enum TraceCommand {
    /// Export a trace from a real bundle, a measured receipt or a fixture.
    Export(TraceExportArgs),
    /// Validate a trace, print its facts and check the record digest.
    Inspect(TraceInspectArgs),
    /// Generate a controlled synthetic trace.
    Generate(TraceGenerateArgs),
}

#[derive(Debug, Args)]
pub(crate) struct TraceExportArgs {
    /// Validated bundle the shard ids belong to.
    #[arg(long)]
    pub(crate) bundle: PathBuf,
    /// Output JSONL path; the sidecar is written next to it.
    #[arg(long)]
    pub(crate) out: PathBuf,
    /// Export the exact-mode whole-bundle pass.
    #[arg(long)]
    pub(crate) whole_bundle: bool,
    /// Reconstruct per-context selector decisions from the M2 fixtures.
    #[arg(long)]
    pub(crate) fixtures: Option<PathBuf>,
    /// Selector name inside the fixture (`magnitude`, `frequency`, ...).
    #[arg(long, default_value = "magnitude")]
    pub(crate) selector: String,
    /// Measured per-request metrics for this bundle, to attach observations.
    #[arg(long)]
    pub(crate) metrics: Option<PathBuf>,
    /// Which measured pass to export (`1` or `2`); both by default.
    #[arg(long)]
    pub(crate) metrics_pass: Option<u32>,
    /// Receipt the trace is derived from, recorded in the provenance.
    #[arg(long)]
    pub(crate) receipt: Option<PathBuf>,
    /// Stamp recorded in the manifest.
    #[arg(long, default_value = "19700101T000000Z")]
    pub(crate) stamp: String,
    /// Revision recorded in the manifest.
    #[arg(long, default_value = "unrecorded")]
    pub(crate) revision: String,
    /// Print the manifest.
    #[arg(long)]
    pub(crate) json: bool,
}

#[derive(Debug, Args)]
pub(crate) struct TraceInspectArgs {
    /// Trace JSONL path.
    #[arg(long)]
    pub(crate) trace: PathBuf,
    /// Sidecar manifest path; defaults next to the JSONL file.
    #[arg(long)]
    pub(crate) manifest: Option<PathBuf>,
    /// Print the full JSON summary.
    #[arg(long)]
    pub(crate) json: bool,
}

#[derive(Debug, Args)]
pub(crate) struct TraceGenerateArgs {
    /// Preset: `default`, `hot`, `cold` or `alternating`.
    #[arg(long, default_value = "default")]
    pub(crate) preset: String,
    /// Specification JSON, overriding the preset.
    #[arg(long)]
    pub(crate) spec: Option<PathBuf>,
    /// Seed override.
    #[arg(long)]
    pub(crate) seed: Option<u64>,
    /// Output JSONL path.
    #[arg(long)]
    pub(crate) out: PathBuf,
    /// Stamp recorded in the manifest.
    #[arg(long, default_value = "19700101T000000Z")]
    pub(crate) stamp: String,
    /// Revision recorded in the manifest.
    #[arg(long, default_value = "unrecorded")]
    pub(crate) revision: String,
    /// Print the manifest.
    #[arg(long)]
    pub(crate) json: bool,
}

/// Arguments for `selvedge simulate`.
#[derive(Debug, Args)]
pub(crate) struct SimulateArgs {
    /// Trace JSONL path.
    #[arg(long)]
    pub(crate) trace: PathBuf,
    /// Sidecar manifest path; defaults next to the JSONL file.
    #[arg(long)]
    pub(crate) manifest: Option<PathBuf>,
    /// Calibration receipt.
    #[arg(long)]
    pub(crate) calibration: PathBuf,
    /// Cache policy: none, lru, lfu, least_stale, pinned.
    #[arg(long, default_value = "none")]
    pub(crate) policy: String,
    /// Cache capacity in bytes.
    #[arg(long, default_value_t = 0)]
    pub(crate) cache_bytes: u64,
    /// Cache capacity as a fraction of the trace's unique payload.
    #[arg(long)]
    pub(crate) cache_fraction: Option<f64>,
    /// Concurrent device servers.
    #[arg(long, default_value_t = 1)]
    pub(crate) queue_depth: u32,
    /// In-flight byte budget.
    #[arg(long, default_value_t = 512 * 1024 * 1024)]
    pub(crate) max_in_flight_bytes: u64,
    /// Gap bridged when coalescing adjacent extents.
    #[arg(long, default_value_t = 0)]
    pub(crate) coalesce_gap_bytes: u64,
    /// Largest coalesced run.
    #[arg(long, default_value_t = 32 * 1024 * 1024)]
    pub(crate) max_run_bytes: u64,
    /// Charge SHA-256 verification.
    #[arg(long, default_value_t = true)]
    pub(crate) verify: bool,
    /// Compute time per record available for overlap.
    #[arg(long, default_value_t = 0)]
    pub(crate) compute_ns: u64,
    /// Per-record latency budget; a larger stall is a deadline miss.
    #[arg(long)]
    pub(crate) deadline_ns: Option<u64>,
    /// Seed.
    #[arg(long, default_value_t = 0x5E1D_E6E5_11ED_0001)]
    pub(crate) seed: u64,
    /// Backpressure: wait or fail.
    #[arg(long, default_value = "wait")]
    pub(crate) backpressure: String,
    /// Predictor: none, last-value, envelope, oracle-next.
    #[arg(long, default_value = "none")]
    pub(crate) prefetch: String,
    /// Placement: sequential, frequency-sorted, co-access-blocked,
    /// replicated-hot.
    #[arg(long)]
    pub(crate) placement: Option<String>,
    /// Envelope set to use (required for `--prefetch envelope`).
    #[arg(long)]
    pub(crate) envelopes: Option<PathBuf>,
    /// Two-timescale working set: none, history or envelope.
    #[arg(long, default_value = "none")]
    pub(crate) two_timescale: String,
    /// Fraction of the cache the working set may use.
    #[arg(long, default_value_t = 0.5)]
    pub(crate) working_set_fraction: f64,
    /// Write the report JSON here.
    #[arg(long)]
    pub(crate) out: Option<PathBuf>,
    /// Stamp recorded in the report.
    #[arg(long, default_value = "19700101T000000Z")]
    pub(crate) stamp: String,
    /// Revision recorded in the report.
    #[arg(long, default_value = "unrecorded")]
    pub(crate) revision: String,
    /// Print the report JSON.
    #[arg(long)]
    pub(crate) json: bool,
}

/// Arguments for `selvedge envelope`.
#[derive(Debug, Args)]
pub(crate) struct EnvelopeArgs {
    #[command(subcommand)]
    pub(crate) command: EnvelopeCommand,
}

/// `selvedge envelope` subcommands.
#[derive(Debug, Subcommand)]
pub(crate) enum EnvelopeCommand {
    /// Build candidate supersets from the trace prefix.
    Build(EnvelopeBuildArgs),
    /// Apply an envelope set and report bytes and time against a baseline.
    Apply(EnvelopeApplyArgs),
}

#[derive(Debug, Args)]
pub(crate) struct EnvelopeBuildArgs {
    /// Trace JSONL path.
    #[arg(long)]
    pub(crate) trace: PathBuf,
    /// Output envelope JSON.
    #[arg(long)]
    pub(crate) out: PathBuf,
    /// Fraction of the trace used as the build prefix.
    #[arg(long, default_value_t = 0.5)]
    pub(crate) build_fraction: f64,
    /// Minimum per-shard coverage inside a build phase.
    #[arg(long, default_value_t = 0.6)]
    pub(crate) min_coverage: f64,
    /// Candidate cap per envelope.
    #[arg(long, default_value_t = 4096)]
    pub(crate) max_candidates: usize,
    /// Stamp.
    #[arg(long, default_value = "19700101T000000Z")]
    pub(crate) stamp: String,
    /// Revision.
    #[arg(long, default_value = "unrecorded")]
    pub(crate) revision: String,
    /// Print the set.
    #[arg(long)]
    pub(crate) json: bool,
}

#[derive(Debug, Args)]
pub(crate) struct EnvelopeApplyArgs {
    /// Trace JSONL path.
    #[arg(long)]
    pub(crate) trace: PathBuf,
    /// Envelope JSON.
    #[arg(long)]
    pub(crate) envelopes: PathBuf,
    /// Calibration receipt.
    #[arg(long)]
    pub(crate) calibration: PathBuf,
    /// Cache policy.
    #[arg(long, default_value = "lru")]
    pub(crate) policy: String,
    /// Cache capacity in bytes.
    #[arg(long, default_value_t = 0)]
    pub(crate) cache_bytes: u64,
    /// Cache capacity as a fraction of the trace's unique payload.
    #[arg(long)]
    pub(crate) cache_fraction: Option<f64>,
    /// Compute time per record available for overlap.
    #[arg(long, default_value_t = 0)]
    pub(crate) compute_ns: u64,
    /// Seed.
    #[arg(long, default_value_t = 0x5E1D_E6E5_11ED_0001)]
    pub(crate) seed: u64,
    /// Output JSON.
    #[arg(long)]
    pub(crate) out: Option<PathBuf>,
    /// Stamp.
    #[arg(long, default_value = "19700101T000000Z")]
    pub(crate) stamp: String,
    /// Revision.
    #[arg(long, default_value = "unrecorded")]
    pub(crate) revision: String,
    /// Print JSON.
    #[arg(long)]
    pub(crate) json: bool,
}

/// Arguments for `selvedge sim`.
#[derive(Debug, Args)]
pub(crate) struct SimArgs {
    #[command(subcommand)]
    pub(crate) command: SimCommand,
}

/// `selvedge sim` subcommands.
#[derive(Debug, Subcommand)]
pub(crate) enum SimCommand {
    /// Fit the transfer model to measured receipts.
    Calibrate(CalibrateArgs),
    /// Replay a measured metrics file and compare predicted stall and bytes.
    Validate(ValidateArgs),
    /// Run the real store and the simulator over one trace and compare.
    CompareStore(CompareStoreArgs),
    /// Policies at capacity fractions: bytes per token and p95.
    Pareto(ParetoArgs),
    /// Compare physical layouts on one trace.
    Placement(PlacementArgs),
}

#[derive(Debug, Args)]
pub(crate) struct CalibrateArgs {
    /// Metrics JSONL files, each labelled `<role>=<path>`.
    #[arg(long = "metrics", value_name = "ROLE=PATH")]
    pub(crate) metrics: Vec<String>,
    /// io-scan receipt supplying the size law.
    #[arg(long)]
    pub(crate) io: PathBuf,
    /// Output calibration JSON.
    #[arg(long)]
    pub(crate) out: PathBuf,
    /// Stamp.
    #[arg(long, default_value = "19700101T000000Z")]
    pub(crate) stamp: String,
    /// Revision.
    #[arg(long, default_value = "unrecorded")]
    pub(crate) revision: String,
    /// Print the calibration.
    #[arg(long)]
    pub(crate) json: bool,
}

#[derive(Debug, Args)]
pub(crate) struct ValidateArgs {
    /// Calibration receipt.
    #[arg(long)]
    pub(crate) calibration: PathBuf,
    /// Target metrics JSONL.
    #[arg(long)]
    pub(crate) metrics: PathBuf,
    /// Arm name, for the receipt.
    #[arg(long)]
    pub(crate) arm: String,
    /// Store-bench receipt to read the arm's cache capacity from.
    #[arg(long)]
    pub(crate) bench: Option<PathBuf>,
    /// Cache capacity in bytes, when no bench receipt is given.
    #[arg(long, default_value_t = 0)]
    pub(crate) cache_bytes: u64,
    /// Policy to replay.
    #[arg(long, default_value = "lru")]
    pub(crate) policy: String,
    /// Seed.
    #[arg(long, default_value_t = 0x5E1D_E6E5_11ED_0001)]
    pub(crate) seed: u64,
    /// Output validation JSON.
    #[arg(long)]
    pub(crate) out: Option<PathBuf>,
    /// Stamp.
    #[arg(long, default_value = "19700101T000000Z")]
    pub(crate) stamp: String,
    /// Revision.
    #[arg(long, default_value = "unrecorded")]
    pub(crate) revision: String,
    /// Print JSON.
    #[arg(long)]
    pub(crate) json: bool,
}

#[derive(Debug, Args)]
pub(crate) struct CompareStoreArgs {
    /// Trace JSONL path.
    #[arg(long)]
    pub(crate) trace: PathBuf,
    /// Bundle to read.
    #[arg(long)]
    pub(crate) bundle: PathBuf,
    /// Calibration receipt.
    #[arg(long)]
    pub(crate) calibration: PathBuf,
    /// Cache policy.
    #[arg(long, default_value = "none")]
    pub(crate) policy: String,
    /// Cache capacity in bytes.
    #[arg(long, default_value_t = 0)]
    pub(crate) cache_bytes: u64,
    /// Seed.
    #[arg(long, default_value_t = 0x5E1D_E6E5_11ED_0001)]
    pub(crate) seed: u64,
    /// Output comparison JSON.
    #[arg(long)]
    pub(crate) out: PathBuf,
    /// Stamp.
    #[arg(long, default_value = "19700101T000000Z")]
    pub(crate) stamp: String,
    /// Revision.
    #[arg(long, default_value = "unrecorded")]
    pub(crate) revision: String,
    /// Print JSON.
    #[arg(long)]
    pub(crate) json: bool,
}

#[derive(Debug, Args)]
pub(crate) struct ParetoArgs {
    /// Trace JSONL path.
    #[arg(long)]
    pub(crate) trace: PathBuf,
    /// Calibration receipt.
    #[arg(long)]
    pub(crate) calibration: PathBuf,
    /// Policies, comma separated.
    #[arg(long, value_delimiter = ',', default_value = "lru")]
    pub(crate) policies: Vec<String>,
    /// Capacity fractions, comma separated.
    #[arg(long, value_delimiter = ',', default_value = "0.125,0.25,0.5,1.0")]
    pub(crate) fractions: Vec<f64>,
    /// Queue depth.
    #[arg(long, default_value_t = 1)]
    pub(crate) queue_depth: u32,
    /// Compute time per record.
    #[arg(long, default_value_t = 0)]
    pub(crate) compute_ns: u64,
    /// Seed.
    #[arg(long, default_value_t = 0x5E1D_E6E5_11ED_0001)]
    pub(crate) seed: u64,
    /// Output Pareto JSON.
    #[arg(long)]
    pub(crate) out: PathBuf,
    /// Stamp.
    #[arg(long, default_value = "19700101T000000Z")]
    pub(crate) stamp: String,
    /// Revision.
    #[arg(long, default_value = "unrecorded")]
    pub(crate) revision: String,
    /// Print JSON.
    #[arg(long)]
    pub(crate) json: bool,
}

#[derive(Debug, Args)]
pub(crate) struct PlacementArgs {
    /// Trace JSONL path.
    #[arg(long)]
    pub(crate) trace: PathBuf,
    /// Calibration receipt.
    #[arg(long)]
    pub(crate) calibration: PathBuf,
    /// Layouts, comma separated.
    #[arg(
        long,
        value_delimiter = ',',
        default_value = "sequential,frequency-sorted,co-access-blocked,replicated-hot"
    )]
    pub(crate) kinds: Vec<String>,
    /// Cache policy.
    #[arg(long, default_value = "lru")]
    pub(crate) policy: String,
    /// Cache capacity as a fraction of the unique payload.
    #[arg(long, default_value_t = 0.25)]
    pub(crate) cache_fraction: f64,
    /// Gap bridged when coalescing.
    #[arg(long, default_value_t = 0)]
    pub(crate) coalesce_gap_bytes: u64,
    /// Byte penalty in the surrogate objective, per byte.
    #[arg(long, default_value_t = 0.0)]
    pub(crate) lambda_bytes: f64,
    /// Eviction penalty in the surrogate objective, per replicated byte.
    #[arg(long, default_value_t = 0.0)]
    pub(crate) lambda_eviction: f64,
    /// Shards per co-access block.
    #[arg(long, default_value_t = 16)]
    pub(crate) block_shards: usize,
    /// Hot shards eligible for replication.
    #[arg(long, default_value_t = 64)]
    pub(crate) hot_shards: usize,
    /// Seed.
    #[arg(long, default_value_t = 0x5E1D_E6E5_11ED_0001)]
    pub(crate) seed: u64,
    /// Output JSON.
    #[arg(long)]
    pub(crate) out: PathBuf,
    /// Stamp.
    #[arg(long, default_value = "19700101T000000Z")]
    pub(crate) stamp: String,
    /// Revision.
    #[arg(long, default_value = "unrecorded")]
    pub(crate) revision: String,
    /// Print JSON.
    #[arg(long)]
    pub(crate) json: bool,
}

fn parse_policy(name: &str) -> Result<PolicyKind, SimError> {
    match name {
        "none" => Ok(PolicyKind::None),
        "lru" => Ok(PolicyKind::Lru),
        "lfu" => Ok(PolicyKind::Lfu),
        "least_stale" => Ok(PolicyKind::LeastStale),
        "pinned" => Ok(PolicyKind::Pinned),
        other => Err(SimError::invalid(
            "policy",
            format!("unknown policy {other}"),
        )),
    }
}

fn load_trace(trace: &std::path::Path, manifest: Option<&PathBuf>) -> Result<Trace, SimError> {
    let manifest = manifest
        .cloned()
        .unwrap_or_else(|| Trace::manifest_path_for(trace));
    Trace::load(trace, &manifest, TraceLimits::default())
}

fn load_calibration(path: &std::path::Path) -> Result<Calibration, SimError> {
    Calibration::load(path)
}

fn load_envelopes(path: &std::path::Path) -> Result<EnvelopeSet, SimError> {
    EnvelopeSet::load(path)
}

fn unique_payload(trace: &Trace) -> u64 {
    trace
        .shards()
        .iter()
        .map(|shard| shard.bytes)
        .fold(0_u64, u64::saturating_add)
}

fn capacity(trace: &Trace, bytes: u64, fraction: Option<f64>) -> Result<u64, SimError> {
    match fraction {
        Some(fraction) => {
            if !(0.0..=1.0).contains(&fraction) || fraction <= 0.0 {
                return Err(SimError::invalid("cache_fraction", "must be in (0, 1]"));
            }
            Ok((unique_payload(trace) as f64 * fraction) as u64)
        }
        None => Ok(bytes),
    }
}

fn build_config(
    trace: &Trace,
    args: &SimulateArgs,
    policy: PolicyKind,
    cache_bytes: u64,
) -> Result<(SimConfig, Box<dyn Predictor>), SimError> {
    let backpressure = match args.backpressure.as_str() {
        "wait" => BackpressureMode::Wait,
        "fail" => BackpressureMode::Fail,
        other => {
            return Err(SimError::invalid(
                "backpressure",
                format!("unknown mode {other}"),
            ));
        }
    };
    let two_timescale = match args.two_timescale.as_str() {
        "none" => None,
        "history" => Some(TwoTimescaleConfig {
            source: WorkingSetSource::History,
            capacity_fraction: args.working_set_fraction,
            min_usage: 1,
        }),
        "envelope" => Some(TwoTimescaleConfig {
            source: WorkingSetSource::Envelope,
            capacity_fraction: args.working_set_fraction,
            min_usage: 1,
        }),
        other => {
            return Err(SimError::invalid(
                "two_timescale",
                format!("unknown source {other}"),
            ));
        }
    };
    let predictor: Box<dyn Predictor> = match args.prefetch.as_str() {
        "none" => Box::new(NoPrefetch),
        "last-value" => Box::new(LastValuePrefetch),
        "envelope" => Box::new(EnvelopePrefetch),
        "oracle-next" => Box::new(OracleNextRecordPrefetch),
        other => {
            return Err(SimError::invalid(
                "prefetch",
                format!("unknown predictor {other}"),
            ));
        }
    };
    let placement = match &args.placement {
        Some(name) => {
            let kind = PlacementKind::parse(name)?;
            let params = PlacementParams::default();
            let weights = placement::weights(
                trace,
                Some(&load_calibration(&args.calibration)?.size_law),
                params,
                args.seed,
            )?;
            Some(placement::build(trace, &weights, kind, params)?)
        }
        None => None,
    };
    Ok((
        SimConfig {
            policy,
            cache_bytes,
            queue_depth: args.queue_depth,
            max_in_flight_bytes: args.max_in_flight_bytes,
            coalesce_gap_bytes: args.coalesce_gap_bytes,
            max_run_bytes: args.max_run_bytes,
            verify: args.verify,
            compute_ns_per_record: args.compute_ns,
            deadline_ns: args.deadline_ns,
            seed: args.seed,
            backpressure,
            placement,
            two_timescale,
            cancel_prefetch_on_phase_change: true,
            preempt_prefetch_for_required: true,
            start_record: 0,
            max_prefetch_shards: 4096,
            inject_checksum_failure: None,
        },
        predictor,
    ))
}

/// Run `selvedge trace`.
pub(crate) fn run_trace(args: &TraceArgs) -> CliResult {
    match &args.command {
        TraceCommand::Export(args) => run_trace_export(args),
        TraceCommand::Inspect(args) => run_trace_inspect(args),
        TraceCommand::Generate(args) => run_trace_generate(args),
    }
}

fn run_trace_export(args: &TraceExportArgs) -> CliResult {
    let bundle = BundleFile::open(&args.bundle)?;
    let manifest = bundle.manifest();
    let shards: Vec<TraceShard> = manifest
        .shards
        .iter()
        .map(|shard| TraceShard {
            id: shard.shard_id,
            bytes: shard.length,
            offset: shard.offset,
        })
        .collect();
    let mode = if args.whole_bundle || args.metrics.is_some() {
        SelectionMode::Exact
    } else {
        SelectionMode::Approximate
    };
    let ids: Vec<_> = shards.iter().map(|shard| shard.id).collect();
    let exact_basis = if mode == SelectionMode::Exact {
        Some(ExactBasis {
            kind: ExactBasisKind::WholeBundlePlanRead,
            native_set_digest: selvedge_core::native_set_digest(&manifest.bundle_id, &ids)?,
            shard_count: ids.len() as u64,
        })
    } else {
        None
    };
    let mut records = Vec::new();
    let source = if args.metrics.is_some() || args.receipt.is_some() {
        TraceSource::Measured
    } else {
        TraceSource::Reconstructed
    };
    if let Some(metrics_path) = &args.metrics {
        let sequence = read_metrics(metrics_path, TraceLimits::default())?;
        let shard_count = manifest.shards.len();
        if shard_count == 0 {
            return Err(SimError::invalid("bundle", "has no shards").into());
        }
        let passes: Vec<&[selvedge_sim::calibration::MetricsRecord]> =
            sequence.reads.chunks(shard_count).collect();
        for (index, pass) in passes.iter().enumerate() {
            let pass_number = index as u32 + 1;
            if let Some(wanted) = args.metrics_pass
                && wanted != pass_number
            {
                continue;
            }
            if pass.len() != shard_count {
                return Err(SimError::invalid(
                    "metrics",
                    format!(
                        "pass {pass_number} has {} reads, bundle has {shard_count} shards",
                        pass.len()
                    ),
                )
                .into());
            }
            let stall: u64 = pass
                .iter()
                .map(|read| read.stall_ns)
                .fold(0, u64::saturating_add);
            let device: u64 = pass
                .iter()
                .map(|read| read.device_bytes)
                .fold(0, u64::saturating_add);
            let used: u64 = pass
                .iter()
                .map(|read| read.bytes_used)
                .fold(0, u64::saturating_add);
            let hits = pass.iter().filter(|read| read.hit).count() as u64;
            let misses = pass.len() as u64 - hits;
            records.push(TraceRecord {
                schema: TRACE_SCHEMA.to_owned(),
                request_id: format!("pass-{pass_number}"),
                phase_id: "whole-bundle".to_owned(),
                token_index: (pass_number - 1) as u64,
                layer: 0,
                selected: ids.clone(),
                selector_id: "native-dense".to_owned(),
                mode,
                task: None,
                timestamps: Some(TraceTimestamps {
                    decision_mono_ns: pass.first().map_or(0, |read| read.mono_ns),
                    complete_mono_ns: pass.last().map(|read| read.mono_ns),
                }),
                observed: Some(ObservedStep {
                    stall_ns: stall,
                    bytes_transferred: device,
                    bytes_used: used,
                    hits,
                    misses,
                }),
            });
        }
    } else if let Some(fixture_path) = &args.fixtures {
        export_fixture_records(fixture_path, manifest, &args.selector, mode, &mut records)?;
    } else if args.whole_bundle {
        records.push(TraceRecord {
            schema: TRACE_SCHEMA.to_owned(),
            request_id: "pass-1".to_owned(),
            phase_id: "whole-bundle".to_owned(),
            token_index: 0,
            layer: 0,
            selected: ids.clone(),
            selector_id: "native-dense".to_owned(),
            mode,
            task: None,
            timestamps: None,
            observed: None,
        });
    } else {
        return Err(
            SimError::invalid("export", "choose --whole-bundle, --metrics or --fixtures").into(),
        );
    }
    let receipt = match &args.receipt {
        Some(path) => Some(ReceiptRef::of(path, "trace_source")?),
        None => None,
    };
    let provenance = TraceProvenance {
        origin: format!("selvedge trace export --bundle {}", args.bundle.display()),
        receipt: receipt.as_ref().map(|reference| reference.path.clone()),
        receipt_sha256: receipt.as_ref().map(|reference| reference.sha256),
        revision: args.revision.clone(),
        stamp: args.stamp.clone(),
        notes: vec![
            "shard ids are the validated bundle's own ids; the trace never restates them"
                .to_owned(),
            "a trace records what was selected, never what should have been selected".to_owned(),
        ],
    };
    let trace_manifest = TraceManifest {
        schema: TRACE_MANIFEST_SCHEMA.to_owned(),
        trace_id: format!("{}-{}", manifest.bundle_id, args.selector),
        source,
        provenance,
        bundle: TraceBundle {
            bundle_id: manifest.bundle_id.clone(),
            manifest_sha256: bundle.manifest_sha256(),
        },
        mode,
        exact_basis,
        shards,
        records: 0,
        records_sha256: Sha256Digest::of(b""),
        generator: None,
        created_utc: args.stamp.clone(),
    };
    let trace = Trace::new(trace_manifest, records)?;
    trace.check_label()?;
    let manifest_path = Trace::manifest_path_for(&args.out);
    trace.write(&args.out, &manifest_path)?;
    println!(
        "trace: {} records, {} shards, source={}, mode={}",
        trace.records.len(),
        trace.shards().len(),
        trace.manifest.source.name(),
        trace.manifest.mode
    );
    println!("records: {}", args.out.display());
    println!("manifest: {}", manifest_path.display());
    println!("records_sha256: {}", trace.manifest.records_sha256);
    if args.json {
        println!("{}", serde_json::to_string_pretty(&trace.manifest)?);
    }
    Ok(ExitCode::SUCCESS)
}

fn export_fixture_records(
    fixture_path: &std::path::Path,
    manifest: &selvedge_format::Manifest,
    selector: &str,
    mode: SelectionMode,
    records: &mut Vec<TraceRecord>,
) -> Result<(), SimError> {
    #[derive(serde::Deserialize)]
    struct Fixtures {
        records: Vec<FixtureRecord>,
    }
    #[derive(serde::Deserialize)]
    struct FixtureRecord {
        expected: BTreeMap<String, Vec<u32>>,
        layer: u32,
        position: u64,
        sequence_key: String,
    }
    // Group index -> shard ids, per layer, from the bundle's own channel ranges.
    let mut groups: BTreeMap<(u32, u64), Vec<selvedge_core::ShardId>> = BTreeMap::new();
    for shard in &manifest.shards {
        let layer = match shard.layer {
            Some(layer) => layer,
            None => continue,
        };
        let selvedge_core::ChannelSet::Range { start, .. } = &shard.channels else {
            return Err(SimError::invalid(
                "fixtures",
                "grouped bundle should use channel ranges",
            ));
        };
        groups
            .entry((layer, *start))
            .or_default()
            .push(shard.shard_id);
    }
    let bytes = std::fs::read(fixture_path)?;
    let fixtures: Fixtures = serde_json::from_slice(&bytes)?;
    for (index, record) in fixtures.records.iter().enumerate() {
        let decisions = match record.expected.get(selector) {
            Some(decisions) => decisions,
            None => {
                return Err(SimError::invalid(
                    "selector",
                    format!("fixture has no selector {selector}"),
                ));
            }
        };
        let mut selected: Vec<selvedge_core::ShardId> = Vec::new();
        for group in decisions {
            let key = (
                record.layer,
                u64::from(*group) * group_width(manifest, record.layer),
            );
            let shards = groups.get(&key).ok_or_else(|| {
                SimError::invalid(
                    "fixtures",
                    format!("layer {} has no group starting at {}", record.layer, key.1),
                )
            })?;
            selected.extend(shards.iter().copied());
        }
        selected.sort_unstable();
        selected.dedup();
        records.push(TraceRecord {
            schema: TRACE_SCHEMA.to_owned(),
            request_id: record.sequence_key.clone(),
            phase_id: "m2-context".to_owned(),
            token_index: record.position,
            layer: record.layer,
            selected,
            selector_id: format!("m2-{selector}"),
            mode,
            task: None,
            timestamps: None,
            observed: None,
        });
        let _ = index;
    }
    Ok(())
}

fn group_width(manifest: &selvedge_format::Manifest, layer: u32) -> u64 {
    // Half-open channel ranges are contiguous and equal-sized per layer.
    let mut starts: Vec<u64> = manifest
        .shards
        .iter()
        .filter(|shard| shard.layer == Some(layer))
        .filter_map(|shard| match &shard.channels {
            selvedge_core::ChannelSet::Range { start, .. } => Some(*start),
            selvedge_core::ChannelSet::List { .. } => None,
        })
        .collect();
    starts.sort_unstable();
    starts.dedup();
    starts.windows(2).next().map_or(0, |pair| pair[1] - pair[0])
}

fn run_trace_inspect(args: &TraceInspectArgs) -> CliResult {
    let trace = load_trace(&args.trace, args.manifest.as_ref())?;
    let phases = trace.phases();
    let tasks = trace.tasks();
    let payload: u64 = unique_payload(&trace);
    let mut summary = serde_json::json!({
        "schema": trace.manifest.schema,
        "trace_id": trace.manifest.trace_id,
        "source": trace.manifest.source.name(),
        "mode": trace.manifest.mode.to_string(),
        "records": trace.records.len(),
        "shards": trace.shards().len(),
        "unique_payload_bytes": payload,
        "records_sha256": trace.manifest.records_sha256.to_hex(),
        "phases": phases.iter().map(|(phase, indices)| serde_json::json!({
            "phase_id": phase,
            "records": indices.len(),
        })).collect::<Vec<_>>(),
        "tasks": tasks,
        "has_observed_steps": trace.records.iter().any(|record| record.observed.is_some()),
    });
    println!("trace id: {}", trace.manifest.trace_id);
    println!(
        "source={} mode={} records={} shards={} payload={} B",
        trace.manifest.source.name(),
        trace.manifest.mode,
        trace.records.len(),
        trace.shards().len(),
        payload
    );
    println!(
        "records_sha256: {} (verified)",
        trace.manifest.records_sha256
    );
    for (phase, indices) in &phases {
        println!("phase {phase}: {} records", indices.len());
    }
    if args.json {
        if let Some(object) = summary.as_object_mut() {
            object.insert(
                "manifest".to_owned(),
                serde_json::to_value(&trace.manifest)?,
            );
        }
        println!("{}", serde_json::to_string_pretty(&summary)?);
    }
    Ok(ExitCode::SUCCESS)
}

fn run_trace_generate(args: &TraceGenerateArgs) -> CliResult {
    let mut spec = match &args.spec {
        Some(path) => {
            let bytes = std::fs::read(path)?;
            serde_json::from_slice::<SyntheticSpec>(&bytes)?
        }
        None => match args.preset.as_str() {
            "default" => SyntheticSpec::default(),
            "hot" => SyntheticSpec::hot(),
            "cold" => SyntheticSpec::cold(),
            "alternating" => SyntheticSpec::alternating(),
            other => {
                return Err(SimError::invalid("preset", format!("unknown preset {other}")).into());
            }
        },
    };
    if let Some(seed) = args.seed {
        spec.seed = seed;
    }
    let generated = synthetic::generate(spec, &args.revision, &args.stamp)?;
    let manifest_path = Trace::manifest_path_for(&args.out);
    generated.trace.write(&args.out, &manifest_path)?;
    println!(
        "synthetic trace: {} records, {} shards, hash {}",
        generated.trace.records.len(),
        generated.trace.shards().len(),
        generated.trace.manifest.records_sha256
    );
    println!("records: {}", args.out.display());
    println!("manifest: {}", manifest_path.display());
    if args.json {
        println!("{}", serde_json::to_string_pretty(&generated.spec)?);
    }
    Ok(ExitCode::SUCCESS)
}

/// Run `selvedge simulate`.
pub(crate) fn run_simulate(args: &SimulateArgs) -> CliResult {
    let trace = load_trace(&args.trace, args.manifest.as_ref())?;
    let calibration = load_calibration(&args.calibration)?;
    let policy = parse_policy(&args.policy)?;
    let cache_bytes = capacity(&trace, args.cache_bytes, args.cache_fraction)?;
    let envelopes = match &args.envelopes {
        Some(path) => Some(load_envelopes(path)?),
        None => None,
    };
    if args.prefetch == "envelope" && envelopes.is_none() {
        return Err(SimError::invalid("prefetch", "`envelope` needs --envelopes").into());
    }
    let (mut config, mut predictor) = build_config(&trace, args, policy, cache_bytes)?;
    if let Some(set) = &envelopes
        && config.two_timescale.is_some()
    {
        config.start_record = selvedge_sim::envelope::evaluation_start(set) as u64;
    }
    let weights = match &config.placement {
        Some(_) => Some(placement::weights(
            &trace,
            Some(&calibration.size_law),
            PlacementParams::default(),
            args.seed,
        )?),
        None => None,
    };
    let report = selvedge_sim::simulate_report(
        &trace,
        &calibration,
        config,
        predictor.as_mut(),
        envelopes.as_ref(),
        weights.as_ref(),
        0.0,
        0.0,
        &args.stamp,
        &args.revision,
        vec![
            "simulated, not measured: see the calibration receipt and the validation receipts"
                .to_owned(),
            "the selection is the trace's; a policy changes latency, bytes and residency only"
                .to_owned(),
        ],
    )?;
    report.check()?;
    if let Some(path) = &args.out {
        report.write(path)?;
        println!("report: {}", path.display());
    }
    print_summary(&report);
    if args.json {
        println!("{}", serde_json::to_string_pretty(&report)?);
    }
    Ok(ExitCode::SUCCESS)
}

fn print_summary(report: &SimReport) {
    let summary = &report.summary;
    println!(
        "records={} p50={}ns p95={}ns p99={}ns",
        summary.evaluated_records, summary.stall.p50_ns, summary.stall.p95_ns, summary.stall.p99_ns
    );
    println!(
        "bytes: needed={} hit={} fetched={} device={} wasted={} (prefetch {} used {} cancelled {})",
        summary.bytes_needed,
        summary.bytes_hit,
        summary.bytes_required_miss,
        summary.bytes_device_read,
        summary.bytes_wasted,
        summary.bytes_prefetch,
        summary.bytes_prefetch_used,
        summary.bytes_prefetch_cancelled
    );
    println!(
        "cache hits={} misses={} evictions={} deadline_misses={}",
        summary.hits, summary.misses, summary.evictions, summary.deadline_misses
    );
}

/// Run `selvedge envelope`.
pub(crate) fn run_envelope(args: &EnvelopeArgs) -> CliResult {
    match &args.command {
        EnvelopeCommand::Build(args) => run_envelope_build(args),
        EnvelopeCommand::Apply(args) => run_envelope_apply(args),
    }
}

fn run_envelope_build(args: &EnvelopeBuildArgs) -> CliResult {
    let trace = load_trace(&args.trace, None)?;
    let params = EnvelopeParams {
        build_fraction: args.build_fraction,
        min_coverage: args.min_coverage,
        max_candidates: args.max_candidates,
        ..EnvelopeParams::default()
    };
    let set = envelope::build(&trace, params, &args.revision, &args.stamp)?;
    set.write(&args.out)?;
    println!(
        "envelopes: {} phases from {} build records; evaluating {} records",
        set.envelopes.len(),
        set.build_records,
        set.evaluate_records
    );
    for envelope in &set.envelopes {
        println!(
            "phase {}: {} candidates, build coverage {:.3}",
            envelope.phase_id,
            envelope.candidate.len(),
            envelope.build_coverage
        );
    }
    if !set.uncovered_phases.is_empty() {
        println!("uncovered phases: {}", set.uncovered_phases.join(", "));
    }
    if args.json {
        println!("{}", serde_json::to_string_pretty(&set)?);
    }
    Ok(ExitCode::SUCCESS)
}

fn run_envelope_apply(args: &EnvelopeApplyArgs) -> CliResult {
    let trace = load_trace(&args.trace, None)?;
    let calibration = load_calibration(&args.calibration)?;
    let set = load_envelopes(&args.envelopes)?;
    let policy = parse_policy(&args.policy)?;
    let cache_bytes = capacity(&trace, args.cache_bytes, args.cache_fraction)?;
    let base = SimConfig {
        policy,
        cache_bytes,
        queue_depth: 1,
        max_in_flight_bytes: 512 * 1024 * 1024,
        coalesce_gap_bytes: 0,
        max_run_bytes: 32 * 1024 * 1024,
        verify: true,
        compute_ns_per_record: args.compute_ns,
        deadline_ns: None,
        seed: args.seed,
        backpressure: BackpressureMode::Wait,
        placement: None,
        two_timescale: None,
        cancel_prefetch_on_phase_change: true,
        preempt_prefetch_for_required: true,
        start_record: 0,
        max_prefetch_shards: 4096,
        inject_checksum_failure: None,
    };
    let (baseline, envelope_run, utility) =
        selvedge_sim::measure_envelope(&trace, &calibration, &base, &set)?;
    baseline.summary.check()?;
    envelope_run.summary.check()?;
    let payload = serde_json::json!({
        "schema": "selvedge/sim-envelope-utility/v1",
        "stamp": args.stamp,
        "revision": args.revision,
        "trace_id": trace.manifest.trace_id,
        "trace_sha256": trace.manifest.records_sha256.to_hex(),
        "utility": utility,
        "baseline_summary": baseline.summary,
        "envelope_summary": envelope_run.summary,
    });
    if let Some(path) = &args.out {
        std::fs::write(path, serde_json::to_vec_pretty(&payload)?)?;
        println!("receipt: {}", path.display());
    }
    println!(
        "bytes: baseline {} -> envelope {} (saved {}, unused prefetch {})",
        utility.baseline_bytes,
        utility.envelope_bytes,
        utility.bytes_not_transferred,
        utility.unused_prefetch_bytes
    );
    println!(
        "p95: baseline {} ns -> envelope {} ns (saved {} ns); cold bytes charged {}; invalidations {}",
        utility.baseline_stall_p95_ns,
        utility.envelope_stall_p95_ns,
        utility.exposed_time_saved_ns,
        utility.cold_bytes_charged,
        utility.invalidations
    );
    if args.json {
        println!("{}", serde_json::to_string_pretty(&payload)?);
    }
    Ok(ExitCode::SUCCESS)
}

/// Run `selvedge sim`.
pub(crate) fn run_sim(args: &SimArgs) -> CliResult {
    match &args.command {
        SimCommand::Calibrate(args) => run_calibrate(args),
        SimCommand::Validate(args) => run_validate(args),
        SimCommand::CompareStore(args) => run_compare_store(args),
        SimCommand::Pareto(args) => run_pareto_cmd(args),
        SimCommand::Placement(args) => run_placement(args),
    }
}

fn run_calibrate(args: &CalibrateArgs) -> CliResult {
    if args.metrics.is_empty() {
        return Err(SimError::invalid("metrics", "at least one file is required").into());
    }
    let mut inputs = Vec::new();
    for entry in &args.metrics {
        let (role, path) = entry
            .split_once('=')
            .ok_or_else(|| SimError::invalid("metrics", "expected <role>=<path>"))?;
        inputs.push(CalibrationInput {
            role: role.to_owned(),
            path: PathBuf::from(path),
        });
    }
    let calibration = calibrate(&inputs, &args.io, &args.revision, &args.stamp)?;
    calibration.write(&args.out)?;
    println!(
        "calibration: {} inputs, shard {} B, transfer p50 {} ns, verify p50 {} ns",
        calibration.inputs.len(),
        calibration.shard_bytes,
        calibration.transfer.p50,
        calibration.verify.p50
    );
    for fit in &calibration.fits {
        println!(
            "fit {} {}: max |rel err| {:.4}, mean {:.4}",
            fit.quantity, fit.family, fit.max_abs_rel_error, fit.mean_abs_rel_error
        );
    }
    println!("receipt: {}", args.out.display());
    if args.json {
        println!("{}", serde_json::to_string_pretty(&calibration)?);
    }
    Ok(ExitCode::SUCCESS)
}

fn run_validate(args: &ValidateArgs) -> CliResult {
    let calibration = load_calibration(&args.calibration)?;
    let policy = parse_policy(&args.policy)?;
    let capacity = match &args.bench {
        Some(bench) => arm_capacity(bench, &args.arm)?,
        None => args.cache_bytes,
    };
    let validation = validate_metrics(
        &calibration,
        &args.metrics,
        &args.arm,
        policy,
        capacity,
        args.seed,
        &args.revision,
        &args.stamp,
    )?;
    if let Some(path) = &args.out {
        std::fs::write(path, serde_json::to_vec_pretty(&validation)?)?;
        println!("receipt: {}", path.display());
    }
    println!(
        "arm {} policy {} capacity {}: reads {} hits recorded/replayed {}/{}, mismatches {}",
        validation.arm,
        validation.policy,
        validation.capacity_bytes,
        validation.reads,
        validation.hits_recorded,
        validation.hits_replayed,
        validation.hit_mismatches
    );
    println!(
        "stall measured p50/p95/p99 {}/{}/{} ns vs predicted {}/{}/{} ns",
        validation.measured_stall.p50_ns,
        validation.measured_stall.p95_ns,
        validation.measured_stall.p99_ns,
        validation.predicted_stall.p50_ns,
        validation.predicted_stall.p95_ns,
        validation.predicted_stall.p99_ns
    );
    for (name, value) in &validation.stall_discrepancy_pct {
        println!("discrepancy {name}: {value:+.1}%");
    }
    println!(
        "device bytes measured {} vs predicted {} ({:+.1}%)",
        validation.bytes.measured_device_bytes,
        validation.bytes.predicted_device_bytes,
        validation.bytes.discrepancy_pct
    );
    println!(
        "verdict: {} ({})",
        validation.verdict, validation.verdict_reason
    );
    if args.json {
        println!("{}", serde_json::to_string_pretty(&validation)?);
    }
    if validation.verdict != "pass" {
        return Ok(ExitCode::from(3));
    }
    Ok(ExitCode::SUCCESS)
}

fn run_compare_store(args: &CompareStoreArgs) -> CliResult {
    let trace = load_trace(&args.trace, None)?;
    let calibration = load_calibration(&args.calibration)?;
    let policy = parse_policy(&args.policy)?;
    let config = SimConfig {
        policy,
        cache_bytes: args.cache_bytes,
        queue_depth: 1,
        max_in_flight_bytes: 512 * 1024 * 1024,
        coalesce_gap_bytes: 0,
        max_run_bytes: 32 * 1024 * 1024,
        verify: true,
        compute_ns_per_record: 0,
        deadline_ns: None,
        seed: args.seed,
        backpressure: BackpressureMode::Wait,
        placement: None,
        two_timescale: None,
        cancel_prefetch_on_phase_change: true,
        preempt_prefetch_for_required: true,
        start_record: 0,
        max_prefetch_shards: 4096,
        inject_checksum_failure: None,
    };
    let comparison = compare_with_store(
        &trace,
        &args.bundle,
        &config,
        &calibration,
        &args.revision,
        &args.stamp,
    )?;
    std::fs::write(&args.out, serde_json::to_vec_pretty(&comparison)?)?;
    println!("receipt: {}", args.out.display());
    for (label, pass) in [("cold", &comparison.cold), ("warm", &comparison.warm)] {
        println!(
            "{label}: store p50/p95/p99 {}/{}/{} ns, sim {}/{}/{} ns; p50 {:+.1}% p95 {:+.1}% bytes {:+.1}% ({} )",
            pass.store.stall.p50_ns,
            pass.store.stall.p95_ns,
            pass.store.stall.p99_ns,
            pass.sim.stall.p50_ns,
            pass.sim.stall.p95_ns,
            pass.sim.stall.p99_ns,
            pass.p50_discrepancy_pct,
            pass.p95_discrepancy_pct,
            pass.bytes_discrepancy_pct,
            pass.verdict
        );
    }
    if args.json {
        println!("{}", serde_json::to_string_pretty(&comparison)?);
    }
    if comparison.cold.verdict != "pass" {
        return Ok(ExitCode::from(3));
    }
    Ok(ExitCode::SUCCESS)
}

fn run_pareto_cmd(args: &ParetoArgs) -> CliResult {
    let trace = load_trace(&args.trace, None)?;
    let calibration = load_calibration(&args.calibration)?;
    let mut policies = Vec::new();
    for name in &args.policies {
        policies.push(parse_policy(name)?);
    }
    let base = SimConfig {
        policy: PolicyKind::None,
        cache_bytes: 0,
        queue_depth: args.queue_depth,
        max_in_flight_bytes: 512 * 1024 * 1024,
        coalesce_gap_bytes: 0,
        max_run_bytes: 32 * 1024 * 1024,
        verify: true,
        compute_ns_per_record: args.compute_ns,
        deadline_ns: None,
        seed: args.seed,
        backpressure: BackpressureMode::Wait,
        placement: None,
        two_timescale: None,
        cancel_prefetch_on_phase_change: true,
        preempt_prefetch_for_required: true,
        start_record: 0,
        max_prefetch_shards: 4096,
        inject_checksum_failure: None,
    };
    let report = run_pareto(
        &trace,
        &calibration,
        &base,
        &policies,
        &args.fractions,
        &args.revision,
        &args.stamp,
    )?;
    std::fs::write(&args.out, serde_json::to_vec_pretty(&report)?)?;
    println!("receipt: {}", args.out.display());
    println!("| arm | bytes/record | p50 ns | p95 ns | hit rate |");
    for row in &report.rows {
        println!(
            "| {} | {:.1} | {} | {} | {:.3} |",
            row.arm, row.bytes_per_record, row.p50_stall_ns, row.p95_stall_ns, row.hit_rate
        );
    }
    if args.json {
        println!("{}", serde_json::to_string_pretty(&report)?);
    }
    Ok(ExitCode::SUCCESS)
}

#[derive(Debug, Serialize)]
struct PlacementRow {
    kind: PlacementKind,
    amplification: f64,
    placed_bytes: u64,
    replicated_bytes: u64,
    p50_stall_ns: u64,
    p95_stall_ns: u64,
    p99_stall_ns: u64,
    bytes_device_read: u64,
    bytes_per_record: f64,
    unused_prefetch_bytes: u64,
    estimated_stall_saved_ns: f64,
    objective: f64,
}

fn run_placement(args: &PlacementArgs) -> CliResult {
    let trace = load_trace(&args.trace, None)?;
    let calibration = load_calibration(&args.calibration)?;
    let policy = parse_policy(&args.policy)?;
    let payload = unique_payload(&trace);
    let cache_bytes = (payload as f64 * args.cache_fraction) as u64;
    let params = PlacementParams {
        block_shards: args.block_shards,
        hot_shards: args.hot_shards,
        ..PlacementParams::default()
    };
    let weights: CoAccessWeights =
        placement::weights(&trace, Some(&calibration.size_law), params, args.seed)?;
    let mut rows = Vec::new();
    for name in &args.kinds {
        let kind = PlacementKind::parse(name)?;
        let placement = placement::build(&trace, &weights, kind, params)?;
        let config = SimConfig {
            policy,
            cache_bytes,
            queue_depth: 1,
            max_in_flight_bytes: 512 * 1024 * 1024,
            coalesce_gap_bytes: args.coalesce_gap_bytes,
            max_run_bytes: 32 * 1024 * 1024,
            verify: true,
            compute_ns_per_record: 0,
            deadline_ns: None,
            seed: args.seed,
            backpressure: BackpressureMode::Wait,
            placement: Some(placement.clone()),
            two_timescale: None,
            cancel_prefetch_on_phase_change: true,
            preempt_prefetch_for_required: true,
            start_record: 0,
            max_prefetch_shards: 4096,
            inject_checksum_failure: None,
        };
        let mut predictor = NoPrefetch;
        let simulation =
            selvedge_sim::simulate(&trace, &calibration, config, &mut predictor, None)?;
        simulation.summary.check()?;
        let objective = placement::objective(
            &placement,
            &weights,
            args.lambda_bytes,
            args.lambda_eviction,
        );
        let records = simulation.summary.evaluated_records.max(1);
        rows.push(PlacementRow {
            kind,
            amplification: placement.amplification,
            placed_bytes: placement.placed_bytes,
            replicated_bytes: placement.replicated_bytes,
            p50_stall_ns: simulation.summary.stall.p50_ns,
            p95_stall_ns: simulation.summary.stall.p95_ns,
            p99_stall_ns: simulation.summary.stall.p99_ns,
            bytes_device_read: simulation.summary.bytes_device_read,
            bytes_per_record: simulation.summary.bytes_device_read as f64 / records as f64,
            unused_prefetch_bytes: simulation.summary.bytes_prefetch_wasted,
            estimated_stall_saved_ns: objective.estimated_stall_saved_ns,
            objective: objective.objective,
        });
    }
    let receipt = serde_json::json!({
        "schema": "selvedge/sim-placement-comparison/v1",
        "stamp": args.stamp,
        "revision": args.revision,
        "trace_id": trace.manifest.trace_id,
        "trace_sha256": trace.manifest.records_sha256.to_hex(),
        "calibration": selvedge_sim::CalibrationRef::of(&calibration),
        "weights": {
            "records": weights.records,
            "shards": weights.shards,
            "edges": weights.edges.len(),
            "checked_edges": weights.checked_edges,
            "stable_edges": weights.stable_edges,
            "stable_fraction": weights.stable_fraction,
            "weighting": weights.weighting,
            "notes": weights.notes,
            "top_edges": weights.edges.iter().take(16).collect::<Vec<_>>(),
        },
        "rows": rows,
        "notes": [
            "layouts are proposals from the surrogate; the measured comparison is the simulator row",
            "a replicated layout's extra bytes are storage amplification, not free",
        ],
    });
    std::fs::write(&args.out, serde_json::to_vec_pretty(&receipt)?)?;
    println!("receipt: {}", args.out.display());
    println!("| layout | amplification | p95 ns | bytes/record | objective |");
    for row in &rows {
        println!(
            "| {} | {:.3} | {} | {:.1} | {:.0} |",
            row.kind.name(),
            row.amplification,
            row.p95_stall_ns,
            row.bytes_per_record,
            row.objective
        );
    }
    if args.json {
        println!("{}", serde_json::to_string_pretty(&receipt)?);
    }
    Ok(ExitCode::SUCCESS)
}
