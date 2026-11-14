//! `selvedge convert` and `selvedge convert verify`.

use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Args, Subcommand};
use selvedge_convert::{
    CheckpointSource, ConversionOptions, DiscoveredFfn, FfnOrientation, FfnTensorMap, GroupSpec,
    TensorSource, Tolerance, VerificationOptions, convert, discover_ffn_maps,
    model_digest_for_path, plan_conversion, report_plan, verify_bundle,
};
use serde::Serialize;

use crate::CliResult;

#[derive(Debug, Args)]
pub(crate) struct ConvertArgs {
    #[command(subcommand)]
    pub(crate) command: Option<ConvertCommand>,
    #[command(flatten)]
    pub(crate) convert: ConvertOptionsArgs,
}

#[derive(Debug, Subcommand)]
pub(crate) enum ConvertCommand {
    /// Verify a bundle's groups against the source checkpoint.
    Verify(VerifyArgs),
}

/// Tensor-map selection shared by convert and verify.
#[derive(Clone, Debug, Args)]
pub(crate) struct MapArgs {
    /// Explicit gate tensor key. Requires `--up-key` and `--down-key`.
    #[arg(long)]
    pub(crate) gate_key: Option<String>,
    /// Explicit up tensor key.
    #[arg(long)]
    pub(crate) up_key: Option<String>,
    /// Explicit down tensor key.
    #[arg(long)]
    pub(crate) down_key: Option<String>,
    /// Explicit orientation: `intermediate_first` or `hidden_first`.
    #[arg(long, default_value = "intermediate_first")]
    pub(crate) orientation: String,
    /// Layer index for an explicit map.
    #[arg(long)]
    pub(crate) layer: Option<u32>,
}

#[derive(Debug, Args)]
pub(crate) struct ConvertOptionsArgs {
    /// Source safetensors file or directory.
    #[arg(long)]
    pub(crate) model: Option<PathBuf>,
    /// Destination directory for the published bundle.
    #[arg(long)]
    pub(crate) out: Option<PathBuf>,
    /// Grouping: a bare count (contiguous groups) or semicolon-separated
    /// explicit channel lists such as `0-3,5;6-9`.
    #[arg(long, default_value = "4")]
    pub(crate) groups: String,
    /// Require the groups to cover every intermediate channel.
    #[arg(long)]
    pub(crate) complete: bool,
    /// Shard alignment in bytes (overrides the configuration).
    #[arg(long)]
    pub(crate) alignment: Option<u64>,
    /// TOML configuration path.
    #[arg(long)]
    pub(crate) config: Option<PathBuf>,
    /// Emit machine-readable JSON.
    #[arg(long)]
    pub(crate) json: bool,
    /// Print the shard plan without reading tensor data or writing a bundle.
    #[arg(long)]
    pub(crate) plan: bool,
    /// In-flight tensor-byte ceiling.
    #[arg(long)]
    pub(crate) max_inflight_bytes: Option<u64>,
    /// Explicit bundle id.
    #[arg(long)]
    pub(crate) bundle_id: Option<String>,
    /// Replace an existing bundle with the same id.
    #[arg(long)]
    pub(crate) force: bool,
    #[command(flatten)]
    pub(crate) map: MapArgs,
}

#[derive(Debug, Args)]
pub(crate) struct VerifyArgs {
    /// Bundle to verify.
    #[arg(long)]
    pub(crate) bundle: PathBuf,
    /// Source safetensors file or directory.
    #[arg(long)]
    pub(crate) model: PathBuf,
    /// Emit machine-readable JSON.
    #[arg(long)]
    pub(crate) json: bool,
    /// TOML configuration path.
    #[arg(long)]
    pub(crate) config: Option<PathBuf>,
    /// Absolute tolerance component.
    #[arg(long)]
    pub(crate) abs_tol: Option<f64>,
    /// Relative tolerance component.
    #[arg(long)]
    pub(crate) rel_tol: Option<f64>,
    /// Deterministic input seed.
    #[arg(long)]
    pub(crate) seed: Option<u64>,
    /// Comma-separated batch sizes (`N` flattened).
    #[arg(long)]
    pub(crate) batch_sizes: Option<String>,
    /// In-flight tensor-byte ceiling for source reads.
    #[arg(long)]
    pub(crate) max_inflight_bytes: Option<u64>,
    /// Per-shard read cap in bytes.
    #[arg(long)]
    pub(crate) max_shard_bytes: Option<u64>,
    #[command(flatten)]
    pub(crate) map: MapArgs,
}

/// Run the convert command tree.
///
/// # Errors
///
/// Propagates argument, configuration, checkpoint, planning, conversion and
/// verification failures.
pub(crate) fn run(args: &ConvertArgs) -> CliResult {
    match &args.command {
        Some(ConvertCommand::Verify(verify)) => run_verify(verify),
        None => run_convert(&args.convert),
    }
}

fn run_convert(args: &ConvertOptionsArgs) -> CliResult {
    let model = args
        .model
        .as_ref()
        .ok_or("convert requires --model <safetensors|dir>")?;
    let mut config = crate::config::load(&args.config)?;
    if let Some(alignment) = args.alignment {
        config.storage.alignment = alignment;
        config.validate()?;
    }
    let config_hash = config.config_hash()?;
    let mut source = CheckpointSource::open(model)?;
    let maps = discover(&source, &args.map)?;
    let digest = model_digest_for_path(model)?;
    let spec = GroupSpec::parse(&args.groups)?;
    let max_inflight_bytes = args
        .max_inflight_bytes
        .unwrap_or(config.memory.resident_cap_bytes);
    let options = ConversionOptions {
        alignment: config.storage.alignment,
        max_inflight_bytes,
        complete: args.complete,
        bundle_id: args.bundle_id.clone(),
        created_by: format!("selvedge convert {}", env!("CARGO_PKG_VERSION")),
        force: args.force,
        config_hash,
    };
    let plan = plan_conversion(&source, &maps, &spec, digest, &options)?;
    if args.plan {
        let report = report_plan(&plan, &options)?;
        print_report(&report, args.json, true)?;
        return Ok(ExitCode::SUCCESS);
    }
    let out = args.out.as_ref().ok_or("convert requires --out <dir>")?;
    let (_published, report) = convert(&mut source, &plan, out, &options)?;
    print_report(&report, args.json, false)?;
    Ok(ExitCode::SUCCESS)
}

fn run_verify(args: &VerifyArgs) -> CliResult {
    let config = crate::config::load(&args.config)?;
    let mut source = CheckpointSource::open(&args.model)?;
    let maps = discover(&source, &args.map)?;
    let mut tolerance = Tolerance::default();
    if let Some(abs) = args.abs_tol {
        tolerance.abs = abs;
    }
    if let Some(rel) = args.rel_tol {
        tolerance.rel = rel;
    }
    let batch_sizes = match &args.batch_sizes {
        Some(text) => {
            let mut values = Vec::new();
            for part in text.split(',') {
                let part = part.trim();
                if part.is_empty() {
                    continue;
                }
                values.push(part.parse::<u64>().map_err(|_| {
                    format!("invalid batch size {part:?}; expected a positive integer")
                })?);
            }
            if values.is_empty() {
                return Err("--batch-sizes needs at least one value".into());
            }
            values
        }
        None => VerificationOptions::default().batch_sizes,
    };
    let defaults = VerificationOptions::default();
    let options = VerificationOptions {
        tolerance,
        batch_sizes,
        seed: args.seed.unwrap_or(defaults.seed),
        max_shard_bytes: args.max_shard_bytes.unwrap_or(defaults.max_shard_bytes),
        max_inflight_bytes: args
            .max_inflight_bytes
            .unwrap_or(config.memory.resident_cap_bytes),
    };
    let report = verify_bundle(&args.bundle, &mut source, &maps, &options)?;
    if args.json {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        print_verification(&report);
    }
    Ok(if report.ok {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    })
}

fn discover(
    source: &CheckpointSource,
    args: &MapArgs,
) -> Result<Vec<DiscoveredFfn>, Box<dyn std::error::Error>> {
    match (&args.gate_key, &args.up_key, &args.down_key) {
        (Some(gate), Some(up), Some(down)) => {
            let orientation = FfnOrientation::parse(&args.orientation)?;
            let map = FfnTensorMap::new(gate.clone(), up.clone(), down.clone(), orientation);
            Ok(vec![DiscoveredFfn {
                prefix: "<explicit>".to_owned(),
                layer: args.layer,
                map,
            }])
        }
        (None, None, None) => {
            let maps = discover_ffn_maps(&source.tensor_keys())?;
            Ok(maps)
        }
        _ => Err("--gate-key, --up-key and --down-key must be given together".into()),
    }
}

fn print_report(
    report: &selvedge_convert::ConversionReport,
    json: bool,
    plan: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    if json {
        println!("{}", serde_json::to_string_pretty(report)?);
        return Ok(());
    }
    println!(
        "{} bundle={} model={} shards={} payload_bytes={} alignment={} complete={}",
        if plan { "plan" } else { "converted" },
        report.bundle_id,
        report.model_digest,
        report.shard_count,
        report.payload_bytes,
        report.alignment,
        report.complete
    );
    if let Some(path) = &report.bundle_path {
        println!("path={path}");
    }
    println!(
        "in-flight peak={} bytes, ceiling={} bytes",
        report.peak_inflight_bytes, report.max_inflight_bytes
    );
    for shard in &report.shards {
        println!(
            "  {} group={} role={} channels={} shape={} storage={} bytes={} transform={}",
            shard.short_id,
            shard.group,
            shard.role,
            shard.channels,
            shard.shape,
            shard.storage_order,
            shard.bytes,
            shard.transform_id
        );
    }
    Ok(())
}

fn print_verification(report: &selvedge_convert::VerificationReport) {
    println!(
        "bundle={} model={} ok={} layers={} validated_shards={} seed={} batches={:?}",
        report.bundle,
        report.model_digest,
        report.ok,
        report.layers.len(),
        report.validated_shards,
        report.seed,
        report.batch_sizes
    );
    for layer in &report.layers {
        println!(
            "layer={:?} source_dtype={} hidden={} intermediate={} groups={} passed={}",
            layer.layer,
            layer.source_dtype,
            layer.hidden,
            layer.intermediate,
            layer.groups.len(),
            layer.passed
        );
        println!(
            "  full f32 max_abs={:.6e} max_rel={:.6e} mean_abs={:.6e} count={}",
            layer.full_f32.max_abs,
            layer.full_f32.max_rel,
            layer.full_f32.mean_abs,
            layer.full_f32.count
        );
        println!(
            "  full f64 max_abs={:.6e} max_rel={:.6e} mean_abs={:.6e} count={}",
            layer.full_f64.max_abs,
            layer.full_f64.max_rel,
            layer.full_f64.mean_abs,
            layer.full_f64.count
        );
        println!(
            "  tolerance abs={:.6e} rel={:.6e}",
            layer.full_f32.tolerance_abs, layer.full_f32.tolerance_rel
        );
    }
}

/// Serialize a report to a path, creating parent directories.
///
/// # Errors
///
/// Propagates serialization and filesystem failures.
pub(crate) fn write_report<T: Serialize>(
    path: &std::path::Path,
    report: &T,
) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let text = serde_json::to_string_pretty(report)?;
    std::fs::write(path, text)?;
    Ok(())
}
