//! Executable Court C and Court D verification drivers.
//!
//! Court C lives in `selvedge_core::courts` so the integration tests and this
//! command share one implementation. Court D is driven here against the
//! deterministic synthetic fixture; the deep failure-mode tests live in
//! `crates/selvedge-format/tests/court_d.rs`.

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::atomic::{AtomicBool, Ordering};

use clap::{Args, Subcommand};
use selvedge_convert::court_a::run_court_a_with_provenance;
use selvedge_core::courts::run_budget_court;
use selvedge_core::testutil::TempDir;
use selvedge_fixtures::tiny_bundle;
use selvedge_format::{
    BundleCatalog, BundleFile, FormatError, ReloadVerdict, ValidationOptions, VerifyMode,
    WriteOptions, publish_bundle, validate_bundle, validate_sidecar,
};
use serde::Serialize;

use crate::CliResult;

#[derive(Debug, Args)]
pub(crate) struct CourtArgs {
    #[command(subcommand)]
    pub(crate) command: CourtCommand,
    /// Emit machine-readable JSON.
    #[arg(long, global = true)]
    pub(crate) json: bool,
    /// Keep Court D artifacts in this directory instead of a temporary one.
    #[arg(long, global = true)]
    pub(crate) workdir: Option<PathBuf>,
    /// Write the Court A report JSON to this path.
    #[arg(long, global = true)]
    pub(crate) report_out: Option<PathBuf>,
    /// Receipt stamp recorded in the Court A report.
    #[arg(long, global = true)]
    pub(crate) stamp: Option<String>,
    /// Git revision recorded in the Court A report.
    #[arg(long, global = true)]
    pub(crate) revision: Option<String>,
    /// Court A seed.
    #[arg(long, global = true)]
    pub(crate) seed: Option<u64>,
}

#[derive(Debug, Subcommand)]
pub(crate) enum CourtCommand {
    /// Court A: exact SwiGLU decomposition and reconstruction from bundles.
    A,
    /// Court A additions: grouped reconstruction on real teacher activations.
    AReal(RealCourtArgs),
    /// Court B cross-check: masked group partial sums from a real bundle.
    BReal(MaskedCourtArgs),
    /// Court C: learned-parameter budget accounting.
    C,
    /// Court D: storage integrity, atomic publication and crash behavior.
    D,
}

/// Arguments for the Court B real-bundle masked partial sum.
#[derive(Debug, Args)]
pub(crate) struct MaskedCourtArgs {
    /// Bundle to read.
    #[arg(long)]
    pub(crate) bundle: PathBuf,
    /// Layer index to read.
    #[arg(long)]
    pub(crate) layer: u32,
    /// Selected group indices, comma separated, ascending.
    #[arg(long, value_delimiter = ',')]
    pub(crate) groups: Vec<usize>,
    /// Raw little-endian f32 input of exactly rows*hidden values.
    #[arg(long)]
    pub(crate) input: PathBuf,
    /// Activation rows.
    #[arg(long)]
    pub(crate) rows: u64,
    /// Hidden width.
    #[arg(long)]
    pub(crate) hidden: u64,
    /// Where to write the raw little-endian f32 output.
    #[arg(long)]
    pub(crate) out_raw: PathBuf,
    /// Per-shard and per-tensor read cap in bytes.
    #[arg(long)]
    pub(crate) max_shard_bytes: Option<u64>,
}

/// Arguments for the real-activation Court A additions.
#[derive(Debug, Args)]
pub(crate) struct RealCourtArgs {
    #[command(flatten)]
    pub(crate) parity: crate::convert_cmd::RealParityCliArgs,
}

#[derive(Debug, Serialize)]
struct Scenario {
    name: &'static str,
    passed: bool,
    detail: String,
}

#[derive(Debug, Serialize)]
struct CourtDReport {
    court: &'static str,
    passed: bool,
    workdir: String,
    scenarios: Vec<Scenario>,
}

/// Run the requested court.
///
/// # Errors
///
/// Propagates fixture generation, publication and serialization failures.
pub(crate) fn run(args: &CourtArgs) -> CliResult {
    match &args.command {
        CourtCommand::A => run_a(
            args.json,
            args.report_out.as_deref(),
            args.stamp.clone(),
            args.revision.clone(),
            args.seed,
        ),
        CourtCommand::AReal(real) => run_a_real(
            real,
            args.json,
            args.report_out.as_deref(),
            args.stamp.clone(),
            args.revision.clone(),
        ),
        CourtCommand::BReal(masked) => run_b_real(
            masked,
            args.json,
            args.report_out.as_deref(),
            args.stamp.clone(),
            args.revision.clone(),
        ),
        CourtCommand::C => run_c(args.json),
        CourtCommand::D => run_d(args.json, args.workdir.as_deref()),
    }
}

fn run_a(
    json: bool,
    report_out: Option<&Path>,
    stamp: Option<String>,
    revision: Option<String>,
    seed: Option<u64>,
) -> CliResult {
    let seed = seed.unwrap_or(selvedge_convert::SYNTHETIC_SEED);
    let report = match run_court_a_with_provenance(seed, stamp, revision) {
        Ok(report) => report,
        Err(error) => {
            if json {
                let failure = serde_json::json!({
                    "schema": "selvedge/court-a-error/v1",
                    "ok": false,
                    "error": error.to_string(),
                });
                println!("{}", serde_json::to_string_pretty(&failure)?);
                return Ok(ExitCode::FAILURE);
            }
            return Err(Box::new(error));
        }
    };
    if let Some(path) = report_out {
        crate::convert_cmd::write_report(path, &report)?;
    }
    if json {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        println!("selvedge court a — SwiGLU decomposition and reconstruction");
        for scenario in &report.scenarios {
            let label = if scenario.passed { "PASS" } else { "FAIL" };
            println!("{} {}: {}", label, scenario.name, scenario.detail);
        }
        println!(
            "tolerance abs={:.3e} rel={:.3e}; scenarios={}",
            report.tolerance_abs, report.tolerance_rel, report.scenario_count
        );
        println!("overall: {}", if report.passed { "pass" } else { "fail" });
    }
    Ok(if report.passed {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    })
}

fn run_a_real(
    args: &RealCourtArgs,
    json: bool,
    report_out: Option<&Path>,
    stamp: Option<String>,
    revision: Option<String>,
) -> CliResult {
    let options = args.parity.options(stamp, revision);
    let report = match selvedge_convert::verify_real_parity(
        &args.parity.bundle,
        &args.parity.model,
        &args.parity.activations,
        &options,
    ) {
        Ok(report) => report,
        Err(error) => {
            if json {
                let failure = serde_json::json!({
                    "schema": "selvedge/court-a-real-error/v1",
                    "ok": false,
                    "error": error.to_string(),
                });
                println!("{}", serde_json::to_string_pretty(&failure)?);
                return Ok(ExitCode::FAILURE);
            }
            return Err(Box::new(error));
        }
    };
    if let Some(path) = report_out {
        crate::convert_cmd::write_report(path, &report)?;
    }
    if json {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        crate::convert_cmd::print_real_parity(&report);
    }
    Ok(if report.ok {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    })
}

fn run_b_real(
    args: &MaskedCourtArgs,
    json: bool,
    report_out: Option<&Path>,
    stamp: Option<String>,
    revision: Option<String>,
) -> CliResult {
    let options = selvedge_convert::MaskedOptions {
        max_shard_bytes: args.max_shard_bytes.unwrap_or(4 * 1024 * 1024 * 1024),
        stamp,
        revision,
    };
    let report = match selvedge_convert::masked_grouped_from_bundle(
        &selvedge_convert::MaskedRequest {
            bundle: &args.bundle,
            layer: args.layer,
            selected_groups: &args.groups,
            input: &args.input,
            rows: args.rows,
            hidden: args.hidden,
            output: &args.out_raw,
        },
        &options,
    ) {
        Ok(report) => report,
        Err(error) => {
            if json {
                let failure = serde_json::json!({
                    "schema": "selvedge/court-b-real-error/v1",
                    "ok": false,
                    "error": error.to_string(),
                });
                println!("{}", serde_json::to_string_pretty(&failure)?);
                return Ok(ExitCode::FAILURE);
            }
            return Err(Box::new(error));
        }
    };
    if let Some(path) = report_out {
        crate::convert_cmd::write_report(path, &report)?;
    }
    if json {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        println!("selvedge court b-real — masked grouped FFN from a real bundle");
        println!(
            "layer {} selected {}/{} groups ({} channels)",
            report.layer,
            report.selected_groups.len(),
            report.groups_total,
            report.selected_channels
        );
        println!("input  {} sha256={}", report.input, report.input_sha256);
        println!("output {} sha256={}", report.output, report.output_sha256);
        println!(
            "sum={:.9} abs_max={:.9}",
            report.output_sum_f64, report.output_abs_max
        );
    }
    Ok(ExitCode::SUCCESS)
}

fn run_c(json: bool) -> CliResult {
    let report = run_budget_court();
    if json {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        println!("selvedge court c — budget accounting");
        for scenario in &report.scenarios {
            let label = if scenario.passed { "PASS" } else { "FAIL" };
            println!("{} {}: {}", label, scenario.name, scenario.detail);
        }
        println!("overall: {}", if report.passed { "pass" } else { "fail" });
    }
    Ok(if report.passed {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    })
}

fn run_d(json: bool, requested_workdir: Option<&Path>) -> CliResult {
    let temporary;
    let workdir: PathBuf = match requested_workdir {
        Some(path) => {
            std::fs::create_dir_all(path)?;
            path.to_path_buf()
        }
        None => {
            temporary = TempDir::new("selvedge-court-d")?;
            temporary.path().to_path_buf()
        }
    };

    let fixture = tiny_bundle()?;
    let encoded = fixture.encoded()?;
    let write_options = WriteOptions {
        force: true,
        verify_before_publish: true,
        publish_sidecar: true,
        ..WriteOptions::default()
    };
    let published = publish_bundle(
        &workdir,
        &fixture.manifest,
        &fixture.payload,
        &write_options,
    )?;
    let options = ValidationOptions::default();
    let mut scenarios = Vec::new();

    scenarios.push(scenario("valid_bundle_validates", || {
        let report = validate_bundle(&published.path, &options).map_err(to_string)?;
        if report.shard_count == 0 || report.verified_shards.is_empty() {
            return Err("validation verified no shards".to_owned());
        }
        Ok(format!(
            "shards={} verified={} payload_bytes={}",
            report.shard_count,
            report.verified_shards.len(),
            report.payload_len
        ))
    }));

    scenarios.push(scenario("payload_bit_flip_detected", || {
        let report = validate_bundle(&published.path, &options).map_err(to_string)?;
        let first = fixture
            .manifest
            .shards
            .first()
            .ok_or_else(|| "fixture has no shards".to_owned())?;
        let index = report
            .payload_offset
            .checked_add(first.offset)
            .ok_or_else(|| "offset overflow".to_owned())?;
        let mut corrupt = encoded.clone();
        let index = usize::try_from(index).map_err(to_string)?;
        let byte = corrupt
            .get_mut(index)
            .ok_or_else(|| "payload index out of range".to_owned())?;
        *byte ^= 0x01;
        let path = workdir.join("corrupt-bitflip.slv");
        std::fs::write(&path, &corrupt).map_err(to_string)?;
        match validate_bundle(&path, &options) {
            Err(FormatError::HashMismatch { shard_id, .. }) => Ok(format!(
                "detected HashMismatch for shard {}",
                shard_id.short()
            )),
            Err(other) => Err(format!("unexpected error: {other}")),
            Ok(_) => Err("corrupted bundle was accepted".to_owned()),
        }
    }));

    scenarios.push(scenario("truncated_bundle_detected", || {
        if encoded.len() < 32 {
            return Err("encoded bundle too short to truncate".to_owned());
        }
        let truncated = encoded[..encoded.len() - 16].to_vec();
        let path = workdir.join("truncated.slv");
        std::fs::write(&path, &truncated).map_err(to_string)?;
        match validate_bundle(&path, &options) {
            Err(FormatError::FileLengthMismatch { .. }) => Ok(format!(
                "detected length mismatch after removing 16 bytes ({} -> {})",
                encoded.len(),
                truncated.len()
            )),
            Err(other) => Err(format!("unexpected error: {other}")),
            Ok(_) => Err("truncated bundle was accepted".to_owned()),
        }
    }));

    scenarios.push(scenario("selected_verification_skips_others", || {
        let first = fixture
            .manifest
            .shards
            .first()
            .ok_or_else(|| "fixture has no shards".to_owned())?;
        let selected = ValidationOptions {
            verify: VerifyMode::Selected(vec![first.shard_id]),
            ..ValidationOptions::default()
        };
        let report = validate_bundle(&published.path, &selected).map_err(to_string)?;
        if report.verified_shards != vec![first.shard_id] {
            return Err("selected shard was not the only verified shard".to_owned());
        }
        if report.skipped_shards.len() as u64 != report.shard_count - 1 {
            return Err("skipped shard count is wrong".to_owned());
        }
        Ok(format!(
            "verified={} skipped={}",
            report.verified_shards.len(),
            report.skipped_shards.len()
        ))
    }));

    scenarios.push(scenario("cancelled_read_aborts_and_recovers", || {
        let first = fixture
            .manifest
            .shards
            .first()
            .ok_or_else(|| "fixture has no shards".to_owned())?
            .clone();
        let mut bundle = BundleFile::open(&published.path).map_err(to_string)?;
        let cancel = AtomicBool::new(true);
        match bundle.read_shard_cancellable(&first.shard_id, 1 << 20, Some(&cancel)) {
            Err(FormatError::Cancelled) => {}
            Err(other) => return Err(format!("unexpected error: {other}")),
            Ok(_) => return Err("cancelled read returned bytes".to_owned()),
        }
        cancel.store(false, Ordering::Relaxed);
        let bytes = bundle
            .read_shard(&first.shard_id, 1 << 20)
            .map_err(to_string)?;
        if bytes.len() as u64 != first.length {
            return Err("recovered read returned the wrong length".to_owned());
        }
        Ok(format!(
            "cancelled then read {} bytes from shard {}",
            bytes.len(),
            first.shard_id.short()
        ))
    }));

    scenarios.push(scenario("concurrent_readers_never_see_torn_bundle", || {
        let path = published.path.clone();
        let mut readers = Vec::new();
        for _ in 0..2 {
            let path = path.clone();
            readers.push(std::thread::spawn(move || -> Result<usize, String> {
                let options = ValidationOptions::default();
                let mut validations = 0;
                for _ in 0..8 {
                    validate_bundle(&path, &options)
                        .map_err(|error| format!("reader observed {error}"))?;
                    validations += 1;
                }
                Ok(validations)
            }));
        }
        for _ in 0..4 {
            publish_bundle(
                &workdir,
                &fixture.manifest,
                &fixture.payload,
                &write_options,
            )
            .map_err(to_string)?;
        }
        let mut total = 0_usize;
        for reader in readers {
            total += reader
                .join()
                .map_err(|_| "reader thread panicked".to_owned())??;
        }
        Ok(format!(
            "{total} reader validations during 4 concurrent republishes"
        ))
    }));

    scenarios.push(scenario("hot_reload_rejects_conflicting_bundle_id", || {
        let mut catalog = BundleCatalog::new();
        match catalog.offer(&fixture.manifest).map_err(to_string)? {
            ReloadVerdict::AcceptedNew => {}
            other => return Err(format!("first offer was {other:?}")),
        }
        match catalog.offer(&fixture.manifest).map_err(to_string)? {
            ReloadVerdict::AcceptedIdentical => {}
            other => return Err(format!("identical offer was {other:?}")),
        }
        let mut conflicting = fixture.manifest.clone();
        conflicting.created_by = "conflicting-writer 0.0.0".to_owned();
        conflicting.validate().map_err(to_string)?;
        match catalog.offer(&conflicting).map_err(to_string)? {
            ReloadVerdict::RejectedConflict { .. } => {}
            other => return Err(format!("conflicting offer was {other:?}")),
        }
        if catalog.len() != 1 {
            return Err("catalog mutated on conflict".to_owned());
        }
        Ok("identical accepted; conflicting content rejected".to_owned())
    }));

    scenarios.push(scenario("sidecar_matches_bundle", || {
        let sidecar = published
            .sidecar
            .as_ref()
            .ok_or_else(|| "publication produced no sidecar".to_owned())?;
        validate_sidecar(&published.path, sidecar).map_err(to_string)?;
        Ok(format!("sidecar {}", sidecar.display()))
    }));

    let passed = scenarios.iter().all(|scenario| scenario.passed);
    let report = CourtDReport {
        court: "d",
        passed,
        workdir: workdir.display().to_string(),
        scenarios,
    };
    if json {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        println!("selvedge court d — storage integrity and crash behavior");
        for scenario in &report.scenarios {
            let label = if scenario.passed { "PASS" } else { "FAIL" };
            println!("{} {}: {}", label, scenario.name, scenario.detail);
        }
        println!(
            "workdir: {}{}",
            report.workdir,
            if requested_workdir.is_some() {
                " (kept)"
            } else {
                " (removed)"
            }
        );
        println!("overall: {}", if passed { "pass" } else { "fail" });
    }
    Ok(if passed {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    })
}

fn scenario(name: &'static str, body: impl FnOnce() -> Result<String, String>) -> Scenario {
    match body() {
        Ok(detail) => Scenario {
            name,
            passed: true,
            detail,
        },
        Err(detail) => Scenario {
            name,
            passed: false,
            detail,
        },
    }
}

fn to_string(error: impl std::fmt::Display) -> String {
    error.to_string()
}
