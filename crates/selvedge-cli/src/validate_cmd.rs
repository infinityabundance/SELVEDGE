//! `selvedge format validate`: the full bundle validation pipeline.

use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Args, ValueEnum};
use selvedge_core::Sha256Digest;
use selvedge_format::{ValidationOptions, ValidationReport, VerifyMode, validate_bundle};
use serde::Serialize;

use crate::CliResult;

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub(crate) enum VerifyArg {
    /// Verify every shard hash.
    All,
    /// Verify only `--select` shards.
    Selected,
    /// Structural checks only; hashes are reported as skipped.
    None,
}

#[derive(Debug, Args)]
pub(crate) struct ValidateArgs {
    /// Bundle to validate.
    pub(crate) bundle: PathBuf,
    /// Emit the structured report as JSON.
    #[arg(long)]
    pub(crate) json: bool,
    /// Expected model digest (64 hex digits, optional `sha256:` prefix).
    #[arg(long)]
    pub(crate) expect_digest: Option<String>,
    /// Expected compatibility id, checked against every shard.
    #[arg(long)]
    pub(crate) expect_compatibility: Option<String>,
    /// Hash verification mode.
    #[arg(long, value_enum, default_value_t = VerifyArg::All)]
    pub(crate) verify: VerifyArg,
    /// Shard ids to verify when `--verify selected` is used. Repeatable or
    /// comma-separated.
    #[arg(long = "select")]
    pub(crate) select: Vec<String>,
    /// Optional TOML configuration (limits and alignment defaults).
    #[arg(long)]
    pub(crate) config: Option<PathBuf>,
    /// Emit JSON on failure as well as success.
    #[arg(long)]
    pub(crate) json_errors: bool,
}

#[derive(Debug, Serialize)]
struct FailureReport<'a> {
    schema: &'a str,
    ok: bool,
    bundle: String,
    error: String,
}

/// Run the validate command.
///
/// # Errors
///
/// Propagates only argument/configuration errors; a validation failure is
/// reported and turns into exit code 1.
pub(crate) fn run(args: &ValidateArgs) -> CliResult {
    let config = crate::config::load(&args.config)?;
    let verify = match args.verify {
        VerifyArg::All => VerifyMode::All,
        VerifyArg::None => VerifyMode::StructuralOnly,
        VerifyArg::Selected => {
            let mut ids = Vec::new();
            for value in &args.select {
                for part in value.split(',') {
                    let part = part.trim();
                    if part.is_empty() {
                        continue;
                    }
                    ids.push(selvedge_core::ShardId::parse(part)?);
                }
            }
            if ids.is_empty() {
                return Err("--verify selected requires at least one --select id".into());
            }
            VerifyMode::Selected(ids)
        }
    };
    let expected_model_digest = match &args.expect_digest {
        Some(value) => Some(Sha256Digest::parse_hex(value)?),
        None => None,
    };
    let options = ValidationOptions {
        verify,
        expected_model_digest,
        expected_compatibility_id: args.expect_compatibility.clone(),
        max_manifest_bytes: config.storage.max_manifest_bytes,
        max_bundle_bytes: config.storage.max_bundle_bytes,
        cancel: None,
    };

    match validate_bundle(&args.bundle, &options) {
        Ok(report) => {
            if args.json {
                println!("{}", serde_json::to_string_pretty(&report)?);
            } else {
                print_human(&args.bundle, &report);
            }
            Ok(ExitCode::SUCCESS)
        }
        Err(error) => {
            if args.json || args.json_errors {
                let failure = FailureReport {
                    schema: "selvedge/validate-error/v1",
                    ok: false,
                    bundle: args.bundle.display().to_string(),
                    error: error.to_string(),
                };
                println!("{}", serde_json::to_string_pretty(&failure)?);
            } else {
                eprintln!("invalid {}: {error}", args.bundle.display());
            }
            Ok(ExitCode::FAILURE)
        }
    }
}

fn print_human(path: &std::path::Path, report: &ValidationReport) {
    println!(
        "valid bundle={} id={} shards={} shard_bytes={} verified={} skipped={}",
        path.display(),
        report.bundle_id,
        report.shard_count,
        report.total_shard_bytes,
        report.verified_shards.len(),
        report.skipped_shards.len()
    );
    println!(
        "model={} manifest_sha256={} format_version={} manifest_schema_version={} alignment={} payload_bytes={}",
        report.model_digest,
        report.manifest_sha256,
        report.format_version,
        report.manifest_schema_version,
        report.alignment,
        report.payload_len
    );
    if report.verify_mode == "structural_only" {
        println!("note: structural validation only; payload hashes were not verified");
    }
}
