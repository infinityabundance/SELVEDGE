//! `selvedge` — command-line tools for the SELVEDGE research program.
//!
//! Implemented commands:
//!
//! - `selvedge admit --config <json>`: admit one arm's declared selection path
//!   against the active-parameter ceiling and emit an admission record;
//! - `selvedge doctor [--json] [--probe-host]`: environment and schema probe;
//!   exits non-zero on any failed check;
//! - `selvedge inspect <bundle|manifest> [--json]`: read-only shard table;
//! - `selvedge inspect-model <dir> [--json]`: exact parameter/FLOP accounting;
//! - `selvedge format validate <bundle> [--json]`: full validator;
//! - `selvedge court c|d`: executable Court C (budget accounting) and Court D
//!   (storage integrity/crash) scenarios;
//! - `selvedge store bench <bundle>`: exact-mode storage measurement;
//! - `selvedge io-scan <file>`: physical I/O characterisation.

#![forbid(unsafe_code)]

use std::process::ExitCode;

use clap::{Args, Parser, Subcommand};

mod admit_cmd;
mod atlas_cmd;
mod config;
mod convert_cmd;
mod court;
mod doctor;
mod inspect;
mod inspect_model_cmd;
mod store_cmd;
mod validate_cmd;

#[derive(Debug, Parser)]
#[command(
    name = "selvedge",
    version,
    about = "SELVEDGE command-line tools",
    long_about = "Deterministic data-contract, bundle and verification tools for \
                  the SELVEDGE research program. Commands never mutate an input \
                  artifact and print typed errors instead of guessing."
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Debug, Subcommand)]
// The CLI variant carries all conversion options by value; boxing it would only
// move the allocation without changing argument handling.
#[allow(clippy::large_enum_variant)]
enum Commands {
    /// Admit a declared selection path against the active-parameter ceiling.
    Admit(admit_cmd::AdmitArgs),
    /// Probe the environment, config, schema versions and toolchain identity.
    Doctor(doctor::DoctorArgs),
    /// Print a shard table for a bundle or a bare manifest. Read-only.
    Inspect(inspect::InspectArgs),
    /// Account the exact learned parameters and FLOPs of a real checkpoint.
    InspectModel(inspect_model_cmd::InspectModelArgs),
    /// Versioned bundle format commands.
    Format(FormatArgs),
    /// Convert a safetensors SwiGLU checkpoint into repacked group bundles.
    Convert(convert_cmd::ConvertArgs),
    /// Run the executable verification courts.
    Court(court::CourtArgs),
    /// Atlas selector fit, select and matched-budget compare.
    Atlas(atlas_cmd::AtlasArgs),
    /// Tiered shard store commands (exact-mode bench).
    Store(store_cmd::StoreArgs),
    /// Measure physical I/O on an existing file.
    IoScan(store_cmd::IoScanArgs),
}

#[derive(Debug, Args)]
struct FormatArgs {
    #[command(subcommand)]
    command: FormatCommand,
}

#[derive(Debug, Subcommand)]
enum FormatCommand {
    /// Validate a bundle's structure and payload hashes.
    Validate(validate_cmd::ValidateArgs),
}

type CliResult = Result<ExitCode, Box<dyn std::error::Error>>;

fn main() -> ExitCode {
    let cli = Cli::parse();
    let result: CliResult = match cli.command {
        Commands::Admit(args) => admit_cmd::run(&args),
        Commands::Doctor(args) => doctor::run(&args),
        Commands::Inspect(args) => inspect::run(&args),
        Commands::InspectModel(args) => inspect_model_cmd::run(&args),
        Commands::Format(args) => match args.command {
            FormatCommand::Validate(args) => validate_cmd::run(&args),
        },
        Commands::Convert(args) => convert_cmd::run(&args),
        Commands::Court(args) => court::run(&args),
        Commands::Atlas(args) => atlas_cmd::run(&args),
        Commands::Store(args) => store_cmd::run_store(&args),
        Commands::IoScan(args) => store_cmd::run_io_scan(&args),
    };
    match result {
        Ok(code) => code,
        Err(error) => {
            eprintln!("selvedge: {error}");
            ExitCode::from(2)
        }
    }
}
