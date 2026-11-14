//! `selvedge io-scan` and `selvedge store bench`: measured storage commands.
//!
//! Both commands write a receipt-shaped JSON document and print a compact
//! summary. They never write test files: `io-scan` reads an existing file and
//! `store bench` reads an existing bundle, so neither can fill the filesystem.

use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Args, Subcommand};
use selvedge_store::io_scan::{self, IoScanConfig};
use selvedge_store::store_bench::{self, BenchConfig};

use crate::CliResult;

/// Arguments for `selvedge store`.
#[derive(Debug, Args)]
pub(crate) struct StoreArgs {
    #[command(subcommand)]
    pub(crate) command: StoreCommand,
}

/// `selvedge store` subcommands.
#[derive(Debug, Subcommand)]
pub(crate) enum StoreCommand {
    /// Measure exact-mode reads of a real bundle under several cache policies.
    Bench(BenchArgs),
}

/// Arguments for `selvedge store bench`.
#[derive(Debug, Args)]
pub(crate) struct BenchArgs {
    /// Validated bundle to read.
    #[arg(long)]
    pub(crate) bundle: PathBuf,
    /// Where to write the JSON receipt.
    #[arg(long)]
    pub(crate) out: PathBuf,
    /// Directory whose filesystem free space is recorded.
    #[arg(long, default_value = ".")]
    pub(crate) workdir: PathBuf,
    /// Comma-separated arm names; empty means the default list.
    #[arg(long, value_delimiter = ',')]
    pub(crate) arms: Vec<String>,
    /// Directory for per-arm JSONL request metrics.
    #[arg(long)]
    pub(crate) metrics_dir: Option<PathBuf>,
    /// Coalescing cap as a multiple of the largest extent.
    #[arg(long, default_value_t = 4)]
    pub(crate) run_multiple: u64,
    /// Layer used by the prefetch-waste arm.
    #[arg(long, default_value_t = 0)]
    pub(crate) prefetch_layer: u32,
    /// Receipt stamp; defaults to now.
    #[arg(long)]
    pub(crate) stamp: Option<String>,
    /// Git revision; defaults to `SELVEDGE_REVISION`.
    #[arg(long)]
    pub(crate) revision: Option<String>,
}

/// Arguments for `selvedge io-scan`.
#[derive(Debug, Args)]
pub(crate) struct IoScanArgs {
    /// Existing file to read. It is opened read-only.
    #[arg(long)]
    pub(crate) file: PathBuf,
    /// Where to write the JSON receipt.
    #[arg(long)]
    pub(crate) out: PathBuf,
    /// Directory whose filesystem free space is recorded.
    #[arg(long, default_value = ".")]
    pub(crate) workdir: PathBuf,
    /// Small sweep for a smoke test; the recorded baseline uses the defaults.
    #[arg(long)]
    pub(crate) quick: bool,
    /// Disable the page-cache drop (warm measurement).
    #[arg(long)]
    pub(crate) no_drop_cache: bool,
    /// Sampling seed.
    #[arg(long, default_value_t = 0x5eed_5eed)]
    pub(crate) seed: u64,
    /// Receipt stamp; defaults to now.
    #[arg(long)]
    pub(crate) stamp: Option<String>,
    /// Git revision; defaults to `SELVEDGE_REVISION`.
    #[arg(long)]
    pub(crate) revision: Option<String>,
}

/// Run `selvedge store ...`.
///
/// # Errors
///
/// Propagates every measurement, validation and receipt-writing failure.
pub(crate) fn run_store(args: &StoreArgs) -> CliResult {
    match &args.command {
        StoreCommand::Bench(bench) => run_bench(bench),
    }
}

fn run_bench(args: &BenchArgs) -> CliResult {
    let config = BenchConfig {
        bundle: args.bundle.clone(),
        workdir: args.workdir.clone(),
        metrics_dir: args.metrics_dir.clone(),
        arms: args.arms.clone(),
        run_multiple: args.run_multiple,
        prefetch_layer: args.prefetch_layer,
        stamp: args.stamp.clone(),
        revision: args.revision.clone(),
        argv: std::env::args().collect(),
    };
    let report = store_bench::run(&config)?;
    write_json(&args.out, &report)?;
    println!(
        "store-bench: bundle={} payload={} bytes shards={} stamp={} revision={}",
        report.bundle.bundle_id,
        report.bundle.payload_len_bytes,
        report.bundle.shard_count,
        report.stamp,
        report.revision
    );
    let mut failures = 0_u64;
    for arm in &report.arms {
        if let Some(reason) = &arm.skipped {
            println!("  {:<22} SKIPPED {reason}", arm.arm);
            continue;
        }
        let pass2 = arm
            .pass2
            .as_ref()
            .map(|pass| {
                format!(
                    " pass2 {:>8.1} MiB/s hits={} page_drop={}",
                    pass.throughput_mib_s, pass.hits, pass.page_cache_dropped
                )
            })
            .unwrap_or_default();
        println!(
            "  {:<22} pass1 {:>8.1} MiB/s p95={} ns st={} dev={}{pass2}",
            arm.arm,
            arm.pass1.throughput_mib_s,
            arm.pass1.stall_p95_ns,
            arm.store.stall_ns,
            arm.pass1
                .device
                .as_ref()
                .map(|device| format!("{} B amp {:.2}", device.read_bytes, device.amplification))
                .unwrap_or_else(|| "n/a".to_owned()),
        );
        failures += arm.pass1.errors;
        if let Some(pass) = &arm.pass2 {
            failures += pass.errors;
        }
        failures += arm.store.checksum_failures;
    }
    if failures > 0 {
        eprintln!("store-bench: {failures} read error(s); the receipt records them");
        return Ok(ExitCode::FAILURE);
    }
    Ok(ExitCode::SUCCESS)
}

/// Run `selvedge io-scan`.
///
/// # Errors
///
/// Propagates every measurement, validation and receipt-writing failure.
pub(crate) fn run_io_scan(args: &IoScanArgs) -> CliResult {
    let mut config = IoScanConfig {
        file: args.file.clone(),
        workdir: args.workdir.clone(),
        drop_cache: !args.no_drop_cache,
        seed: args.seed,
        stamp: args.stamp.clone(),
        revision: args.revision.clone(),
        argv: std::env::args().collect(),
        ..IoScanConfig::default()
    };
    if args.quick {
        config.block_sizes = vec![4096, 65_536];
        config.sequential_bytes = 16 * 1024 * 1024;
        config.random_samples = vec![64, 32];
        config.queue_depths = vec![1, 4];
        config.qd_total_bytes = 16 * 1024 * 1024;
        config.alignment_read_bytes = 1024 * 1024;
        config.overread_bytes = 2 * 1024 * 1024;
        config.checksum_bytes = 64 * 1024 * 1024;
        config.min_bytes = 8 * 1024 * 1024;
    }
    let report = io_scan::run(&config)?;
    write_json(&args.out, &report)?;
    println!(
        "io-scan: file={} bytes={} mount={} ({}) free={} bytes cache_dropped={} ratio={:.1}",
        report.file.path.display(),
        report.file.len_bytes,
        report.mount.target,
        report.mount.fstype,
        report.mount.free_bytes,
        report.cache_control.dropped,
        report.cache_control.cold_warm_ratio
    );
    for point in &report.sequential {
        println!(
            "  seq {:>9} B {:>8.1} MiB/s p50={} p95={} p99={} ns dev_amp={}",
            point.block_bytes,
            point.throughput_mib_s,
            point.p50_ns,
            point.p95_ns,
            point.p99_ns,
            point
                .device_amplification
                .map(|value| format!("{value:.2}"))
                .unwrap_or_else(|| "n/a".to_owned())
        );
    }
    for point in &report.sequential_noreadahead {
        println!(
            "  seq-nr {:>6} B {:>8.1} MiB/s p50={} p95={} p99={} ns dev_amp={}",
            point.block_bytes,
            point.throughput_mib_s,
            point.p50_ns,
            point.p95_ns,
            point.p99_ns,
            point
                .device_amplification
                .map(|value| format!("{value:.2}"))
                .unwrap_or_else(|| "n/a".to_owned())
        );
    }
    for point in &report.random {
        println!(
            "  rnd {:>9} B p50={} p95={} p99={} ns dev_amp={}",
            point.block_bytes,
            point.p50_ns,
            point.p95_ns,
            point.p99_ns,
            point
                .device_amplification
                .map(|value| format!("{value:.2}"))
                .unwrap_or_else(|| "n/a".to_owned())
        );
    }
    for point in &report.queue_depth {
        println!(
            "  qd  {:>2}        {:>8.1} MiB/s p50={} p99={} ns",
            point.queue_depth, point.throughput_mib_s, point.p50_ns, point.p99_ns
        );
    }
    for point in &report.alignment {
        println!(
            "  align {:<38} req={} useful={} dev={} amp={}",
            point.label,
            point.bytes_requested,
            point.bytes_useful,
            point
                .device_bytes
                .map(|value| value.to_string())
                .unwrap_or_else(|| "n/a".to_owned()),
            point
                .device_per_useful
                .map(|value| format!("{value:.3}"))
                .unwrap_or_else(|| "n/a".to_owned())
        );
    }
    println!(
        "  sha256 {:.1} MiB/s warm, {:.0} ns/GiB, shard {} B in {} ns",
        report.checksum.warm_throughput_mib_s,
        report.checksum.warm_ns_per_gib,
        report.checksum.shard_bytes,
        report.checksum.shard_hash_ns
    );
    Ok(ExitCode::SUCCESS)
}

fn write_json<T: serde::Serialize>(path: &std::path::Path, value: &T) -> CliResult {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)?;
    }
    let text = serde_json::to_string_pretty(value)?;
    let temporary = path.with_extension("json.tmp");
    {
        use std::io::Write;
        let mut file = std::fs::File::create(&temporary)?;
        file.write_all(text.as_bytes())?;
        file.write_all(b"\n")?;
        file.sync_all()?;
    }
    std::fs::rename(&temporary, path)?;
    Ok(ExitCode::SUCCESS)
}
