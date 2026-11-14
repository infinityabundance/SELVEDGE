//! `selvedge inspect`: read-only shard table for a bundle or bare manifest.

use std::collections::BTreeMap;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clap::Args;
use selvedge_core::{ChannelSet, StorageOrder};
use selvedge_format::{BundleFile, MAGIC, Manifest, ShardRecord};
use serde::Serialize;

use crate::CliResult;

#[derive(Debug, Args)]
pub(crate) struct InspectArgs {
    /// Bundle (`.slv`) or manifest JSON path.
    pub(crate) path: PathBuf,
    /// Emit machine-readable JSON.
    #[arg(long)]
    pub(crate) json: bool,
}

#[derive(Debug, Serialize)]
struct InspectReport {
    schema: &'static str,
    source: String,
    kind: &'static str,
    bundle_id: String,
    model_digest: String,
    model_revision: Option<String>,
    tokenizer_digest: Option<String>,
    alignment: u64,
    payload_len: u64,
    file_len: Option<u64>,
    shard_count: u64,
    total_shard_bytes: u64,
    total_auxiliary_bytes: u64,
    by_dtype: BTreeMap<String, u64>,
    by_role: BTreeMap<String, u64>,
    shards: Vec<ShardRow>,
}

#[derive(Debug, Serialize)]
struct ShardRow {
    id: String,
    short: String,
    role: String,
    layer: Option<u32>,
    channels: String,
    channel_axis: u32,
    shape: String,
    dtype: String,
    storage_order: String,
    offset: u64,
    bytes: u64,
    auxiliary_bytes: u64,
    alignment: u64,
    sha256: String,
    compatibility_id: String,
    transform_id: String,
}

/// Run the inspect command.
///
/// # Errors
///
/// Rejects unreadable inputs and invalid bundles/manifests.
pub(crate) fn run(args: &InspectArgs) -> CliResult {
    let (manifest, kind, file_len) = load(&args.path)?;
    let report = build_report(&args.path, kind, file_len, &manifest)?;
    if args.json {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        print_human(&report);
    }
    Ok(ExitCode::SUCCESS)
}

fn load(path: &Path) -> Result<(Manifest, &'static str, Option<u64>), Box<dyn std::error::Error>> {
    let mut file = std::fs::File::open(path)?;
    let mut prefix = [0_u8; 8];
    let read = file.read(&mut prefix)?;
    let is_bundle = read > 0 && read <= MAGIC.len() && MAGIC[..read] == prefix[..read];
    if is_bundle {
        let bundle = BundleFile::open(path)?;
        let file_len = bundle.file_len();
        let manifest = bundle.manifest().clone();
        return Ok((manifest, "bundle", Some(file_len)));
    }
    let metadata = file.metadata()?;
    if metadata.len() > 8 * 1024 * 1024 {
        return Err(format!(
            "manifest {} is {} bytes, above the 8 MiB inspection cap",
            path.display(),
            metadata.len()
        )
        .into());
    }
    let bytes = std::fs::read(path)?;
    let manifest: Manifest = serde_json::from_slice(&bytes)?;
    manifest.validate()?;
    Ok((manifest, "manifest", Some(metadata.len())))
}

fn build_report(
    path: &Path,
    kind: &'static str,
    file_len: Option<u64>,
    manifest: &Manifest,
) -> Result<InspectReport, Box<dyn std::error::Error>> {
    let mut by_dtype = BTreeMap::new();
    let mut by_role = BTreeMap::new();
    let mut total_auxiliary = 0_u64;
    let mut shards = Vec::with_capacity(manifest.shards.len());
    for shard in &manifest.shards {
        let aux = shard.auxiliary_bytes()?;
        total_auxiliary = total_auxiliary
            .checked_add(aux)
            .ok_or("auxiliary byte overflow")?;
        *by_dtype.entry(shard.dtype.stable_name()).or_insert(0_u64) += shard.length;
        *by_role.entry(shard.role.to_string()).or_insert(0_u64) += shard.length;
        shards.push(row(shard, aux));
    }
    Ok(InspectReport {
        schema: "selvedge/inspect/v1",
        source: path.display().to_string(),
        kind,
        bundle_id: manifest.bundle_id.clone(),
        model_digest: manifest.model.digest().to_hex(),
        model_revision: manifest.model.upstream_revision().map(str::to_owned),
        tokenizer_digest: manifest
            .model
            .tokenizer_digest()
            .map(|digest| digest.to_hex()),
        alignment: manifest.alignment,
        payload_len: manifest.payload_len,
        file_len,
        shard_count: manifest.shards.len() as u64,
        total_shard_bytes: manifest.total_shard_bytes()?,
        total_auxiliary_bytes: total_auxiliary,
        by_dtype,
        by_role,
        shards,
    })
}

fn row(shard: &ShardRecord, auxiliary: u64) -> ShardRow {
    ShardRow {
        id: shard.shard_id.to_string(),
        short: shard.shard_id.short(),
        role: shard.role.to_string(),
        layer: shard.layer,
        channels: channels_text(&shard.channels),
        channel_axis: shard.channel_axis,
        shape: shard
            .shape
            .iter()
            .map(u64::to_string)
            .collect::<Vec<_>>()
            .join("x"),
        dtype: shard.dtype.stable_name(),
        storage_order: storage_text(&shard.storage_order),
        offset: shard.offset,
        bytes: shard.length,
        auxiliary_bytes: auxiliary,
        alignment: shard.alignment,
        sha256: shard.sha256.to_hex(),
        compatibility_id: shard.compatibility_id.clone(),
        transform_id: shard.transform_id.clone(),
    }
}

fn channels_text(channels: &ChannelSet) -> String {
    match channels {
        ChannelSet::Range { start, end } => format!("{start}..{end}"),
        ChannelSet::List { indices } => format!(
            "[{}]",
            indices
                .iter()
                .map(u64::to_string)
                .collect::<Vec<_>>()
                .join(",")
        ),
    }
}

fn storage_text(order: &StorageOrder) -> String {
    match order {
        StorageOrder::RowMajor => "row_major".to_owned(),
        StorageOrder::ColumnMajor => "column_major".to_owned(),
        StorageOrder::Transposed { axes } => format!(
            "transposed[{}]",
            axes.iter()
                .map(u32::to_string)
                .collect::<Vec<_>>()
                .join(",")
        ),
    }
}

fn print_human(report: &InspectReport) {
    println!("bundle      {}", report.bundle_id);
    println!("source      {} ({})", report.source, report.kind);
    println!("model       {}", report.model_digest);
    if let Some(revision) = &report.model_revision {
        println!("revision    {revision}");
    }
    println!(
        "alignment   {} bytes; payload {} bytes; shards {}; shard bytes {}; auxiliary bytes {}",
        report.alignment,
        report.payload_len,
        report.shard_count,
        report.total_shard_bytes,
        report.total_auxiliary_bytes
    );
    if let Some(file_len) = report.file_len {
        println!("file length {file_len} bytes");
    }
    println!();
    println!(
        "{:<12} {:<18} {:<5} {:<10} {:<12} {:<11} {:<9} {:>10} {:<6} {:<12}",
        "id", "role", "layer", "channels", "shape", "dtype", "storage", "bytes", "align", "sha256"
    );
    for shard in &report.shards {
        println!(
            "{:<12} {:<18} {:<5} {:<10} {:<12} {:<11} {:<9} {:>10} {:<6} {:<12}",
            shard.short,
            shard.role,
            shard
                .layer
                .map_or_else(|| "-".to_owned(), |layer| layer.to_string()),
            shard.channels,
            shard.shape,
            shard.dtype,
            shard.storage_order,
            shard.bytes,
            shard.alignment,
            &shard.sha256[..12],
        );
    }
    println!();
    println!("by dtype:");
    for (dtype, bytes) in &report.by_dtype {
        println!("  {dtype:<24} {bytes} bytes");
    }
}
