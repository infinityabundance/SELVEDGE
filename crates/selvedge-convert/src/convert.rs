//! Conversion planning, streaming repacking and publication.
//!
//! The converter reads one group at a time, writes each extent to a spool file
//! with a hash, and only then builds the manifest and publishes through the
//! format's atomic writer. It never holds a whole checkpoint or even a whole
//! matrix in memory; the in-flight budget covers the f32 region being converted.

use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use selvedge_core::{
    ChannelSet, DType, FORMAT_VERSION, ModelDigest, Sha256Builder, Sha256Digest, ShardId,
    ShardIdentity, StorageOrder, TensorRole, align_up,
};
use selvedge_format::{
    MANIFEST_SCHEMA, MANIFEST_SCHEMA_VERSION, Manifest, PublishedBundle, ShardRecord, WriteOptions,
    publish_bundle_from_reader,
};
use serde::Serialize;

use crate::error::{ConvertError, Result};
use crate::ffn::{
    DiscoveredFfn, FfnDims, FfnTensorMap, Group, GroupSpec, resolve_groups, validate_ffn_map,
};
use crate::tensor_source::{MemoryBudget, TensorSource};

/// Transform id for a gate shard stored row-major in group order.
pub const TRANSFORM_GATE: &str = "swiglu.gate.rowmajor.v1";
/// Transform id for an up shard stored row-major in group order.
pub const TRANSFORM_UP: &str = "swiglu.up.rowmajor.v1";
/// Transform id for a down shard stored as an explicitly transposed contiguous
/// copy. The logical shape is `[|G|, H]`; the physical bytes are `[H, |G|]`.
pub const TRANSFORM_DOWN: &str = "swiglu.down.transpose.v1";

/// Conversion knobs.
#[derive(Clone, Debug)]
pub struct ConversionOptions {
    /// Shard extent alignment in bytes.
    pub alignment: u64,
    /// In-flight tensor-byte ceiling.
    pub max_inflight_bytes: u64,
    /// Require groups to cover every intermediate channel.
    pub complete: bool,
    /// Explicit bundle id, or `None` to derive one.
    pub bundle_id: Option<String>,
    /// Producing tool string recorded in the manifest.
    pub created_by: String,
    /// Replace an existing bundle with the same id.
    pub force: bool,
    /// Canonical configuration hash recorded in the report.
    pub config_hash: String,
}

impl Default for ConversionOptions {
    fn default() -> Self {
        Self {
            alignment: selvedge_format::DEFAULT_ALIGNMENT,
            max_inflight_bytes: 256 * 1024 * 1024,
            complete: true,
            bundle_id: None,
            created_by: format!("selvedge-convert {}", env!("CARGO_PKG_VERSION")),
            force: false,
            config_hash: String::new(),
        }
    }
}

impl ConversionOptions {
    /// Validate the knobs.
    ///
    /// # Errors
    ///
    /// Rejects a non-power-of-two or too-small alignment, a zero budget and an
    /// empty producer string.
    pub fn validate(&self) -> Result<()> {
        if !self.alignment.is_power_of_two() || self.alignment < selvedge_format::MIN_ALIGNMENT {
            return Err(ConvertError::plan(format!(
                "alignment {} must be a power of two of at least {}",
                self.alignment,
                selvedge_format::MIN_ALIGNMENT
            )));
        }
        if self.max_inflight_bytes == 0 {
            return Err(ConvertError::plan("in-flight byte budget is zero"));
        }
        if self.created_by.is_empty() || self.created_by.len() > 256 {
            return Err(ConvertError::plan("created_by must be 1..=256 bytes"));
        }
        Ok(())
    }
}

/// One shard in the plan.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlannedShard {
    /// Layer index, when known.
    pub layer: Option<u32>,
    /// Group index inside the layer.
    pub group_index: usize,
    /// Tensor role.
    pub role: TensorRole,
    /// Group channels.
    pub channels: ChannelSet,
    /// Logical axis the channels apply to (always 0 here).
    pub channel_axis: u32,
    /// Logical shape of the stored slice.
    pub shape: Vec<u64>,
    /// Physical storage order.
    pub storage_order: StorageOrder,
    /// Transform id.
    pub transform_id: String,
    /// Channels as sorted indices.
    pub group_channels: Vec<u64>,
    /// Extent length in bytes.
    pub byte_len: u64,
}

/// One layer's plan.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LayerPlan {
    /// Layer index.
    pub layer: Option<u32>,
    /// Key prefix.
    pub prefix: String,
    /// Typed map.
    pub map: FfnTensorMap,
    /// Validated dimensions.
    pub dims: FfnDims,
    /// Groups in output order.
    pub groups: Vec<Group>,
}

/// The complete deterministic conversion plan.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConversionPlan {
    /// Source model identity.
    pub model: ModelDigest,
    /// Layers in output order.
    pub layers: Vec<LayerPlan>,
    /// Shards in output order.
    pub shards: Vec<PlannedShard>,
    /// Sorted transform ids.
    pub transforms: Vec<String>,
    /// Bundle id.
    pub bundle_id: String,
    /// Alignment.
    pub alignment: u64,
    /// Required coverage.
    pub complete: bool,
    /// Estimated payload length.
    pub payload_len: u64,
}

/// A serializable shard row.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct ShardReport {
    /// Short shard id (`-` before publication).
    pub short_id: String,
    /// Role name.
    pub role: String,
    /// Layer index.
    pub layer: Option<u32>,
    /// Group index.
    pub group: usize,
    /// Channels in manifest form.
    pub channels: String,
    /// Shape as `AxB`.
    pub shape: String,
    /// Storage order.
    pub storage_order: String,
    /// Transform id.
    pub transform_id: String,
    /// Extent bytes.
    pub bytes: u64,
    /// Extent SHA-256 (`-` before publication).
    pub sha256: String,
}

/// A serializable conversion report used by `--json` and `--plan`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct ConversionReport {
    /// Report schema.
    pub schema: String,
    /// Whether the command completed without error.
    pub ok: bool,
    /// Canonical configuration hash.
    pub config_hash: String,
    /// Model digest hex.
    pub model_digest: String,
    /// Bundle id.
    pub bundle_id: String,
    /// Published bundle path, absent in plan mode.
    pub bundle_path: Option<String>,
    /// Alignment.
    pub alignment: u64,
    /// Whether groups are required to cover every channel.
    pub complete: bool,
    /// Payload length in bytes.
    pub payload_bytes: u64,
    /// Shard count.
    pub shard_count: u64,
    /// Peak in-flight tensor bytes used by this run.
    pub peak_inflight_bytes: u64,
    /// Configured in-flight ceiling.
    pub max_inflight_bytes: u64,
    /// Sorted transform ids.
    pub transforms: Vec<String>,
    /// Shard rows.
    pub shards: Vec<ShardReport>,
}

/// Plan a conversion without reading tensor data.
///
/// # Errors
///
/// Rejects invalid options, missing/inconsistent FFN maps, invalid groupings and
/// size overflow.
pub fn plan_conversion(
    source: &impl TensorSource,
    discovered: &[DiscoveredFfn],
    spec: &GroupSpec,
    model: ModelDigest,
    options: &ConversionOptions,
) -> Result<ConversionPlan> {
    options.validate()?;
    if discovered.is_empty() {
        return Err(ConvertError::NoFfnTensors);
    }
    let mut layers = Vec::with_capacity(discovered.len());
    let mut shards = Vec::new();
    let mut transforms: Vec<String> = vec![
        TRANSFORM_DOWN.to_owned(),
        TRANSFORM_GATE.to_owned(),
        TRANSFORM_UP.to_owned(),
    ];
    for item in discovered {
        let dims = validate_ffn_map(source, &item.map)?;
        let groups = resolve_groups(spec, dims.intermediate, options.complete)?;
        for group in &groups {
            let width = group.indices.len() as u64;
            let elements = width
                .checked_mul(dims.hidden)
                .ok_or_else(|| ConvertError::plan("group element count overflows u64"))?;
            let byte_len = elements
                .checked_mul(4)
                .ok_or_else(|| ConvertError::plan("group byte length overflows u64"))?;
            for (role, transform_id, storage_order) in [
                (
                    TensorRole::MlpGate,
                    TRANSFORM_GATE.to_owned(),
                    StorageOrder::RowMajor,
                ),
                (
                    TensorRole::MlpUp,
                    TRANSFORM_UP.to_owned(),
                    StorageOrder::RowMajor,
                ),
                (
                    TensorRole::MlpDown,
                    TRANSFORM_DOWN.to_owned(),
                    StorageOrder::Transposed { axes: vec![1, 0] },
                ),
            ] {
                shards.push(PlannedShard {
                    layer: item.layer,
                    group_index: group.index,
                    role,
                    channels: group.channels.clone(),
                    channel_axis: 0,
                    shape: vec![width, dims.hidden],
                    storage_order,
                    transform_id,
                    group_channels: group.indices.clone(),
                    byte_len,
                });
            }
        }
        layers.push(LayerPlan {
            layer: item.layer,
            prefix: item.prefix.clone(),
            map: item.map.clone(),
            dims,
            groups,
        });
    }
    transforms.sort();
    transforms.dedup();
    let mut payload_len = 0_u64;
    for shard in &shards {
        payload_len = align_up(payload_len, options.alignment)?;
        payload_len = payload_len
            .checked_add(shard.byte_len)
            .ok_or_else(|| ConvertError::plan("payload length overflows u64"))?;
    }
    payload_len = align_up(payload_len, options.alignment)?;
    let plan_without_id = ConversionPlan {
        model,
        layers,
        shards,
        transforms,
        bundle_id: String::new(),
        alignment: options.alignment,
        complete: options.complete,
        payload_len,
    };
    let bundle_id = match &options.bundle_id {
        Some(id) => id.clone(),
        None => derive_bundle_id(&plan_without_id)?,
    };
    let mut plan = plan_without_id;
    plan.bundle_id = bundle_id;
    validate_bundle_id(&plan.bundle_id)?;
    Ok(plan)
}

fn derive_bundle_id(plan: &ConversionPlan) -> Result<String> {
    let mut builder = Sha256Builder::new();
    builder.update(plan.model.digest().as_bytes());
    for shard in &plan.shards {
        builder.update(shard.role.to_string().as_bytes());
        builder.update(b"\0");
        builder.update(format!("{:?}", shard.layer).as_bytes());
        builder.update(format!("{:?}", shard.channels).as_bytes());
        builder.update(format!("{:?}", shard.shape).as_bytes());
        builder.update(shard.transform_id.as_bytes());
        builder.update(b"\0");
    }
    let digest = builder.finish().to_hex();
    let model = plan.model.digest().to_hex();
    Ok(format!("swiglu-{}-{}", &model[..12], &digest[..12]))
}

fn validate_bundle_id(id: &str) -> Result<()> {
    if id.is_empty() || id.len() > 128 || id.starts_with('.') {
        return Err(ConvertError::plan(
            "bundle id must be 1..=128 bytes and must not start with '.'",
        ));
    }
    if !id
        .bytes()
        .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-'))
    {
        return Err(ConvertError::plan(
            "bundle id must contain only A-Za-z0-9._-",
        ));
    }
    Ok(())
}

/// Convert and publish a bundle.
///
/// # Errors
///
/// Propagates planning, reading, spool, manifest and publication failures. On
/// every failure path the spool file is removed and no final bundle is created.
pub fn convert(
    source: &mut impl TensorSource,
    plan: &ConversionPlan,
    out_dir: &Path,
    options: &ConversionOptions,
) -> Result<(PublishedBundle, ConversionReport)> {
    options.validate()?;
    std::fs::create_dir_all(out_dir)?;
    let mut budget = MemoryBudget::new(options.max_inflight_bytes)?;
    let mut spool = SpoolWriter::create(out_dir)?;
    let mut written: Vec<WrittenShard> = Vec::with_capacity(plan.shards.len());
    for layer in &plan.layers {
        for group in &layer.groups {
            let gate = read_rows(
                source,
                &layer.map.gate,
                &group.indices,
                layer.dims.hidden,
                &mut budget,
            )?;
            let (offset, length, sha256) = spool.write_extent(&gate, options.alignment)?;
            written.push(WrittenShard::new(
                layer,
                group,
                TensorRole::MlpGate,
                offset,
                length,
                sha256,
            ));
            budget.release((gate.len() as u64) * 4)?;

            let up = read_rows(
                source,
                &layer.map.up,
                &group.indices,
                layer.dims.hidden,
                &mut budget,
            )?;
            let (offset, length, sha256) = spool.write_extent(&up, options.alignment)?;
            written.push(WrittenShard::new(
                layer,
                group,
                TensorRole::MlpUp,
                offset,
                length,
                sha256,
            ));
            budget.release((up.len() as u64) * 4)?;

            let down = read_down_physical(
                source,
                &layer.map.down,
                &group.indices,
                layer.dims.hidden,
                &mut budget,
            )?;
            let (offset, length, sha256) = spool.write_extent(&down, options.alignment)?;
            written.push(WrittenShard::new(
                layer,
                group,
                TensorRole::MlpDown,
                offset,
                length,
                sha256,
            ));
            budget.release((down.len() as u64) * 4)?;
        }
    }
    let payload_len = spool.finish(options.alignment)?;
    if payload_len != plan.payload_len {
        return Err(ConvertError::plan(format!(
            "actual payload {payload_len} differs from planned {}",
            plan.payload_len
        )));
    }
    let manifest = build_manifest(plan, &written, payload_len, options)?;
    let mut reader = std::fs::File::open(spool.path())?;
    let write_options = WriteOptions {
        force: options.force,
        verify_before_publish: true,
        publish_sidecar: true,
        ..WriteOptions::default()
    };
    let published = publish_bundle_from_reader(out_dir, &manifest, &mut reader, &write_options)?;
    let report = report_for(
        plan,
        &written,
        payload_len,
        budget.peak(),
        options,
        Some(&published),
    )?;
    Ok((published, report))
}

struct WrittenShard {
    layer: Option<u32>,
    group_index: usize,
    role: TensorRole,
    channels: ChannelSet,
    channel_axis: u32,
    shape: Vec<u64>,
    storage_order: StorageOrder,
    transform_id: String,
    offset: u64,
    length: u64,
    sha256: Sha256Digest,
}

impl WrittenShard {
    fn new(
        layer: &LayerPlan,
        group: &Group,
        role: TensorRole,
        offset: u64,
        length: u64,
        sha256: Sha256Digest,
    ) -> Self {
        let (storage_order, transform_id) = match role {
            TensorRole::MlpGate => (StorageOrder::RowMajor, TRANSFORM_GATE),
            TensorRole::MlpUp => (StorageOrder::RowMajor, TRANSFORM_UP),
            _ => (
                StorageOrder::Transposed { axes: vec![1, 0] },
                TRANSFORM_DOWN,
            ),
        };
        Self {
            layer: layer.layer,
            group_index: group.index,
            role,
            channels: group.channels.clone(),
            channel_axis: 0,
            shape: vec![group.indices.len() as u64, layer.dims.hidden],
            storage_order,
            transform_id: transform_id.to_owned(),
            offset,
            length,
            sha256,
        }
    }
}

fn build_manifest(
    plan: &ConversionPlan,
    written: &[WrittenShard],
    payload_len: u64,
    options: &ConversionOptions,
) -> Result<Manifest> {
    let compatibility_id = if plan.complete {
        "swiglu-f32-complete-v1".to_owned()
    } else {
        "swiglu-f32-partial-v1".to_owned()
    };
    let mut ids: BTreeMap<(Option<u32>, usize, String), ShardId> = BTreeMap::new();
    let mut records = Vec::with_capacity(written.len());
    for shard in written {
        let identity = ShardIdentity {
            format_version: FORMAT_VERSION,
            model: plan.model.clone(),
            role: shard.role.clone(),
            layer: shard.layer,
            channels: shard.channels.clone(),
            channel_axis: shard.channel_axis,
            dtype: DType::F32,
            quant: None,
            transform_id: shard.transform_id.clone(),
        };
        let shard_id = ShardId::derive(&identity)?;
        ids.insert(
            (shard.layer, shard.group_index, shard.role.to_string()),
            shard_id,
        );
        records.push(ShardRecord {
            shard_id,
            role: shard.role.clone(),
            layer: shard.layer,
            channels: shard.channels.clone(),
            channel_axis: shard.channel_axis,
            shape: shard.shape.clone(),
            storage_order: shard.storage_order.clone(),
            dtype: DType::F32,
            quant: None,
            offset: shard.offset,
            length: shard.length,
            sha256: shard.sha256,
            alignment: options.alignment,
            compatibility_id: compatibility_id.clone(),
            transform_id: shard.transform_id.clone(),
            co_access: Vec::new(),
            depends_on: Vec::new(),
            calibration: None,
        });
    }
    for (shard, record) in written.iter().zip(records.iter_mut()) {
        let mut links = Vec::new();
        for role in [TensorRole::MlpGate, TensorRole::MlpUp, TensorRole::MlpDown] {
            if role == record.role {
                continue;
            }
            if let Some(id) = ids.get(&(shard.layer, shard.group_index, role.to_string())) {
                links.push(*id);
            }
        }
        links.sort_unstable();
        links.dedup();
        record.co_access = links;
    }
    records.sort_by_key(|record| record.shard_id);
    let manifest = Manifest {
        schema: MANIFEST_SCHEMA.to_owned(),
        schema_version: MANIFEST_SCHEMA_VERSION,
        bundle_id: plan.bundle_id.clone(),
        model: plan.model.clone(),
        transforms: plan.transforms.clone(),
        alignment: options.alignment,
        payload_len,
        created_by: options.created_by.clone(),
        shards: records,
    };
    manifest.validate()?;
    Ok(manifest)
}

pub(crate) fn read_rows(
    source: &mut impl TensorSource,
    key: &str,
    channels: &[u64],
    hidden: u64,
    budget: &mut MemoryBudget,
) -> Result<Vec<f32>> {
    let width = channels.len() as u64;
    let out_bytes = width
        .checked_mul(hidden)
        .and_then(|value| value.checked_mul(4))
        .ok_or_else(|| ConvertError::plan("row group byte length overflows"))?;
    budget.reserve(out_bytes)?;
    let row_bytes = hidden
        .checked_mul(4)
        .ok_or_else(|| ConvertError::plan("row byte length overflows"))?;
    let contiguous = is_contiguous(channels);
    if contiguous {
        let start = channels.first().copied().unwrap_or(0);
        let end = start + width;
        // The reader holds the output plus one row-sized scratch buffer.
        budget.reserve(row_bytes)?;
        let result = source.read_region_f32(key, start..end, 0..hidden);
        budget.release(row_bytes)?;
        return result;
    }
    let mut out: Vec<f32> = Vec::new();
    crate::tensor_source::reserve_elements(&mut out, width * hidden)?;
    for channel in channels {
        // A gathered row read allocates its own output and scratch row.
        let double_row = row_bytes
            .checked_mul(2)
            .ok_or_else(|| ConvertError::plan("row reservation overflows"))?;
        budget.reserve(double_row)?;
        let row = source.read_region_f32(key, *channel..(*channel + 1), 0..hidden)?;
        budget.release(double_row)?;
        if row.len() as u64 != hidden {
            return Err(ConvertError::inconsistent(
                "row read returned the wrong length",
            ));
        }
        out.extend_from_slice(&row);
    }
    Ok(out)
}

pub(crate) fn read_down_physical(
    source: &mut impl TensorSource,
    key: &str,
    channels: &[u64],
    hidden: u64,
    budget: &mut MemoryBudget,
) -> Result<Vec<f32>> {
    let width = channels.len() as u64;
    let out_bytes = width
        .checked_mul(hidden)
        .and_then(|value| value.checked_mul(4))
        .ok_or_else(|| ConvertError::plan("down group byte length overflows"))?;
    budget.reserve(out_bytes)?;
    let row_bytes = width
        .checked_mul(4)
        .ok_or_else(|| ConvertError::plan("down row byte length overflows"))?;
    let contiguous = is_contiguous(channels);
    if contiguous {
        let start = channels.first().copied().unwrap_or(0);
        let end = start + width;
        // The reader holds the output plus one row-sized scratch buffer.
        budget.reserve(row_bytes)?;
        let result = source.read_region_f32(key, 0..hidden, start..end);
        budget.release(row_bytes)?;
        return result;
    }
    let column_bytes = hidden
        .checked_mul(4)
        .ok_or_else(|| ConvertError::plan("column byte length overflows"))?;
    let mut out: Vec<f32> = Vec::new();
    crate::tensor_source::reserve_elements(&mut out, width * hidden)?;
    out.resize(
        usize::try_from(width * hidden).map_err(|_| ConvertError::plan("size overflow"))?,
        0.0,
    );
    for (j, channel) in channels.iter().enumerate() {
        let double_column = column_bytes
            .checked_mul(2)
            .ok_or_else(|| ConvertError::plan("column reservation overflows"))?;
        budget.reserve(double_column)?;
        let column = source.read_region_f32(key, 0..hidden, *channel..(*channel + 1))?;
        budget.release(double_column)?;
        if column.len() as u64 != hidden {
            return Err(ConvertError::inconsistent(
                "column read returned the wrong length",
            ));
        }
        for (h, value) in column.iter().enumerate() {
            out[h * channels.len() + j] = *value;
        }
    }
    Ok(out)
}

fn is_contiguous(channels: &[u64]) -> bool {
    channels.first().is_some_and(|first| {
        channels
            .iter()
            .enumerate()
            .all(|(offset, channel)| *channel == first + offset as u64)
    })
}

fn report_for(
    plan: &ConversionPlan,
    written: &[WrittenShard],
    payload_len: u64,
    peak: u64,
    options: &ConversionOptions,
    published: Option<&PublishedBundle>,
) -> Result<ConversionReport> {
    let rows = written
        .iter()
        .map(|shard| {
            let identity = ShardIdentity {
                format_version: FORMAT_VERSION,
                model: plan.model.clone(),
                role: shard.role.clone(),
                layer: shard.layer,
                channels: shard.channels.clone(),
                channel_axis: shard.channel_axis,
                dtype: DType::F32,
                quant: None,
                transform_id: shard.transform_id.clone(),
            };
            let id = ShardId::derive(&identity)?;
            Ok(ShardReport {
                short_id: id.short(),
                role: shard.role.to_string(),
                layer: shard.layer,
                group: shard.group_index,
                channels: channels_text(&shard.channels),
                shape: shard
                    .shape
                    .iter()
                    .map(u64::to_string)
                    .collect::<Vec<_>>()
                    .join("x"),
                storage_order: order_text(&shard.storage_order),
                transform_id: shard.transform_id.clone(),
                bytes: shard.length,
                sha256: shard.sha256.to_hex(),
            })
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(ConversionReport {
        schema: "selvedge/convert-report/v1".to_owned(),
        ok: true,
        config_hash: options.config_hash.clone(),
        model_digest: plan.model.digest().to_hex(),
        bundle_id: plan.bundle_id.clone(),
        bundle_path: published.map(|bundle| bundle.path.display().to_string()),
        alignment: options.alignment,
        complete: plan.complete,
        payload_bytes: payload_len,
        shard_count: rows.len() as u64,
        peak_inflight_bytes: peak,
        max_inflight_bytes: options.max_inflight_bytes,
        transforms: plan.transforms.clone(),
        shards: rows,
    })
}

/// Report a plan without reading data.
///
/// # Errors
///
/// Propagates shard-id derivation.
pub fn report_plan(plan: &ConversionPlan, options: &ConversionOptions) -> Result<ConversionReport> {
    let shards: Vec<WrittenShard> = plan
        .shards
        .iter()
        .map(|shard| WrittenShard {
            layer: shard.layer,
            group_index: shard.group_index,
            role: shard.role.clone(),
            channels: shard.channels.clone(),
            channel_axis: shard.channel_axis,
            shape: shard.shape.clone(),
            storage_order: shard.storage_order.clone(),
            transform_id: shard.transform_id.clone(),
            offset: 0,
            length: shard.byte_len,
            sha256: Sha256Digest::of(b""),
        })
        .collect();
    report_for(plan, &shards, plan.payload_len, 0, options, None)
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

fn order_text(order: &StorageOrder) -> String {
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

static SPOOL_COUNTER: AtomicU64 = AtomicU64::new(0);

struct SpoolWriter {
    file: std::fs::File,
    path: PathBuf,
    length: u64,
}

impl SpoolWriter {
    fn create(dir: &Path) -> Result<Self> {
        for _ in 0..64 {
            let counter = SPOOL_COUNTER.fetch_add(1, Ordering::Relaxed);
            let path = dir.join(format!(
                ".selvedge-spool-{}-{}",
                std::process::id(),
                counter
            ));
            match std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&path)
            {
                Ok(file) => {
                    return Ok(Self {
                        file,
                        path,
                        length: 0,
                    });
                }
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(error) => return Err(ConvertError::Io(error)),
            }
        }
        Err(ConvertError::Io(std::io::Error::new(
            std::io::ErrorKind::AlreadyExists,
            "cannot allocate a unique spool file name",
        )))
    }

    fn path(&self) -> &Path {
        &self.path
    }

    fn write_extent(&mut self, values: &[f32], alignment: u64) -> Result<(u64, u64, Sha256Digest)> {
        let bytes_len = (values.len() as u64)
            .checked_mul(4)
            .ok_or_else(|| ConvertError::plan("extent byte length overflows"))?;
        let offset = align_up(self.length, alignment)?;
        Self::write_zeros(&mut self.file, offset - self.length)?;
        let mut hasher = Sha256Builder::new();
        let mut buffer = [0_u8; 64 * 1024];
        let mut used = 0_usize;
        for value in values {
            if used + 4 > buffer.len() {
                self.file.write_all(&buffer[..used])?;
                hasher.update(&buffer[..used]);
                used = 0;
            }
            buffer[used..used + 4].copy_from_slice(&value.to_le_bytes());
            used += 4;
        }
        if used > 0 {
            self.file.write_all(&buffer[..used])?;
            hasher.update(&buffer[..used]);
        }
        self.length = offset + bytes_len;
        Ok((offset, bytes_len, hasher.finish()))
    }

    fn finish(&mut self, alignment: u64) -> Result<u64> {
        let payload_len = align_up(self.length, alignment)?;
        Self::write_zeros(&mut self.file, payload_len - self.length)?;
        self.length = payload_len;
        self.file.sync_all()?;
        Ok(payload_len)
    }

    fn write_zeros(file: &mut std::fs::File, mut count: u64) -> Result<()> {
        let zeros = [0_u8; 8192];
        while count > 0 {
            let chunk = usize::try_from(count.min(zeros.len() as u64))
                .map_err(|_| ConvertError::plan("zero-padding chunk overflow"))?;
            file.write_all(&zeros[..chunk])?;
            count -= chunk as u64;
        }
        Ok(())
    }
}

impl Drop for SpoolWriter {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

/// Source model digest from an opened checkpoint.
///
/// # Errors
///
/// Propagates I/O and digest failures.
pub fn model_digest_for_path(path: &Path) -> Result<ModelDigest> {
    let mut files = Vec::new();
    let metadata = std::fs::metadata(path)?;
    if metadata.is_file() {
        let name = path
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .ok_or_else(|| ConvertError::plan("checkpoint path has no file name"))?;
        files.push((name, path.to_path_buf()));
    } else if metadata.is_dir() {
        collect_files(path, path, &mut files)?;
    } else {
        return Err(ConvertError::plan(
            "checkpoint path is not a file or directory",
        ));
    }
    files.sort_by(|left, right| left.0.cmp(&right.0));
    let mut checkpoint_files = Vec::with_capacity(files.len());
    for (relative, file_path) in files {
        let length = std::fs::metadata(&file_path)?.len();
        let mut file = std::fs::File::open(&file_path)?;
        let mut builder = Sha256Builder::new();
        let mut buffer = vec![0_u8; 64 * 1024];
        loop {
            let read = std::io::Read::read(&mut file, &mut buffer)?;
            if read == 0 {
                break;
            }
            builder.update(&buffer[..read]);
        }
        checkpoint_files.push(selvedge_core::CheckpointFile {
            path: relative,
            sha256: builder.finish(),
            length,
        });
    }
    ModelDigest::of_fileset(&checkpoint_files).map_err(ConvertError::from)
}

fn collect_files(root: &Path, dir: &Path, out: &mut Vec<(String, PathBuf)>) -> Result<()> {
    let mut entries: Vec<_> = std::fs::read_dir(dir)?.collect::<core::result::Result<_, _>>()?;
    entries.sort_by_key(std::fs::DirEntry::file_name);
    for entry in entries {
        let file_type = entry.file_type()?;
        if file_type.is_dir() {
            collect_files(root, &entry.path(), out)?;
        } else if file_type.is_file()
            && entry
                .path()
                .extension()
                .is_some_and(|ext| ext == "safetensors")
        {
            let path = entry.path();
            let relative = path.strip_prefix(root).map_err(|_| {
                ConvertError::plan(format!("{} is outside {}", path.display(), root.display()))
            })?;
            let relative = relative
                .components()
                .map(|component| component.as_os_str().to_string_lossy())
                .collect::<Vec<_>>()
                .join("/");
            out.push((relative, path));
        }
    }
    Ok(())
}
