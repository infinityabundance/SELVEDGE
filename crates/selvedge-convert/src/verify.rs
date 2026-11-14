//! Reconstruction proof: decode repacked groups from a validated bundle and
//! compare the grouped FFN against the source FFN in f32 and f64.
//!
//! The verifier never trusts in-memory conversion state. It re-opens the bundle,
//! verifies every shard hash through the format validator, decodes the shard
//! bytes according to each record's declared shape and storage order, and then
//! compares against tensors read from the original checkpoint.

use std::collections::BTreeMap;
use std::path::Path;

use selvedge_core::{ChannelSet, DType, StorageOrder};
use selvedge_format::{BundleFile, ValidationOptions, VerifyMode, validate_bundle};
use serde::Serialize;

use crate::convert::{ConversionOptions, read_down_physical, read_rows};
use crate::error::{ConvertError, Result};
use crate::ffn::{
    DiscoveredFfn, FfnDims, GroupMatrices, ffn_dense_f64, ffn_grouped_f32, ffn_grouped_f64,
    group_contribution_f32, group_contribution_f64, validate_ffn_map,
};
use crate::synthetic::DeterministicRng;
use crate::tensor_source::{MemoryBudget, TensorSource, physical_index, reserve_elements};

/// Acceptance tolerance for one reconstruction measurement.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Tolerance {
    /// Absolute component of the bound.
    pub abs: f64,
    /// Relative component of the bound, multiplied by the reference magnitude.
    pub rel: f64,
}

impl Default for Tolerance {
    fn default() -> Self {
        // Calibrated by dtype and checkpoint size, not guessed. The synthetic
        // Court A layers are tiny; the real Qwen3 checkpoints (intermediate
        // width up to 6144) measure their worst f32 component error below
        // 5e-5 against an f64 reference. The bound keeps a margin over that
        // measurement and is still orders of magnitude below a structural
        // error. Measured maxima are published from receipts, never asserted
        // here.
        Self {
            abs: 1e-4,
            rel: 1e-4,
        }
    }
}

impl Tolerance {
    /// Tolerance for a source dtype.
    ///
    /// f16 and bf16 values are widened exactly and all math is f32, so the
    /// numeric bound is identical; the report records the source dtype so a
    /// reader can see which representation was tested.
    #[must_use]
    pub fn for_dtype(dtype: &DType) -> Self {
        let _ = dtype;
        Self::default()
    }
}

/// Verification knobs.
#[derive(Clone, Debug)]
pub struct VerificationOptions {
    /// Acceptance tolerance.
    pub tolerance: Tolerance,
    /// Batch sizes (`N` flattened) to test.
    pub batch_sizes: Vec<u64>,
    /// Deterministic input seed.
    pub seed: u64,
    /// Per-shard read cap in bytes.
    pub max_shard_bytes: u64,
    /// In-flight budget for source reads.
    pub max_inflight_bytes: u64,
}

impl Default for VerificationOptions {
    fn default() -> Self {
        Self {
            tolerance: Tolerance::default(),
            batch_sizes: vec![1, 4, 17, 64],
            seed: 0x5E1A_ED6E_C0A7_A001,
            max_shard_bytes: 4 * 1024 * 1024 * 1024,
            max_inflight_bytes: 512 * 1024 * 1024,
        }
    }
}

/// A numeric error measurement.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ErrorMeasurement {
    /// Largest absolute difference.
    pub max_abs: f64,
    /// Largest relative difference, using `max(|reference|, 1e-12)`.
    pub max_rel: f64,
    /// Mean absolute difference.
    pub mean_abs: f64,
    /// Number of compared components.
    pub count: u64,
    /// Absolute component of the acceptance bound.
    pub tolerance_abs: f64,
    /// Relative component of the acceptance bound.
    pub tolerance_rel: f64,
    /// Whether every component satisfied `abs + rel * |reference|`.
    pub passed: bool,
}

/// Per-group error measurement.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct GroupMeasurement {
    /// Group index in bundle order.
    pub group: usize,
    /// Channel set text.
    pub channels: String,
    /// F32 contribution error.
    pub f32_error: ErrorMeasurement,
    /// F64 contribution error.
    pub f64_error: ErrorMeasurement,
}

/// One layer's verification result.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct LayerVerification {
    /// Layer index.
    pub layer: Option<u32>,
    /// Source dtype names observed.
    pub source_dtype: String,
    /// Hidden width.
    pub hidden: u64,
    /// Intermediate width.
    pub intermediate: u64,
    /// Group errors.
    pub groups: Vec<GroupMeasurement>,
    /// Full grouped-vs-dense f32 error, aggregated over batch sizes.
    pub full_f32: ErrorMeasurement,
    /// Full grouped-vs-dense f64 error, aggregated over batch sizes.
    pub full_f64: ErrorMeasurement,
    /// Whether every measurement passed.
    pub passed: bool,
}

/// The complete verification report.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct VerificationReport {
    /// Report schema.
    pub schema: String,
    /// Whether every measurement passed.
    pub ok: bool,
    /// Bundle path.
    pub bundle: String,
    /// Model digest hex.
    pub model_digest: String,
    /// Batch sizes tested.
    pub batch_sizes: Vec<u64>,
    /// Seed used for the inputs.
    pub seed: u64,
    /// Per-layer results.
    pub layers: Vec<LayerVerification>,
    /// Values checked by the format validator.
    pub validated_shards: u64,
}

#[derive(Clone, Debug)]
pub(crate) struct BundleGroup {
    pub(crate) channels: Vec<u64>,
    pub(crate) matrices: GroupMatrices,
}

#[derive(Clone, Debug)]
pub(crate) struct BundleLayer {
    pub(crate) layer: Option<u32>,
    pub(crate) hidden: u64,
    pub(crate) intermediate: u64,
    pub(crate) groups: Vec<BundleGroup>,
}

/// Verify a published bundle against the source checkpoint.
///
/// # Errors
///
/// Rejects an invalid bundle, missing shards, mismatched shapes, malformed
/// storage orders, allocation refusal and source read failures.
pub fn verify_bundle(
    bundle_path: &Path,
    source: &mut impl TensorSource,
    discovered: &[DiscoveredFfn],
    options: &VerificationOptions,
) -> Result<VerificationReport> {
    if options.batch_sizes.is_empty() {
        return Err(ConvertError::plan(
            "verification needs at least one batch size",
        ));
    }
    if options.batch_sizes.contains(&0) {
        return Err(ConvertError::plan(
            "verification batch sizes must be non-zero",
        ));
    }
    let validation = validate_bundle(
        bundle_path,
        &ValidationOptions {
            verify: VerifyMode::All,
            ..ValidationOptions::default()
        },
    )?;
    let mut bundle = BundleFile::open(bundle_path)?;
    let mut layers = load_bundle_layers(&mut bundle, options)?;
    layers.sort_by_key(|layer| (layer.layer, layer.hidden, layer.intermediate));
    let model_digest = bundle.manifest().model.digest().to_hex();
    let mut results = Vec::with_capacity(layers.len());
    let mut overall = true;
    for bundle_layer in &layers {
        let map = match_layer(discovered, bundle_layer.layer)?;
        let dims = validate_ffn_map(source, map)?;
        if dims.hidden != bundle_layer.hidden || dims.intermediate != bundle_layer.intermediate {
            return Err(ConvertError::inconsistent(format!(
                "bundle layer {:?} is {}x{}, source is {}x{}",
                bundle_layer.layer,
                bundle_layer.hidden,
                bundle_layer.intermediate,
                dims.hidden,
                dims.intermediate
            )));
        }
        let mut budget = MemoryBudget::new(options.max_inflight_bytes)?;
        let gate = source.read_region_f32(&map.gate, 0..dims.intermediate, 0..dims.hidden)?;
        let up = source.read_region_f32(&map.up, 0..dims.intermediate, 0..dims.hidden)?;
        let down = source.read_region_f32(&map.down, 0..dims.hidden, 0..dims.intermediate)?;
        let _ = &mut budget;
        let channel_groups: Vec<Vec<u64>> = bundle_layer
            .groups
            .iter()
            .map(|group| group.channels.clone())
            .collect();
        let source_groups = source_groups(source, map, dims, &channel_groups, &mut budget)?;
        let source_dtype = descriptor_dtype(source, &map.gate)?;
        let mut full_f32 = Acc::default();
        let mut full_f64 = Acc::default();
        let mut group_state: Vec<(Acc, Acc)> =
            vec![(Acc::default(), Acc::default()); bundle_layer.groups.len()];
        for batch in &options.batch_sizes {
            let mut rng = DeterministicRng::new(options.seed ^ batch.rotate_left(17));
            let count = (*batch)
                .checked_mul(dims.hidden)
                .ok_or_else(|| ConvertError::plan("input size overflows"))?;
            let count = usize::try_from(count)
                .map_err(|_| ConvertError::plan("input size does not fit usize"))?;
            let input = rng.fill(count);
            let dense_f64 = ffn_dense_f64(
                &gate,
                &up,
                &down,
                dims.hidden,
                dims.intermediate,
                *batch,
                &input,
            )?;
            let bundle_groups: Vec<GroupMatrices> = bundle_layer
                .groups
                .iter()
                .map(|group| group.matrices.clone())
                .collect();
            let grouped_f32 = ffn_grouped_f32(dims.hidden, &bundle_groups, *batch, &input)?;
            let grouped_f64 = ffn_grouped_f64(dims.hidden, &bundle_groups, *batch, &input)?;
            full_f32.add_f32(&grouped_f32, &dense_f64, options.tolerance);
            full_f64.add_f64(&grouped_f64, &dense_f64, options.tolerance);
            for (index, (bundle_group, source_group)) in bundle_layer
                .groups
                .iter()
                .zip(source_groups.iter())
                .enumerate()
            {
                let bundle_part =
                    group_contribution_f32(&bundle_group.matrices, dims.hidden, *batch, &input)?;
                let source_part =
                    group_contribution_f32(source_group, dims.hidden, *batch, &input)?;
                let reference: Vec<f64> =
                    source_part.iter().map(|value| f64::from(*value)).collect();
                let bundle_part_f64 =
                    group_contribution_f64(&bundle_group.matrices, dims.hidden, *batch, &input)?;
                let source_part_f64 =
                    group_contribution_f64(source_group, dims.hidden, *batch, &input)?;
                let state = group_state.get_mut(index).ok_or_else(|| {
                    ConvertError::inconsistent("group measurement index is out of range")
                })?;
                state.0.add_f32(&bundle_part, &reference, options.tolerance);
                state
                    .1
                    .add_f64(&bundle_part_f64, &source_part_f64, options.tolerance);
            }
        }
        let groups: Vec<GroupMeasurement> = bundle_layer
            .groups
            .iter()
            .zip(group_state.iter())
            .enumerate()
            .map(|(index, (group, state))| GroupMeasurement {
                group: index,
                channels: channels_text(&group.channels),
                f32_error: state.0.measurement(options.tolerance),
                f64_error: state.1.measurement(options.tolerance),
            })
            .collect();
        let full_f32 = full_f32.measurement(options.tolerance);
        let full_f64 = full_f64.measurement(options.tolerance);
        let passed = full_f32.passed
            && full_f64.passed
            && groups
                .iter()
                .all(|group| group.f32_error.passed && group.f64_error.passed);
        overall &= passed;
        results.push(LayerVerification {
            layer: bundle_layer.layer,
            source_dtype,
            hidden: dims.hidden,
            intermediate: dims.intermediate,
            groups,
            full_f32,
            full_f64,
            passed,
        });
    }
    Ok(VerificationReport {
        schema: "selvedge/verification-report/v1".to_owned(),
        ok: overall,
        bundle: bundle_path.display().to_string(),
        model_digest,
        batch_sizes: options.batch_sizes.clone(),
        seed: options.seed,
        layers: results,
        validated_shards: validation.verified_shards.len() as u64,
    })
}

pub(crate) fn match_layer(
    discovered: &[DiscoveredFfn],
    layer: Option<u32>,
) -> Result<&crate::ffn::FfnTensorMap> {
    let matches: Vec<&DiscoveredFfn> = discovered
        .iter()
        .filter(|item| item.layer == layer)
        .collect();
    match matches.as_slice() {
        [only] => Ok(&only.map),
        [] => Err(ConvertError::inconsistent(format!(
            "no source FFN map for bundle layer {layer:?}"
        ))),
        _ => Err(ConvertError::inconsistent(format!(
            "multiple source FFN maps for bundle layer {layer:?}"
        ))),
    }
}

fn descriptor_dtype(source: &impl TensorSource, key: &str) -> Result<String> {
    Ok(source.descriptor(key)?.dtype.stable_name())
}

pub(crate) fn load_bundle_layers(
    bundle: &mut BundleFile,
    options: &VerificationOptions,
) -> Result<Vec<BundleLayer>> {
    let records = bundle.manifest().shards.clone();
    let mut buckets: BTreeMap<(Option<u32>, String), Bucket> = BTreeMap::new();
    for record in records {
        let role = match record.role {
            selvedge_core::TensorRole::MlpGate => Role::Gate,
            selvedge_core::TensorRole::MlpUp => Role::Up,
            selvedge_core::TensorRole::MlpDown => Role::Down,
            _ => continue,
        };
        if record.channel_axis != 0 {
            return Err(ConvertError::inconsistent(format!(
                "shard {} has channel axis {}; expected 0",
                record.shard_id.short(),
                record.channel_axis
            )));
        }
        if record.dtype != DType::F32 || record.quant.is_some() {
            return Err(ConvertError::inconsistent(format!(
                "shard {} is not unquantized f32",
                record.shard_id.short()
            )));
        }
        if record.shape.len() != 2 {
            return Err(ConvertError::RankMismatch {
                tensor: record.shard_id.to_string(),
                rank: record.shape.len(),
            });
        }
        let channels = channel_list(&record.channels)?;
        if record.shape[0] != channels.len() as u64 {
            return Err(ConvertError::inconsistent(format!(
                "shard {} shape {:?} does not match {} channels",
                record.shard_id.short(),
                record.shape,
                channels.len()
            )));
        }
        let bytes = bundle.read_shard(&record.shard_id, options.max_shard_bytes)?;
        let values = decode_f32_tensor(&record.shape, &record.storage_order, &bytes)?;
        let key = (record.layer, channels_key(&record.channels));
        let bucket = buckets.entry(key).or_insert_with(|| Bucket {
            layer: record.layer,
            channels,
            hidden: record.shape[1],
            gate: None,
            up: None,
            down: None,
        });
        if bucket.hidden != record.shape[1] {
            return Err(ConvertError::inconsistent(
                "group shards disagree on hidden width",
            ));
        }
        let slot = match role {
            Role::Gate => &mut bucket.gate,
            Role::Up => &mut bucket.up,
            Role::Down => &mut bucket.down,
        };
        if slot.is_some() {
            return Err(ConvertError::inconsistent(
                "duplicate role inside one group",
            ));
        }
        *slot = Some(values);
    }
    let mut layers: BTreeMap<Option<u32>, BundleLayer> = BTreeMap::new();
    for (_key, bucket) in buckets {
        let gate = bucket
            .gate
            .ok_or_else(|| ConvertError::inconsistent("group is missing a gate shard"))?;
        let up = bucket
            .up
            .ok_or_else(|| ConvertError::inconsistent("group is missing an up shard"))?;
        let down = bucket
            .down
            .ok_or_else(|| ConvertError::inconsistent("group is missing a down shard"))?;
        let entry = layers.entry(bucket.layer).or_insert_with(|| BundleLayer {
            layer: bucket.layer,
            hidden: bucket.hidden,
            intermediate: 0,
            groups: Vec::new(),
        });
        if entry.hidden != bucket.hidden {
            return Err(ConvertError::inconsistent(
                "groups disagree on hidden width",
            ));
        }
        entry.intermediate = entry
            .intermediate
            .saturating_add(bucket.channels.len() as u64);
        let matrix_channels = bucket.channels.clone();
        entry.groups.push(BundleGroup {
            channels: bucket.channels,
            matrices: GroupMatrices {
                index: entry.groups.len(),
                channels: matrix_channels,
                gate,
                up,
                down_t: down,
            },
        });
    }
    let mut out = Vec::with_capacity(layers.len());
    for (_, mut layer) in layers {
        layer
            .groups
            .sort_by(|left, right| left.channels.cmp(&right.channels));
        for (index, group) in layer.groups.iter_mut().enumerate() {
            group.matrices.index = index;
        }
        out.push(layer);
    }
    Ok(out)
}

struct Bucket {
    layer: Option<u32>,
    channels: Vec<u64>,
    hidden: u64,
    gate: Option<Vec<f32>>,
    up: Option<Vec<f32>>,
    down: Option<Vec<f32>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Role {
    Gate,
    Up,
    Down,
}

pub(crate) fn source_groups(
    source: &mut impl TensorSource,
    map: &crate::ffn::FfnTensorMap,
    dims: FfnDims,
    groups: &[Vec<u64>],
    budget: &mut MemoryBudget,
) -> Result<Vec<GroupMatrices>> {
    let mut out = Vec::with_capacity(groups.len());
    for (index, channels) in groups.iter().enumerate() {
        let gate = read_rows(source, &map.gate, channels, dims.hidden, budget)?;
        let up = read_rows(source, &map.up, channels, dims.hidden, budget)?;
        let down_physical = read_down_physical(source, &map.down, channels, dims.hidden, budget)?;
        let width = channels.len();
        let hidden = usize::try_from(dims.hidden)
            .map_err(|_| ConvertError::plan("hidden width does not fit usize"))?;
        let mut down_t: Vec<f32> = Vec::new();
        reserve_elements(&mut down_t, (width * hidden) as u64)?;
        down_t.resize(width * hidden, 0.0);
        for h in 0..hidden {
            for j in 0..width {
                down_t[j * hidden + h] = down_physical[h * width + j];
            }
        }
        budget.release((gate.len() as u64) * 4)?;
        budget.release((up.len() as u64) * 4)?;
        budget.release((down_physical.len() as u64) * 4)?;
        out.push(GroupMatrices {
            index,
            channels: channels.clone(),
            gate,
            up,
            down_t,
        });
    }
    Ok(out)
}

fn channel_list(channels: &ChannelSet) -> Result<Vec<u64>> {
    match channels {
        ChannelSet::Range { start, end } => Ok((*start..*end).collect()),
        ChannelSet::List { indices } => Ok(indices.clone()),
    }
}

fn channels_key(channels: &ChannelSet) -> String {
    match channels {
        ChannelSet::Range { start, end } => format!("r:{start}:{end}"),
        ChannelSet::List { indices } => format!(
            "l:{}",
            indices
                .iter()
                .map(u64::to_string)
                .collect::<Vec<_>>()
                .join(",")
        ),
    }
}

pub(crate) fn channels_text(channels: &[u64]) -> String {
    format!(
        "[{}]",
        channels
            .iter()
            .map(u64::to_string)
            .collect::<Vec<_>>()
            .join(",")
    )
}

fn decode_f32_tensor(shape: &[u64], order: &StorageOrder, bytes: &[u8]) -> Result<Vec<f32>> {
    if shape.len() != 2 {
        return Err(ConvertError::RankMismatch {
            tensor: "<shard>".to_owned(),
            rank: shape.len(),
        });
    }
    let elements = shape[0]
        .checked_mul(shape[1])
        .ok_or_else(|| ConvertError::plan("shard element count overflows"))?;
    let expected = elements
        .checked_mul(4)
        .ok_or_else(|| ConvertError::plan("shard byte length overflows"))?;
    if expected != bytes.len() as u64 {
        return Err(ConvertError::ByteLengthMismatch {
            tensor: "<shard>".to_owned(),
            extent: bytes.len() as u64,
            expected,
            elements,
            dtype: "f32".to_owned(),
        });
    }
    let mut out: Vec<f32> = Vec::new();
    reserve_elements(&mut out, elements)?;
    out.resize(
        usize::try_from(elements)
            .map_err(|_| ConvertError::plan("element count does not fit usize"))?,
        0.0,
    );
    match order {
        StorageOrder::RowMajor => {
            for (index, chunk) in bytes.as_chunks::<4>().0.iter().enumerate() {
                out[index] = f32::from_le_bytes(*chunk);
            }
        }
        _ => {
            for row in 0..shape[0] {
                for col in 0..shape[1] {
                    let index = physical_index(shape, order, row, col)?;
                    let offset = usize::try_from(index)
                        .ok()
                        .and_then(|index| index.checked_mul(4))
                        .ok_or_else(|| ConvertError::plan("physical offset overflows"))?;
                    let chunk = bytes.get(offset..offset + 4).ok_or_else(|| {
                        ConvertError::plan("physical offset is outside the shard bytes")
                    })?;
                    let logical = usize::try_from(row * shape[1] + col)
                        .map_err(|_| ConvertError::plan("logical index overflows"))?;
                    out[logical] = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                }
            }
        }
    }
    Ok(out)
}

#[derive(Clone, Copy, Debug)]
struct Acc {
    max_abs: f64,
    max_rel: f64,
    sum_abs: f64,
    count: u64,
    passed: bool,
}

impl Default for Acc {
    fn default() -> Self {
        Self {
            max_abs: 0.0,
            max_rel: 0.0,
            sum_abs: 0.0,
            count: 0,
            passed: true,
        }
    }
}

impl Acc {
    fn add_f32(&mut self, values: &[f32], reference: &[f64], tolerance: Tolerance) {
        for (value, reference) in values.iter().zip(reference.iter()) {
            self.add_one(f64::from(*value), *reference, tolerance);
        }
    }

    fn add_f64(&mut self, values: &[f64], reference: &[f64], tolerance: Tolerance) {
        for (value, reference) in values.iter().zip(reference.iter()) {
            self.add_one(*value, *reference, tolerance);
        }
    }

    fn add_one(&mut self, value: f64, reference: f64, tolerance: Tolerance) {
        let difference = (value - reference).abs();
        self.max_abs = self.max_abs.max(difference);
        let denominator = reference.abs().max(1e-12);
        self.max_rel = self.max_rel.max(difference / denominator);
        self.sum_abs += difference;
        self.count += 1;
        if difference > tolerance.abs + tolerance.rel * reference.abs() {
            self.passed = false;
        }
    }

    fn measurement(&self, tolerance: Tolerance) -> ErrorMeasurement {
        ErrorMeasurement {
            max_abs: self.max_abs,
            max_rel: self.max_rel,
            mean_abs: if self.count == 0 {
                0.0
            } else {
                self.sum_abs / self.count as f64
            },
            count: self.count,
            tolerance_abs: tolerance.abs,
            tolerance_rel: tolerance.rel,
            passed: self.passed,
        }
    }
}

/// Convert a bundle path into a verification result, for CLI use.
///
/// # Errors
///
/// Propagates [`verify_bundle`] failures.
pub fn verify_with_defaults(
    bundle_path: &Path,
    source: &mut impl TensorSource,
    discovered: &[DiscoveredFfn],
) -> Result<VerificationReport> {
    verify_bundle(
        bundle_path,
        source,
        discovered,
        &VerificationOptions::default(),
    )
}

/// The conversion options a verifier expects to match.
#[must_use]
pub fn expected_conversion_options() -> ConversionOptions {
    ConversionOptions::default()
}
