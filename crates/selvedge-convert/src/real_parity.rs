//! Real-checkpoint, real-activation parity for the grouped FFN.
//!
//! This is the Court A extension that replaces random inputs with activations
//! captured from the teacher itself. It answers three questions with measured
//! numbers:
//!
//! 1. does the grouped sum reconstructed from the validated bundle equal the
//!    dense FFN recomputed in f32 from the same checkpoint (`bundle_vs_dense`)?
//! 2. does that f32 recomputation equal the teacher's own f32 forward
//!    (`f32_vs_teacher_f32`)?
//! 3. does the same bundle reproduce the teacher's native bf16 output when fed
//!    the teacher's native bf16 inputs (`bf16_vs_teacher_bf16`)?
//!
//! The bundle stores f32, so it is an exact widening of the bf16 checkpoint.
//! That choice is deliberate: it lets the grouped path reproduce the
//! f32-equivalent computation, and it costs two extra bytes per FFN element
//! compared with bf16 storage. No bf16 bundle is claimed to meet a tighter
//! bound than the teacher's own output rounding.
//!
//! Reduction order is fixed: groups in bundle order, channels ascending inside
//! each group. Rows are independent and are split across threads, which changes
//! wall-clock but not the arithmetic of any row.

use std::path::Path;

use selvedge_core::DType;
use selvedge_format::{BundleFile, ValidationOptions, VerifyMode, validate_bundle};
use serde::Serialize;

use crate::error::{ConvertError, Result};
use crate::ffn::{GroupMatrices, discover_ffn_maps, silu_f32, validate_ffn_map};
use crate::safetensors::Safetensors;
use crate::tensor_source::{TensorSource, reserve_elements};
use crate::verify::{BundleLayer, load_bundle_layers, match_layer};

/// Absolute tolerance for the f32 paths.
pub const F32_PARITY_ABS: f64 = 1.0e-4;
/// Relative tolerance for the f32 paths.
pub const F32_PARITY_REL: f64 = 1.0e-4;
/// Extra absolute tolerance as a fraction of the reference RMS for f32 paths.
///
/// Near-zero output components are produced by cancellation of terms that are
/// thousands of times larger, so an elementwise relative test alone is
/// meaningless there. The scale term is the reference RMS of the compared
/// tensor.
pub const F32_PARITY_SCALE: f64 = 1.0e-3;
/// Absolute tolerance for the teacher's native bf16 output.
pub const BF16_PARITY_ABS: f64 = 0.05;
/// Relative tolerance for the teacher's native bf16 output.
pub const BF16_PARITY_REL: f64 = 0.015625;
/// Extra absolute tolerance as a fraction of the reference RMS for the native
/// bf16 path.
pub const BF16_PARITY_SCALE: f64 = 0.015625;
/// Human-readable tolerance policy recorded in the report.
pub const TOLERANCE_POLICY: &str = "elementwise |a-b| <= (abs + scale*rms) + rel*|reference|; f32 abs=1e-4 rel=1e-4 scale=1e-3; bf16 abs=0.05 rel=0.015625 scale=0.015625; rms is the reference RMS; documented in docs/ARCHITECTURE.md";

/// Options for [`verify_real_parity`].
#[derive(Clone, Debug)]
pub struct RealParityOptions {
    /// Absolute tolerance for f32 comparisons.
    pub f32_abs: f64,
    /// Relative tolerance for f32 comparisons.
    pub f32_rel: f64,
    /// Scale term for f32 comparisons, as a fraction of the reference RMS.
    pub f32_scale: f64,
    /// Absolute tolerance for the native bf16 comparison.
    pub bf16_abs: f64,
    /// Relative tolerance for the native bf16 comparison.
    pub bf16_rel: f64,
    /// Scale term for the native bf16 comparison, as a fraction of the
    /// reference RMS.
    pub bf16_scale: f64,
    /// Per-shard read cap.
    pub max_shard_bytes: u64,
    /// Receipt stamp.
    pub stamp: Option<String>,
    /// Git revision.
    pub revision: Option<String>,
}

impl Default for RealParityOptions {
    fn default() -> Self {
        Self {
            f32_abs: F32_PARITY_ABS,
            f32_rel: F32_PARITY_REL,
            f32_scale: F32_PARITY_SCALE,
            bf16_abs: BF16_PARITY_ABS,
            bf16_rel: BF16_PARITY_REL,
            bf16_scale: BF16_PARITY_SCALE,
            max_shard_bytes: 4 * 1024 * 1024 * 1024,
            stamp: None,
            revision: None,
        }
    }
}

impl RealParityOptions {
    fn validate(&self) -> Result<()> {
        for (name, value) in [
            ("f32_abs", self.f32_abs),
            ("f32_rel", self.f32_rel),
            ("f32_scale", self.f32_scale),
            ("bf16_abs", self.bf16_abs),
            ("bf16_rel", self.bf16_rel),
            ("bf16_scale", self.bf16_scale),
        ] {
            if !value.is_finite() || value < 0.0 {
                return Err(ConvertError::plan(format!(
                    "{name} must be a finite non-negative tolerance"
                )));
            }
        }
        if self.max_shard_bytes == 0 {
            return Err(ConvertError::plan("max_shard_bytes must be non-zero"));
        }
        Ok(())
    }
}

/// Accumulated error statistics.
#[derive(Clone, Copy, Debug, Serialize)]
pub struct ParityErrors {
    /// Largest absolute difference.
    pub max_abs: f64,
    /// Largest relative difference against `max(|reference|, 1e-12)`.
    pub max_rel: f64,
    /// Mean absolute difference.
    pub mean_abs: f64,
    /// Compared components.
    pub count: u64,
    /// Components outside the documented bound.
    pub violations: u64,
    /// Effective absolute part of the bound (`abs + scale * rms`).
    pub tolerance_abs: f64,
    /// Relative part of the bound.
    pub tolerance_rel: f64,
    /// Reference RMS used by the scale term.
    pub reference_rms: f64,
    /// Reference maximum absolute value.
    pub reference_max_abs: f64,
    /// Whether every component passed.
    pub passed: bool,
    /// Running absolute-difference sum (not serialized).
    #[serde(skip)]
    sum: f64,
}

impl ParityErrors {
    fn new() -> Self {
        Self {
            max_abs: 0.0,
            max_rel: 0.0,
            mean_abs: 0.0,
            count: 0,
            violations: 0,
            tolerance_abs: 0.0,
            tolerance_rel: 0.0,
            reference_rms: 0.0,
            reference_max_abs: 0.0,
            passed: true,
            sum: 0.0,
        }
    }

    /// Compare two buffers under the documented scale-aware bound.
    fn compare(
        &mut self,
        values: &[f32],
        reference: &[f32],
        abs_floor: f64,
        rel: f64,
        scale: f64,
    ) -> Result<()> {
        if values.len() != reference.len() {
            return Err(ConvertError::inconsistent(format!(
                "compared buffers differ in length: {} vs {}",
                values.len(),
                reference.len()
            )));
        }
        let mut sum_sq = 0.0_f64;
        let mut max_abs = 0.0_f64;
        for reference in reference {
            let value = f64::from(*reference);
            sum_sq += value * value;
            max_abs = max_abs.max(value.abs());
        }
        let rms = if reference.is_empty() {
            0.0
        } else {
            (sum_sq / reference.len() as f64).sqrt()
        };
        let tolerance_abs = abs_floor + scale * rms;
        self.tolerance_abs = tolerance_abs;
        self.tolerance_rel = rel;
        self.reference_rms = rms;
        self.reference_max_abs = max_abs;
        for (value, reference) in values.iter().zip(reference.iter()) {
            let difference = (f64::from(*value) - f64::from(*reference)).abs();
            self.max_abs = self.max_abs.max(difference);
            let denominator = f64::from(*reference).abs().max(1e-12);
            self.max_rel = self.max_rel.max(difference / denominator);
            self.sum += difference;
            self.count += 1;
            if difference > tolerance_abs + rel * f64::from(*reference).abs() {
                self.violations += 1;
                self.passed = false;
            }
        }
        Ok(())
    }

    fn merge(&mut self, other: &Self) {
        self.max_abs = self.max_abs.max(other.max_abs);
        self.max_rel = self.max_rel.max(other.max_rel);
        self.sum += other.sum;
        self.count += other.count;
        self.violations += other.violations;
        self.tolerance_abs = self.tolerance_abs.max(other.tolerance_abs);
        self.tolerance_rel = self.tolerance_rel.max(other.tolerance_rel);
        self.reference_rms = self.reference_rms.max(other.reference_rms);
        self.reference_max_abs = self.reference_max_abs.max(other.reference_max_abs);
        self.passed &= other.passed;
    }

    fn finish(mut self) -> Self {
        self.mean_abs = if self.count == 0 {
            0.0
        } else {
            self.sum / self.count as f64
        };
        self
    }
}

impl Default for ParityErrors {
    fn default() -> Self {
        Self::new()
    }
}

/// One layer's measured paths.
#[derive(Clone, Debug, Serialize)]
pub struct LayerParity {
    /// Layer index.
    pub layer: u32,
    /// Activation rows compared.
    pub rows: u64,
    /// Hidden width.
    pub hidden: u64,
    /// Intermediate width.
    pub intermediate: u64,
    /// Groups in the bundle for this layer.
    pub groups: usize,
    /// Grouped-from-bundle f32 against dense-from-checkpoint f32.
    pub bundle_vs_dense_f32: ParityErrors,
    /// Dense-from-checkpoint f32 against the teacher's f32 forward.
    pub dense_f32_vs_teacher_f32: ParityErrors,
    /// Grouped-from-bundle f32 against the teacher's f32 forward.
    pub grouped_f32_vs_teacher_f32: ParityErrors,
    /// Grouped-from-bundle with explicit bf16 intermediate rounding against the
    /// teacher's native bf16 output.
    pub grouped_bf16_emulated_vs_teacher_bf16: ParityErrors,
    /// Grouped-from-bundle in plain f32 arithmetic on the bf16 inputs. This is
    /// informational: it is not the teacher's arithmetic, and its deviation is
    /// reported rather than hidden.
    pub grouped_f32_on_bf16_inputs_vs_teacher_bf16: ParityErrors,
    /// Whether every path passed.
    pub passed: bool,
}

/// A named Court A scenario.
#[derive(Clone, Debug, Serialize)]
pub struct NamedScenario {
    /// Scenario name.
    pub name: String,
    /// Whether it passed.
    pub passed: bool,
    /// Measured detail.
    pub detail: String,
}

/// The complete report.
#[derive(Clone, Debug, Serialize)]
pub struct RealParityReport {
    /// Report schema.
    pub schema: String,
    /// Whether every measurement passed.
    pub ok: bool,
    /// Receipt stamp.
    pub stamp: Option<String>,
    /// Git revision.
    pub revision: Option<String>,
    /// Producer identity.
    pub generated_by: String,
    /// Bundle path.
    pub bundle: String,
    /// Source checkpoint path.
    pub model: String,
    /// Activation file path.
    pub activations: String,
    /// Activation file SHA-256.
    pub activations_sha256: String,
    /// Activation file bytes.
    pub activations_bytes: u64,
    /// Compared rows per layer.
    pub rows: u64,
    /// Compared layers.
    pub layers: u64,
    /// Tolerance policy text.
    pub tolerance_policy: String,
    /// f32 absolute tolerance.
    pub f32_abs: f64,
    /// f32 relative tolerance.
    pub f32_rel: f64,
    /// f32 scale term (fraction of reference RMS).
    pub f32_scale: f64,
    /// bf16 absolute tolerance.
    pub bf16_abs: f64,
    /// bf16 relative tolerance.
    pub bf16_rel: f64,
    /// bf16 scale term (fraction of reference RMS).
    pub bf16_scale: f64,
    /// Shards verified by the format validator.
    pub validated_shards: u64,
    /// Manifest payload length in bytes.
    pub bundle_payload_bytes: u64,
    /// Whole bundle file length in bytes.
    pub bundle_file_bytes: u64,
    /// Shard records in the manifest.
    pub bundle_shard_count: u64,
    /// Manifest alignment.
    pub bundle_alignment: u64,
    /// Bundle id.
    pub bundle_id: String,
    /// Group count per layer as emitted by the converter.
    pub bundle_groups_per_layer: Vec<usize>,
    /// Per-layer measurements.
    pub layer_results: Vec<LayerParity>,
    /// Aggregate `bundle_vs_dense_f32`.
    pub aggregate_bundle_vs_dense_f32: ParityErrors,
    /// Aggregate `dense_f32_vs_teacher_f32`.
    pub aggregate_dense_f32_vs_teacher_f32: ParityErrors,
    /// Aggregate `grouped_f32_vs_teacher_f32`.
    pub aggregate_grouped_f32_vs_teacher_f32: ParityErrors,
    /// Aggregate `grouped_bf16_emulated_vs_teacher_bf16`.
    pub aggregate_grouped_bf16_emulated_vs_teacher_bf16: ParityErrors,
    /// Aggregate `grouped_f32_on_bf16_inputs_vs_teacher_bf16`.
    pub aggregate_grouped_f32_on_bf16_inputs_vs_teacher_bf16: ParityErrors,
    /// Named Court A additions.
    pub scenarios: Vec<NamedScenario>,
}

fn dims_usize(hidden: u64, intermediate: u64, rows: u64) -> Result<(usize, usize, usize)> {
    let h = usize::try_from(hidden)
        .map_err(|_| ConvertError::inconsistent("hidden width does not fit usize"))?;
    let i = usize::try_from(intermediate)
        .map_err(|_| ConvertError::inconsistent("intermediate width does not fit usize"))?;
    let n = usize::try_from(rows)
        .map_err(|_| ConvertError::inconsistent("row count does not fit usize"))?;
    Ok((h, i, n))
}

fn read_activation(
    store: &Safetensors<std::fs::File>,
    key: &str,
    max_bytes: u64,
) -> Result<(Vec<f32>, u64, u64)> {
    let descriptor = TensorSource::descriptor(store, key)?;
    if descriptor.dtype != DType::F32 {
        return Err(ConvertError::inconsistent(format!(
            "activation {key:?} has dtype {}, expected f32",
            descriptor.dtype.stable_name()
        )));
    }
    if descriptor.rank() != 2 {
        return Err(ConvertError::RankMismatch {
            tensor: key.to_owned(),
            rank: descriptor.rank(),
        });
    }
    let bytes = store.read_tensor_bytes(key, max_bytes)?;
    let elements = usize::try_from(descriptor.elements)
        .map_err(|_| ConvertError::inconsistent("activation element count does not fit usize"))?;
    let mut out: Vec<f32> = Vec::new();
    reserve_elements(&mut out, descriptor.elements)?;
    out.resize(elements, 0.0);
    for (index, chunk) in bytes.as_chunks::<4>().0.iter().enumerate() {
        let slot = out.get_mut(index).ok_or_else(|| {
            ConvertError::inconsistent(format!(
                "activation {key:?} has more bytes than its declared shape"
            ))
        })?;
        *slot = f32::from_le_bytes(*chunk);
    }
    if bytes.len() as u64 != descriptor.elements * 4 {
        return Err(ConvertError::inconsistent(format!(
            "activation {key:?} decoded {} bytes, expected {}",
            bytes.len(),
            descriptor.elements * 4
        )));
    }
    Ok((out, descriptor.shape[0], descriptor.shape[1]))
}

fn thread_count(rows: usize) -> usize {
    let available = std::thread::available_parallelism()
        .map(|value| value.get())
        .unwrap_or(1);
    available.clamp(1, 16).min(rows.max(1))
}

#[allow(clippy::too_many_arguments)]
fn dense_block(
    gate: &[f32],
    up: &[f32],
    down_t: &[f32],
    h: usize,
    i: usize,
    x: &[f32],
    row_start: usize,
    row_end: usize,
    out: &mut [f32],
) {
    for row in row_start..row_end {
        let input = &x[row * h..(row + 1) * h];
        let output = &mut out[(row - row_start) * h..(row - row_start + 1) * h];
        for channel in 0..i {
            let gate_row = &gate[channel * h..(channel + 1) * h];
            let up_row = &up[channel * h..(channel + 1) * h];
            let mut gate_value = 0.0_f32;
            let mut up_value = 0.0_f32;
            for index in 0..h {
                gate_value += gate_row[index] * input[index];
                up_value += up_row[index] * input[index];
            }
            let activation = silu_f32(gate_value) * up_value;
            let down_row = &down_t[channel * h..(channel + 1) * h];
            for index in 0..h {
                output[index] += down_row[index] * activation;
            }
        }
    }
}

fn dense_parallel(
    gate: &[f32],
    up: &[f32],
    down_t: &[f32],
    hidden: u64,
    intermediate: u64,
    rows: u64,
    x: &[f32],
) -> Result<Vec<f32>> {
    let (h, i, n) = dims_usize(hidden, intermediate, rows)?;
    if gate.len() != i * h || up.len() != i * h || down_t.len() != i * h {
        return Err(ConvertError::inconsistent(
            "dense FFN buffers disagree with hidden/intermediate",
        ));
    }
    let expected = n
        .checked_mul(h)
        .ok_or_else(|| ConvertError::inconsistent("output size overflows"))?;
    if x.len() != expected {
        return Err(ConvertError::inconsistent(format!(
            "input has {} values, expected {expected}",
            x.len()
        )));
    }
    let mut out: Vec<f32> = Vec::new();
    reserve_elements(&mut out, expected as u64)?;
    out.resize(expected, 0.0);
    let threads = thread_count(n);
    if threads == 1 {
        dense_block(gate, up, down_t, h, i, x, 0, n, &mut out);
        return Ok(out);
    }
    let chunk = n.div_ceil(threads);
    std::thread::scope(|scope| {
        let mut rest = out.as_mut_slice();
        let mut start = 0;
        while start < n {
            let end = (start + chunk).min(n);
            let width = (end - start) * h;
            let (block, tail) = rest.split_at_mut(width);
            scope.spawn(move || {
                dense_block(gate, up, down_t, h, i, x, start, end, block);
            });
            rest = tail;
            start = end;
        }
    });
    Ok(out)
}

fn grouped_block(
    groups: &[GroupMatrices],
    h: usize,
    x: &[f32],
    row_start: usize,
    row_end: usize,
    out: &mut [f32],
) {
    let widest = groups
        .iter()
        .map(|group| group.channels.len())
        .max()
        .unwrap_or(0);
    let mut activations = vec![0.0_f32; widest];
    for row in row_start..row_end {
        let input = &x[row * h..(row + 1) * h];
        let output = &mut out[(row - row_start) * h..(row - row_start + 1) * h];
        for group in groups {
            let width = group.channels.len();
            for (channel, slot) in activations.iter_mut().enumerate().take(width) {
                let gate_row = &group.gate[channel * h..(channel + 1) * h];
                let up_row = &group.up[channel * h..(channel + 1) * h];
                let mut gate_value = 0.0_f32;
                let mut up_value = 0.0_f32;
                for index in 0..h {
                    gate_value += gate_row[index] * input[index];
                    up_value += up_row[index] * input[index];
                }
                *slot = silu_f32(gate_value) * up_value;
            }
            for (channel, activation) in activations.iter().enumerate().take(width) {
                let down_row = &group.down_t[channel * h..(channel + 1) * h];
                for index in 0..h {
                    output[index] += down_row[index] * *activation;
                }
            }
        }
    }
}

fn grouped_parallel(
    groups: &[GroupMatrices],
    hidden: u64,
    rows: u64,
    x: &[f32],
) -> Result<Vec<f32>> {
    let (h, n) = {
        let h = usize::try_from(hidden)
            .map_err(|_| ConvertError::inconsistent("hidden width does not fit usize"))?;
        let n = usize::try_from(rows)
            .map_err(|_| ConvertError::inconsistent("row count does not fit usize"))?;
        (h, n)
    };
    let expected = n
        .checked_mul(h)
        .ok_or_else(|| ConvertError::inconsistent("output size overflows"))?;
    if x.len() != expected {
        return Err(ConvertError::inconsistent(format!(
            "input has {} values, expected {expected}",
            x.len()
        )));
    }
    for group in groups {
        let width = group.channels.len();
        if group.gate.len() != width * h
            || group.up.len() != width * h
            || group.down_t.len() != width * h
        {
            return Err(ConvertError::inconsistent(format!(
                "group {} buffers do not match |G|={width} and hidden={h}",
                group.index
            )));
        }
    }
    let mut out: Vec<f32> = Vec::new();
    reserve_elements(&mut out, expected as u64)?;
    out.resize(expected, 0.0);
    let threads = thread_count(n);
    if threads == 1 {
        grouped_block(groups, h, x, 0, n, &mut out);
        return Ok(out);
    }
    let chunk = n.div_ceil(threads);
    std::thread::scope(|scope| {
        let mut rest = out.as_mut_slice();
        let mut start = 0;
        while start < n {
            let end = (start + chunk).min(n);
            let width = (end - start) * h;
            let (block, tail) = rest.split_at_mut(width);
            scope.spawn(move || {
                grouped_block(groups, h, x, start, end, block);
            });
            rest = tail;
            start = end;
        }
    });
    Ok(out)
}

/// Round an f32 to the nearest bf16 value, ties to even, returned as f32.
fn round_bf16(value: f32) -> f32 {
    crate::tensor_source::decode_bf16(crate::tensor_source::encode_bf16(value))
}

/// The teacher's native bf16 arithmetic, emulated explicitly.
///
/// PyTorch runs bf16 linear layers with f32 accumulation and rounds the
/// result to bf16; elementwise ops on bf16 compute in f32 internally and round
/// the result to bf16. The emulation applies exactly those roundings:
///
/// * gate and up dot products accumulate in f32, then round to bf16;
/// * SiLU is evaluated in f32 on the bf16 gate value, then rounded to bf16;
/// * the gate/up product is computed in f32 and rounded to bf16;
/// * the down projection accumulates all groups in f32 and rounds once at the
///   end.
fn grouped_bf16_block(
    groups: &[GroupMatrices],
    h: usize,
    x: &[f32],
    row_start: usize,
    row_end: usize,
    out: &mut [f32],
) {
    let widest = groups
        .iter()
        .map(|group| group.channels.len())
        .max()
        .unwrap_or(0);
    let mut activations = vec![0.0_f32; widest];
    for row in row_start..row_end {
        let input = &x[row * h..(row + 1) * h];
        let output = &mut out[(row - row_start) * h..(row - row_start + 1) * h];
        for group in groups {
            let width = group.channels.len();
            for (channel, slot) in activations.iter_mut().enumerate().take(width) {
                let gate_row = &group.gate[channel * h..(channel + 1) * h];
                let up_row = &group.up[channel * h..(channel + 1) * h];
                let mut gate_value = 0.0_f32;
                let mut up_value = 0.0_f32;
                for index in 0..h {
                    gate_value += gate_row[index] * input[index];
                    up_value += up_row[index] * input[index];
                }
                let gate_value = round_bf16(gate_value);
                let up_value = round_bf16(up_value);
                let silu_value = round_bf16(silu_f32(gate_value));
                *slot = round_bf16(silu_value * up_value);
            }
            for (channel, activation) in activations.iter().enumerate().take(width) {
                let down_row = &group.down_t[channel * h..(channel + 1) * h];
                for index in 0..h {
                    output[index] += down_row[index] * *activation;
                }
            }
        }
        for value in output.iter_mut() {
            *value = round_bf16(*value);
        }
    }
}

fn grouped_bf16_parallel(
    groups: &[GroupMatrices],
    hidden: u64,
    rows: u64,
    x: &[f32],
) -> Result<Vec<f32>> {
    let (h, n) = {
        let h = usize::try_from(hidden)
            .map_err(|_| ConvertError::inconsistent("hidden width does not fit usize"))?;
        let n = usize::try_from(rows)
            .map_err(|_| ConvertError::inconsistent("row count does not fit usize"))?;
        (h, n)
    };
    let expected = n
        .checked_mul(h)
        .ok_or_else(|| ConvertError::inconsistent("output size overflows"))?;
    if x.len() != expected {
        return Err(ConvertError::inconsistent(format!(
            "input has {} values, expected {expected}",
            x.len()
        )));
    }
    for group in groups {
        let width = group.channels.len();
        if group.gate.len() != width * h
            || group.up.len() != width * h
            || group.down_t.len() != width * h
        {
            return Err(ConvertError::inconsistent(format!(
                "group {} buffers do not match |G|={width} and hidden={h}",
                group.index
            )));
        }
    }
    let mut out: Vec<f32> = Vec::new();
    reserve_elements(&mut out, expected as u64)?;
    out.resize(expected, 0.0);
    let threads = thread_count(n);
    if threads == 1 {
        grouped_bf16_block(groups, h, x, 0, n, &mut out);
        return Ok(out);
    }
    let chunk = n.div_ceil(threads);
    std::thread::scope(|scope| {
        let mut rest = out.as_mut_slice();
        let mut start = 0;
        while start < n {
            let end = (start + chunk).min(n);
            let width = (end - start) * h;
            let (block, tail) = rest.split_at_mut(width);
            scope.spawn(move || {
                grouped_bf16_block(groups, h, x, start, end, block);
            });
            rest = tail;
            start = end;
        }
    });
    Ok(out)
}

fn hash_file(path: &Path) -> Result<(String, u64)> {
    let mut file = std::fs::File::open(path)?;
    let mut builder = selvedge_core::Sha256Builder::new();
    let mut buffer = vec![0_u8; 1024 * 1024];
    let mut total = 0_u64;
    loop {
        let read = std::io::Read::read(&mut file, &mut buffer)?;
        if read == 0 {
            break;
        }
        builder.update(&buffer[..read]);
        total += read as u64;
    }
    Ok((builder.finish().to_hex(), total))
}

/// Compare a bundle against real teacher activations for every layer.
///
/// # Errors
///
/// Rejects an invalid bundle, a missing or malformed activation, shape
/// disagreement between the bundle, the checkpoint and the activations, and
/// every reader failure.
pub fn verify_real_parity(
    bundle_path: &Path,
    model_path: &Path,
    activations_path: &Path,
    options: &RealParityOptions,
) -> Result<RealParityReport> {
    options.validate()?;
    let validation = validate_bundle(
        bundle_path,
        &ValidationOptions {
            verify: VerifyMode::All,
            ..ValidationOptions::default()
        },
    )?;
    let mut bundle = BundleFile::open(bundle_path)?;
    let mut bundle_layers = load_bundle_layers(
        &mut bundle,
        &crate::verify::VerificationOptions {
            max_shard_bytes: options.max_shard_bytes,
            ..crate::verify::VerificationOptions::default()
        },
    )?;
    bundle_layers.sort_by_key(|layer| layer.layer);
    if bundle_layers.is_empty() {
        return Err(ConvertError::inconsistent(
            "bundle contains no FFN group layers",
        ));
    }

    let mut source = crate::safetensors::CheckpointSource::open(model_path)?;
    let maps = discover_ffn_maps(&source.tensor_keys())?;
    let store = Safetensors::open(activations_path)?;
    let (activations_sha256, activations_bytes) = hash_file(activations_path)?;
    let bundle_payload_bytes = bundle.manifest().payload_len;
    let bundle_shard_count = bundle.manifest().shards.len() as u64;
    let bundle_alignment = bundle.manifest().alignment;
    let bundle_id = bundle.manifest().bundle_id.clone();
    let bundle_file_bytes = std::fs::metadata(bundle_path)?.len();

    let mut layer_results = Vec::with_capacity(bundle_layers.len());
    let mut agg_bundle_dense = ParityErrors::new();
    let mut agg_dense_teacher = ParityErrors::new();
    let mut agg_grouped_teacher = ParityErrors::new();
    let mut agg_bf16_emulated = ParityErrors::new();
    let mut agg_bf16_plain = ParityErrors::new();
    let mut expected_rows: Option<u64> = None;
    let mut all_passed = true;

    for layer in &bundle_layers {
        let layer_index = layer.layer.ok_or_else(|| {
            ConvertError::inconsistent("real parity requires bundle layers with an index")
        })?;
        let map = match_layer(&maps, layer.layer)?;
        let dims = validate_ffn_map(&source, map)?;
        if dims.hidden != layer.hidden || dims.intermediate != layer.intermediate {
            return Err(ConvertError::inconsistent(format!(
                "layer {layer_index}: bundle is {}x{}, checkpoint is {}x{}",
                layer.hidden, layer.intermediate, dims.hidden, dims.intermediate
            )));
        }
        check_bundle_layer(layer, dims.intermediate)?;
        let (h, i, _) = dims_usize(dims.hidden, dims.intermediate, 0)?;

        let gate = source.read_region_f32(&map.gate, 0..dims.intermediate, 0..dims.hidden)?;
        let up = source.read_region_f32(&map.up, 0..dims.intermediate, 0..dims.hidden)?;
        let down = source.read_region_f32(&map.down, 0..dims.hidden, 0..dims.intermediate)?;
        if gate.len() != i * h || up.len() != i * h || down.len() != h * i {
            return Err(ConvertError::inconsistent(format!(
                "layer {layer_index}: checkpoint tensors disagree with config dimensions"
            )));
        }
        let mut down_t: Vec<f32> = Vec::new();
        reserve_elements(&mut down_t, (i * h) as u64)?;
        down_t.resize(i * h, 0.0);
        for channel in 0..i {
            for index in 0..h {
                down_t[channel * h + index] = down[index * i + channel];
            }
        }

        let input_f32_key = format!("layer{layer_index}.input_f32");
        let output_f32_key = format!("layer{layer_index}.output_f32");
        let input_bf16_key = format!("layer{layer_index}.input_bf16");
        let output_bf16_key = format!("layer{layer_index}.output_bf16");
        let (x_f32, rows, hidden) =
            read_activation(&store, &input_f32_key, options.max_shard_bytes)?;
        let (t_f32, t_rows, t_hidden) =
            read_activation(&store, &output_f32_key, options.max_shard_bytes)?;
        let (x_bf16, b_rows, b_hidden) =
            read_activation(&store, &input_bf16_key, options.max_shard_bytes)?;
        let (t_bf16, bt_rows, bt_hidden) =
            read_activation(&store, &output_bf16_key, options.max_shard_bytes)?;
        if hidden != dims.hidden
            || t_hidden != dims.hidden
            || b_hidden != dims.hidden
            || bt_hidden != dims.hidden
        {
            return Err(ConvertError::inconsistent(format!(
                "layer {layer_index}: activation hidden width disagrees with the checkpoint"
            )));
        }
        if rows != t_rows
            || rows != b_rows
            || rows != bt_rows
            || rows != x_bf16.len() as u64 / hidden
        {
            return Err(ConvertError::inconsistent(format!(
                "layer {layer_index}: activation row counts disagree"
            )));
        }
        if let Some(expected) = expected_rows {
            if rows != expected {
                return Err(ConvertError::inconsistent(format!(
                    "layer {layer_index}: {rows} rows, previous layers had {expected}"
                )));
            }
        } else {
            expected_rows = Some(rows);
        }

        let dense_f32 = dense_parallel(
            &gate,
            &up,
            &down_t,
            dims.hidden,
            dims.intermediate,
            rows,
            &x_f32,
        )?;
        let groups: Vec<GroupMatrices> = layer
            .groups
            .iter()
            .map(|group| group.matrices.clone())
            .collect();
        let grouped_f32 = grouped_parallel(&groups, dims.hidden, rows, &x_f32)?;
        let grouped_bf16_emulated = grouped_bf16_parallel(&groups, dims.hidden, rows, &x_bf16)?;
        let grouped_f32_plain = grouped_parallel(&groups, dims.hidden, rows, &x_bf16)?;

        let mut bundle_dense = ParityErrors::new();
        let mut dense_teacher = ParityErrors::new();
        let mut grouped_teacher = ParityErrors::new();
        let mut bf16_emulated = ParityErrors::new();
        let mut bf16_plain = ParityErrors::new();
        bundle_dense.compare(
            &grouped_f32,
            &dense_f32,
            options.f32_abs,
            options.f32_rel,
            options.f32_scale,
        )?;
        dense_teacher.compare(
            &dense_f32,
            &t_f32,
            options.f32_abs,
            options.f32_rel,
            options.f32_scale,
        )?;
        grouped_teacher.compare(
            &grouped_f32,
            &t_f32,
            options.f32_abs,
            options.f32_rel,
            options.f32_scale,
        )?;
        bf16_emulated.compare(
            &grouped_bf16_emulated,
            &t_bf16,
            options.bf16_abs,
            options.bf16_rel,
            options.bf16_scale,
        )?;
        bf16_plain.compare(
            &grouped_f32_plain,
            &t_bf16,
            options.bf16_abs,
            options.bf16_rel,
            options.bf16_scale,
        )?;

        let passed = bundle_dense.passed
            && dense_teacher.passed
            && grouped_teacher.passed
            && bf16_emulated.passed;
        all_passed &= passed;
        agg_bundle_dense.merge(&bundle_dense);
        agg_dense_teacher.merge(&dense_teacher);
        agg_grouped_teacher.merge(&grouped_teacher);
        agg_bf16_emulated.merge(&bf16_emulated);
        agg_bf16_plain.merge(&bf16_plain);
        layer_results.push(LayerParity {
            layer: layer_index,
            rows,
            hidden: dims.hidden,
            intermediate: dims.intermediate,
            groups: layer.groups.len(),
            bundle_vs_dense_f32: bundle_dense.finish(),
            dense_f32_vs_teacher_f32: dense_teacher.finish(),
            grouped_f32_vs_teacher_f32: grouped_teacher.finish(),
            grouped_bf16_emulated_vs_teacher_bf16: bf16_emulated.finish(),
            grouped_f32_on_bf16_inputs_vs_teacher_bf16: bf16_plain.finish(),
            passed,
        });
    }

    let agg_bundle_dense = agg_bundle_dense.finish();
    let agg_dense_teacher = agg_dense_teacher.finish();
    let agg_grouped_teacher = agg_grouped_teacher.finish();
    let agg_bf16_emulated = agg_bf16_emulated.finish();
    let agg_bf16_plain = agg_bf16_plain.finish();
    let rows = expected_rows.unwrap_or(0);
    let layers = layer_results.len() as u64;

    let roundtrip_passed = layer_results
        .iter()
        .all(|layer| layer.bundle_vs_dense_f32.passed && layer.grouped_f32_vs_teacher_f32.passed);
    let teacher_passed = layer_results.iter().all(|layer| {
        layer.grouped_f32_vs_teacher_f32.passed
            && layer.grouped_bf16_emulated_vs_teacher_bf16.passed
    });
    let documented = (options.f32_abs - F32_PARITY_ABS).abs() < f64::EPSILON
        && (options.f32_rel - F32_PARITY_REL).abs() < f64::EPSILON
        && (options.f32_scale - F32_PARITY_SCALE).abs() < f64::EPSILON
        && (options.bf16_abs - BF16_PARITY_ABS).abs() < f64::EPSILON
        && (options.bf16_rel - BF16_PARITY_REL).abs() < f64::EPSILON
        && (options.bf16_scale - BF16_PARITY_SCALE).abs() < f64::EPSILON
        && agg_bf16_emulated.passed;

    let scenarios = vec![
        NamedScenario {
            name: "real_activation_bundle_roundtrip".to_owned(),
            passed: roundtrip_passed,
            detail: format!(
                "{layers} layers, {rows} rows/layer; grouped-from-bundle f32 vs dense-from-checkpoint f32 \
                 max abs {:.6e} rel {:.6e} over {} components; grouped f32 vs teacher f32 max abs {:.6e} rel {:.6e}",
                agg_bundle_dense.max_abs,
                agg_bundle_dense.max_rel,
                agg_bundle_dense.count,
                agg_grouped_teacher.max_abs,
                agg_grouped_teacher.max_rel
            ),
        },
        NamedScenario {
            name: "real_checkpoint_all_groups_match_teacher".to_owned(),
            passed: teacher_passed,
            detail: format!(
                "{layers} layers, {rows} rows/layer; grouped f32 vs teacher f32 max abs {:.6e} rel {:.6e}; \
                 grouped bf16-emulated vs teacher native bf16 max abs {:.6e} rel {:.6e} over {} components",
                agg_grouped_teacher.max_abs,
                agg_grouped_teacher.max_rel,
                agg_bf16_emulated.max_abs,
                agg_bf16_emulated.max_rel,
                agg_bf16_emulated.count
            ),
        },
        NamedScenario {
            name: "bf16_roundtrip_tolerance_is_documented".to_owned(),
            passed: documented,
            detail: format!(
                "bounds f32 abs {} rel {} scale {}, bf16 abs {} rel {} scale {}; \
                 emulated bf16 max abs {:.6e} rel {:.6e}; plain f32-on-bf16-inputs max abs {:.6e} (informational); \
                 policy: {TOLERANCE_POLICY}",
                options.f32_abs,
                options.f32_rel,
                options.f32_scale,
                options.bf16_abs,
                options.bf16_rel,
                options.bf16_scale,
                agg_bf16_emulated.max_abs,
                agg_bf16_emulated.max_rel,
                agg_bf16_plain.max_abs
            ),
        },
    ];

    Ok(RealParityReport {
        schema: "selvedge/real-parity/v1".to_owned(),
        ok: all_passed,
        stamp: options.stamp.clone(),
        revision: options.revision.clone(),
        generated_by: format!("selvedge-convert {}", env!("CARGO_PKG_VERSION")),
        bundle: bundle_path.display().to_string(),
        model: model_path.display().to_string(),
        activations: activations_path.display().to_string(),
        activations_sha256,
        activations_bytes,
        rows,
        layers,
        tolerance_policy: TOLERANCE_POLICY.to_owned(),
        f32_abs: options.f32_abs,
        f32_rel: options.f32_rel,
        f32_scale: options.f32_scale,
        bf16_abs: options.bf16_abs,
        bf16_rel: options.bf16_rel,
        bf16_scale: options.bf16_scale,
        validated_shards: validation.verified_shards.len() as u64,
        bundle_payload_bytes,
        bundle_file_bytes,
        bundle_shard_count,
        bundle_alignment,
        bundle_id,
        bundle_groups_per_layer: layer_results.iter().map(|layer| layer.groups).collect(),
        layer_results,
        aggregate_bundle_vs_dense_f32: agg_bundle_dense,
        aggregate_dense_f32_vs_teacher_f32: agg_dense_teacher,
        aggregate_grouped_f32_vs_teacher_f32: agg_grouped_teacher,
        aggregate_grouped_bf16_emulated_vs_teacher_bf16: agg_bf16_emulated,
        aggregate_grouped_f32_on_bf16_inputs_vs_teacher_bf16: agg_bf16_plain,
        scenarios,
    })
}

fn check_bundle_layer(layer: &BundleLayer, intermediate: u64) -> Result<()> {
    let covered: u64 = layer
        .groups
        .iter()
        .map(|group| group.channels.len() as u64)
        .sum();
    if covered != intermediate {
        return Err(ConvertError::inconsistent(format!(
            "layer {:?}: bundle groups cover {covered} of {intermediate} channels",
            layer.layer
        )));
    }
    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn group(
        index: usize,
        channels: Vec<u64>,
        gate: Vec<f32>,
        up: Vec<f32>,
        down_t: Vec<f32>,
    ) -> GroupMatrices {
        GroupMatrices {
            index,
            channels,
            gate,
            up,
            down_t,
        }
    }

    #[test]
    fn contiguous_groups_reproduce_dense_bitwise() {
        // hidden = 2, intermediate = 3, one row.
        let hidden = 2_usize;
        let gate = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let up = vec![6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
        // down is [hidden, intermediate]; transpose to [intermediate, hidden].
        let down = [0.5, -0.5, 1.5, -1.5, 2.5, -2.5];
        let mut down_t = vec![0.0_f32; hidden * 3];
        for channel in 0..3 {
            for index in 0..hidden {
                down_t[channel * hidden + index] = down[index * 3 + channel];
            }
        }
        let groups = vec![
            group(
                0,
                vec![0, 1],
                gate[0..4].to_vec(),
                up[0..4].to_vec(),
                down_t[0..4].to_vec(),
            ),
            group(
                1,
                vec![2],
                gate[4..6].to_vec(),
                up[4..6].to_vec(),
                down_t[4..6].to_vec(),
            ),
        ];
        let x = vec![0.25_f32, -0.75];
        let dense = dense_parallel(&gate, &up, &down_t, 2, 3, 1, &x).unwrap();
        let grouped = grouped_parallel(&groups, 2, 1, &x).unwrap();
        assert_eq!(dense, grouped);
    }

    #[test]
    fn bf16_rounding_is_round_to_nearest_even() {
        assert_eq!(round_bf16(1.0), 1.0);
        // bf16 spacing at 1.0 is 2^-7 and the halfway point is 2^-8.
        assert_eq!(round_bf16(1.0 + 1.0 / 512.0), 1.0);
        assert_eq!(round_bf16(1.0 + 3.0 / 512.0), 1.0 + 1.0 / 128.0);
        // Exactly halfway: ties-to-even keeps the even significand (1.0).
        assert_eq!(round_bf16(1.0 + 1.0 / 256.0), 1.0);
    }

    #[test]
    fn tolerance_constants_are_the_documented_ones() {
        assert_eq!(F32_PARITY_ABS, 1.0e-4);
        assert_eq!(F32_PARITY_REL, 1.0e-4);
        assert_eq!(F32_PARITY_SCALE, 1.0e-3);
        assert_eq!(BF16_PARITY_ABS, 0.05);
        assert_eq!(BF16_PARITY_REL, 0.015625);
        assert_eq!(BF16_PARITY_SCALE, 0.015625);
        assert!(TOLERANCE_POLICY.contains("docs/ARCHITECTURE.md"));
    }
}
