//! Masked group partial sums from a validated bundle (Court B cross-check).
//!
//! The frontier harness computes a masked MLP output in Python. This module
//! computes the same quantity independently from the published f32 bundle with
//! the Rust grouped reduction order, so a subset of the real harness can be
//! cross-checked rather than trusted.
//!
//! Inputs and outputs are raw little-endian f32 files, not safetensors: the
//! cross-check is a numerical instrument, and a raw buffer has no header or
//! dtype ambiguity to validate.
//!
//! Every allocation is bounded: the input shape is checked against the bundle
//! layer before it is read, the read is capped, and the output has exactly
//! `rows * hidden` elements.

use std::io::Read;
use std::path::Path;

use selvedge_core::Sha256Builder;
use selvedge_format::{BundleFile, ValidationOptions, VerifyMode, validate_bundle};
use serde::Serialize;

use crate::error::{ConvertError, Result};
use crate::ffn::ffn_grouped_f32;
use crate::tensor_source::reserve_bytes;
use crate::verify::{VerificationOptions, load_bundle_layer};

/// Largest accepted raw input buffer (one layer of f32 activations).
pub const MAX_INPUT_BYTES: u64 = 1024 * 1024 * 1024;

/// Options for [`masked_grouped_from_bundle`].
#[derive(Clone, Debug)]
pub struct MaskedOptions {
    /// Per-shard read cap.
    pub max_shard_bytes: u64,
    /// Receipt stamp recorded verbatim.
    pub stamp: Option<String>,
    /// Git revision recorded verbatim.
    pub revision: Option<String>,
}

impl Default for MaskedOptions {
    fn default() -> Self {
        Self {
            max_shard_bytes: 4 * 1024 * 1024 * 1024,
            stamp: None,
            revision: None,
        }
    }
}

/// Machine-readable result of one masked partial sum.
#[derive(Clone, Debug, Serialize)]
pub struct MaskedReport {
    /// Report schema.
    pub schema: String,
    /// Whether the computation completed.
    pub ok: bool,
    /// Receipt stamp.
    pub stamp: Option<String>,
    /// Git revision.
    pub revision: Option<String>,
    /// Producer identity.
    pub generated_by: String,
    /// Bundle path.
    pub bundle: String,
    /// Bundle id from the manifest.
    pub bundle_id: String,
    /// Layer index.
    pub layer: u32,
    /// Activation rows.
    pub rows: u64,
    /// Hidden width.
    pub hidden: u64,
    /// Groups in the bundle for this layer.
    pub groups_total: usize,
    /// Selected group indices, ascending.
    pub selected_groups: Vec<usize>,
    /// Intermediate channels selected.
    pub selected_channels: u64,
    /// Intermediate channels in the full layer.
    pub intermediate_total: u64,
    /// Input file path.
    pub input: String,
    /// Input byte length.
    pub input_bytes: u64,
    /// Input file SHA-256.
    pub input_sha256: String,
    /// Output file path.
    pub output: String,
    /// Output byte length.
    pub output_bytes: u64,
    /// Output file SHA-256.
    pub output_sha256: String,
    /// Sum of output components, as f64, for a cheap cross-language check.
    pub output_sum_f64: f64,
    /// Largest output magnitude.
    pub output_abs_max: f64,
    /// Reduction order description.
    pub reduction_order: String,
}

fn hash_bytes(bytes: &[u8]) -> String {
    let mut builder = Sha256Builder::new();
    builder.update(bytes);
    builder.finish().to_hex()
}

fn read_raw_f32(path: &Path, elements: u64) -> Result<Vec<u8>> {
    let expected = elements
        .checked_mul(4)
        .ok_or_else(|| ConvertError::inconsistent("input byte length overflows"))?;
    if expected > MAX_INPUT_BYTES {
        return Err(ConvertError::plan(format!(
            "masked input would be {expected} bytes, above the {MAX_INPUT_BYTES} cap"
        )));
    }
    let file = std::fs::File::open(path)?;
    let mut bytes = Vec::new();
    reserve_bytes(&mut bytes, expected)?;
    let mut reader = std::io::BufReader::new(file);
    let mut buffer = vec![0_u8; 1024 * 1024];
    loop {
        let read = reader.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        bytes.extend_from_slice(&buffer[..read]);
        if bytes.len() as u64 > expected {
            return Err(ConvertError::inconsistent(format!(
                "{} is longer than the {expected} bytes implied by rows*hidden",
                path.display()
            )));
        }
    }
    if bytes.len() as u64 != expected {
        return Err(ConvertError::inconsistent(format!(
            "{} has {} bytes, expected {expected}",
            path.display(),
            bytes.len()
        )));
    }
    Ok(bytes)
}

fn decode_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|chunk| f32::from_le_bytes(*chunk))
        .collect()
}

/// One masked partial-sum request.
#[derive(Clone, Debug)]
pub struct MaskedRequest<'a> {
    /// Bundle to read.
    pub bundle: &'a Path,
    /// Layer index.
    pub layer: u32,
    /// Selected group indices.
    pub selected_groups: &'a [usize],
    /// Raw little-endian f32 input of exactly `rows * hidden` values.
    pub input: &'a Path,
    /// Activation rows.
    pub rows: u64,
    /// Hidden width.
    pub hidden: u64,
    /// Where to write the raw little-endian f32 output.
    pub output: &'a Path,
}

/// Compute the sum of the selected group contributions for one real layer.
///
/// # Errors
///
/// Rejects an invalid bundle, a layer without groups, an out-of-range or
/// duplicate group index, a wrong-sized input, an unwritable output, and every
/// reader failure.
pub fn masked_grouped_from_bundle(
    request: &MaskedRequest<'_>,
    options: &MaskedOptions,
) -> Result<MaskedReport> {
    let bundle_path = request.bundle;
    let layer = request.layer;
    let selected = request.selected_groups;
    let input_path = request.input;
    let rows = request.rows;
    let hidden = request.hidden;
    let output_path = request.output;
    if rows == 0 || hidden == 0 {
        return Err(ConvertError::plan("rows and hidden must be non-zero"));
    }
    if options.max_shard_bytes == 0 {
        return Err(ConvertError::plan("max_shard_bytes must be non-zero"));
    }
    validate_bundle(
        bundle_path,
        &ValidationOptions {
            verify: VerifyMode::All,
            ..ValidationOptions::default()
        },
    )?;
    let mut bundle = BundleFile::open(bundle_path)?;
    let bundle_id = bundle.manifest().bundle_id.clone();
    let layer_data = load_bundle_layer(
        &mut bundle,
        &VerificationOptions {
            max_shard_bytes: options.max_shard_bytes,
            ..VerificationOptions::default()
        },
        layer,
    )?;
    if layer_data.hidden != hidden {
        return Err(ConvertError::inconsistent(format!(
            "layer {layer}: bundle hidden {0}, requested {hidden}",
            layer_data.hidden
        )));
    }
    let groups_total = layer_data.groups.len();
    if groups_total == 0 {
        return Err(ConvertError::inconsistent(format!(
            "layer {layer}: bundle contains no groups"
        )));
    }
    let mut sorted = selected.to_vec();
    sorted.sort_unstable();
    sorted.dedup();
    if sorted.len() != selected.len() {
        return Err(ConvertError::plan(
            "selected group indices contain duplicates",
        ));
    }
    let mut subset = Vec::with_capacity(sorted.len());
    for index in &sorted {
        let group = layer_data.groups.get(*index).ok_or_else(|| {
            ConvertError::plan(format!(
                "layer {layer}: group index {index} is out of range (0..{groups_total})"
            ))
        })?;
        subset.push(group.matrices.clone());
    }

    let elements = rows
        .checked_mul(hidden)
        .ok_or_else(|| ConvertError::inconsistent("rows*hidden overflows"))?;
    let input_bytes = read_raw_f32(input_path, elements)?;
    let input_sha256 = hash_bytes(&input_bytes);
    let x = decode_f32(&input_bytes);
    let output = ffn_grouped_f32(hidden, &subset, rows, &x)?;

    let mut out_bytes = Vec::new();
    reserve_bytes(&mut out_bytes, elements * 4)?;
    let mut sum = 0.0_f64;
    let mut abs_max = 0.0_f64;
    for value in &output {
        let value = f64::from(*value);
        sum += value;
        abs_max = abs_max.max(value.abs());
        out_bytes.extend_from_slice(&value.to_le_bytes()[..4]);
    }
    // `to_le_bytes` above widened back to f64; rebuild from the f32 values so
    // the file is exactly the computed f32 buffer.
    out_bytes.clear();
    for value in &output {
        out_bytes.extend_from_slice(&value.to_le_bytes());
    }
    if let Some(parent) = output_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(output_path, &out_bytes)?;

    let selected_channels: u64 = subset.iter().map(|group| group.channels.len() as u64).sum();
    Ok(MaskedReport {
        schema: "selvedge/court-b-real/v1".to_owned(),
        ok: true,
        stamp: options.stamp.clone(),
        revision: options.revision.clone(),
        generated_by: format!("selvedge-convert {}", env!("CARGO_PKG_VERSION")),
        bundle: bundle_path.display().to_string(),
        bundle_id,
        layer,
        rows,
        hidden,
        groups_total,
        selected_groups: sorted,
        selected_channels,
        intermediate_total: layer_data.intermediate,
        input: input_path.display().to_string(),
        input_bytes: elements * 4,
        input_sha256,
        output: output_path.display().to_string(),
        output_bytes: elements * 4,
        output_sha256: hash_bytes(&out_bytes),
        output_sum_f64: sum,
        output_abs_max: abs_max,
        reduction_order: "groups in ascending index order, channels ascending inside each group; f32 accumulation".to_owned(),
    })
}
