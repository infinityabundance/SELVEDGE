//! Deterministic synthetic SwiGLU checkpoints and a minimal safetensors writer.
//!
//! The generator uses an in-repo xorshift64* PRNG with a documented seed, so the
//! bytes are identical on every platform. It exists to prove the conversion and
//! reconstruction path end to end without any upstream model or network access.

use std::collections::BTreeMap;
use std::path::Path;

use selvedge_core::StorageOrder;
use serde_json::Value;

use crate::error::{ConvertError, Result};
use crate::ffn::{
    DiscoveredFfn, FfnOrientation, FfnTensorMap, discover_ffn_maps, validate_ffn_map,
};
use crate::tensor_source::{
    MemoryTensor, MemoryTensorSource, TensorSource, encode_bf16, encode_f16,
};

/// Default seed for the committed synthetic checkpoint.
pub const SYNTHETIC_SEED: u64 = 0x5E1A_ED6E_51E1_2026;
/// Default hidden width of the committed synthetic checkpoint.
pub const SYNTHETIC_HIDDEN: u64 = 8;
/// Default intermediate width of the committed synthetic checkpoint.
pub const SYNTHETIC_INTERMEDIATE: u64 = 10;

/// A deterministic xorshift64* PRNG.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeterministicRng {
    state: u64,
}

impl DeterministicRng {
    /// Create a generator. A zero seed is replaced by a fixed non-zero constant
    /// so the sequence is always well defined.
    #[must_use]
    pub const fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 {
                0x9E37_79B9_7F4A_7C15
            } else {
                seed
            },
        }
    }

    /// Next raw value.
    pub fn next_u64(&mut self) -> u64 {
        let mut state = self.state;
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        self.state = state;
        state.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Uniform f32 in `[-1, 1)` derived from the top 24 bits.
    pub fn next_f32(&mut self) -> f32 {
        let unit = (self.next_u64() >> 40) as f32 / ((1_u32 << 24) as f32);
        unit * 2.0 - 1.0
    }

    /// Fill a vector with uniform values in `[-1, 1)`.
    pub fn fill(&mut self, len: usize) -> Vec<f32> {
        let mut out = Vec::with_capacity(len);
        for _ in 0..len {
            out.push(self.next_f32());
        }
        out
    }
}

/// One synthetic tensor in logical row-major order.
#[derive(Clone, Debug, PartialEq)]
pub struct SyntheticTensor {
    /// Logical shape.
    pub shape: Vec<u64>,
    /// Values in logical row-major order.
    pub values: Vec<f32>,
}

/// A deterministic synthetic SwiGLU checkpoint.
#[derive(Clone, Debug, PartialEq)]
pub struct SyntheticModel {
    /// Hidden width.
    pub hidden: u64,
    /// Intermediate width.
    pub intermediate: u64,
    /// Seed used to generate the values.
    pub seed: u64,
    /// Tensors sorted by key, logical row-major.
    pub tensors: BTreeMap<String, SyntheticTensor>,
    /// Discovered FFN maps.
    pub maps: Vec<DiscoveredFfn>,
}

/// Source dtype used when encoding the synthetic checkpoint.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SyntheticDType {
    /// IEEE binary32.
    F32,
    /// IEEE binary16.
    F16,
    /// bfloat16.
    BF16,
}

impl SyntheticDType {
    /// Safetensors dtype string.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::F32 => "F32",
            Self::F16 => "F16",
            Self::BF16 => "BF16",
        }
    }

    /// Bytes per value.
    #[must_use]
    pub const fn bytes_per_value(self) -> u64 {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::BF16 => 2,
        }
    }
}

/// Build a one-layer synthetic SwiGLU checkpoint.
///
/// Tensor keys are `model.layers.0.mlp.{gate,up,down}_proj.weight`; gate and up
/// are `[intermediate, hidden]`, down is `[hidden, intermediate]`.
///
/// # Errors
///
/// Rejects zero dimensions and inconsistent generated shapes.
pub fn swiglu_model(hidden: u64, intermediate: u64, seed: u64) -> Result<SyntheticModel> {
    if hidden == 0 || intermediate == 0 {
        return Err(ConvertError::inconsistent(
            "synthetic dimensions must be non-zero",
        ));
    }
    let hidden_usize = usize::try_from(hidden)
        .map_err(|_| ConvertError::inconsistent("hidden width does not fit usize"))?;
    let intermediate_usize = usize::try_from(intermediate)
        .map_err(|_| ConvertError::inconsistent("intermediate width does not fit usize"))?;
    let mut rng = DeterministicRng::new(seed);
    let gate_values = rng.fill(intermediate_usize * hidden_usize);
    let up_values = rng.fill(intermediate_usize * hidden_usize);
    let down_values = rng.fill(hidden_usize * intermediate_usize);
    let gate_shape = FfnOrientation::IntermediateFirst.gate_up_shape(intermediate, hidden);
    let down_shape = FfnOrientation::IntermediateFirst.down_shape(intermediate, hidden);
    let mut tensors = BTreeMap::new();
    tensors.insert(
        "model.layers.0.mlp.gate_proj.weight".to_owned(),
        SyntheticTensor {
            shape: gate_shape.to_vec(),
            values: gate_values,
        },
    );
    tensors.insert(
        "model.layers.0.mlp.up_proj.weight".to_owned(),
        SyntheticTensor {
            shape: gate_shape.to_vec(),
            values: up_values,
        },
    );
    tensors.insert(
        "model.layers.0.mlp.down_proj.weight".to_owned(),
        SyntheticTensor {
            shape: down_shape.to_vec(),
            values: down_values,
        },
    );
    let keys: Vec<String> = tensors.keys().cloned().collect();
    let maps = discover_ffn_maps(&keys)?;
    Ok(SyntheticModel {
        hidden,
        intermediate,
        seed,
        tensors,
        maps,
    })
}

impl SyntheticModel {
    /// The first FFN map for this checkpoint, if discovery produced one.
    #[must_use]
    pub fn map(&self) -> Option<&FfnTensorMap> {
        self.maps.first().map(|discovered| &discovered.map)
    }

    /// Build an in-memory source with a chosen dtype and storage order.
    ///
    /// # Errors
    ///
    /// Propagates shape/value mismatches.
    pub fn to_source(
        &self,
        dtype: SyntheticDType,
        order: StorageOrder,
    ) -> Result<MemoryTensorSource> {
        let mut source = MemoryTensorSource::new();
        for (key, tensor) in &self.tensors {
            let memory = match dtype {
                SyntheticDType::F32 => MemoryTensor::from_f32(
                    key.clone(),
                    tensor.shape.clone(),
                    order.clone(),
                    &tensor.values,
                )?,
                SyntheticDType::F16 => MemoryTensor::from_f16(
                    key.clone(),
                    tensor.shape.clone(),
                    order.clone(),
                    &tensor.values,
                )?,
                SyntheticDType::BF16 => MemoryTensor::from_bf16(
                    key.clone(),
                    tensor.shape.clone(),
                    order.clone(),
                    &tensor.values,
                )?,
            };
            source.insert(memory)?;
        }
        Ok(source)
    }

    /// Validate the discovered maps against a source.
    ///
    /// # Errors
    ///
    /// Propagates map validation failures.
    pub fn validate(&self, source: &impl TensorSource) -> Result<()> {
        for discovered in &self.maps {
            let dims = validate_ffn_map(source, &discovered.map)?;
            if dims.hidden != self.hidden || dims.intermediate != self.intermediate {
                return Err(ConvertError::inconsistent(format!(
                    "synthetic map dimensions {dims:?} disagree with {}x{}",
                    self.hidden, self.intermediate
                )));
            }
        }
        Ok(())
    }
}

/// Encode the model as a safetensors byte image.
///
/// The header is padded with spaces to a multiple of eight bytes, tensors are
/// written in sorted key order, and the data section contains no gaps.
///
/// # Errors
///
/// Rejects size overflow and serialization failure.
pub fn encode_safetensors(model: &SyntheticModel, dtype: SyntheticDType) -> Result<Vec<u8>> {
    let mut data: Vec<u8> = Vec::new();
    let mut header = BTreeMap::<String, Value>::new();
    let metadata = serde_json::json!({
        "format": "pt",
        "source": "selvedge-synthetic",
        "seed": format!("{:#018x}", model.seed),
    });
    header.insert("__metadata__".to_owned(), metadata);
    for (key, tensor) in &model.tensors {
        let begin = data.len() as u64;
        encode_values(dtype, &tensor.values, &mut data)?;
        let end = data.len() as u64;
        let entry = serde_json::json!({
            "dtype": dtype.name(),
            "shape": tensor.shape,
            "data_offsets": [begin, end],
        });
        header.insert(key.clone(), entry);
    }
    let header_bytes = serde_json::to_vec(&header)
        .map_err(|error| ConvertError::malformed(format!("cannot serialize header: {error}")))?;
    let padded = header_bytes.len().div_ceil(8) * 8;
    let mut out = Vec::new();
    let header_len = u64::try_from(padded).map_err(|_| ConvertError::Allocation {
        bytes: padded as u64,
        reason: "header length does not fit u64".to_owned(),
    })?;
    out.extend_from_slice(&header_len.to_le_bytes());
    out.extend_from_slice(&header_bytes);
    out.resize(8 + padded, b' ');
    out.extend_from_slice(&data);
    Ok(out)
}

/// Write the model to a safetensors file.
///
/// # Errors
///
/// Propagates encoding and filesystem failures.
pub fn write_safetensors(model: &SyntheticModel, dtype: SyntheticDType, path: &Path) -> Result<()> {
    let bytes = encode_safetensors(model, dtype)?;
    std::fs::write(path, bytes)?;
    Ok(())
}

fn encode_values(dtype: SyntheticDType, values: &[f32], out: &mut Vec<u8>) -> Result<()> {
    let extra = (values.len() as u64)
        .checked_mul(dtype.bytes_per_value())
        .ok_or(ConvertError::Allocation {
            bytes: u64::MAX,
            reason: "tensor byte length overflows".to_owned(),
        })?;
    let extra = usize::try_from(extra).map_err(|_| ConvertError::Allocation {
        bytes: extra,
        reason: "tensor byte length does not fit usize".to_owned(),
    })?;
    out.try_reserve_exact(extra)
        .map_err(|error| ConvertError::Allocation {
            bytes: extra as u64,
            reason: error.to_string(),
        })?;
    match dtype {
        SyntheticDType::F32 => {
            for value in values {
                out.extend_from_slice(&value.to_le_bytes());
            }
        }
        SyntheticDType::F16 => {
            for value in values {
                out.extend_from_slice(&encode_f16(*value).to_le_bytes());
            }
        }
        SyntheticDType::BF16 => {
            for value in values {
                out.extend_from_slice(&encode_bf16(*value).to_le_bytes());
            }
        }
    }
    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn generation_is_reproducible() {
        let first = swiglu_model(8, 10, SYNTHETIC_SEED).unwrap();
        let second = swiglu_model(8, 10, SYNTHETIC_SEED).unwrap();
        assert_eq!(first, second);
        let bytes = encode_safetensors(&first, SyntheticDType::F32).unwrap();
        let more = encode_safetensors(&second, SyntheticDType::F32).unwrap();
        assert_eq!(bytes, more);
    }

    #[test]
    fn encoded_checkpoint_round_trips_through_the_reader() {
        let model = swiglu_model(8, 10, SYNTHETIC_SEED).unwrap();
        let bytes = encode_safetensors(&model, SyntheticDType::F32).unwrap();
        let index = crate::safetensors::parse_bytes(&bytes).unwrap();
        assert_eq!(index.tensors.len(), 3);
        for (key, tensor) in &model.tensors {
            let entry = index.tensors.get(key).unwrap();
            assert_eq!(entry.shape, tensor.shape);
            let region = crate::safetensors::read_region_in_memory(
                &bytes,
                key,
                0..tensor.shape[0],
                0..tensor.shape[1],
            )
            .unwrap();
            assert_eq!(region, tensor.values);
        }
    }

    #[test]
    fn small_fixture_stays_under_two_hundred_kilobytes() {
        let model = swiglu_model(SYNTHETIC_HIDDEN, SYNTHETIC_INTERMEDIATE, SYNTHETIC_SEED).unwrap();
        let bytes = encode_safetensors(&model, SyntheticDType::F32).unwrap();
        assert!(bytes.len() < 200 * 1024);
    }
}
