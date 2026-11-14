//! Deterministic, non-proprietary fixtures for SELVEDGE courts.
//!
//! The generator is in-repo and uses a small xorshift PRNG, so no `rand`
//! dependency or platform RNG is involved. Regenerate with:
//!
//! ```text
//! scripts/court rust -- cargo run --offline -p selvedge-fixtures -- fixtures
//! ```
//!
//! The committed artifact is `fixtures/tiny-synthetic.slv` plus its sidecar;
//! `fixtures/SHA256SUMS` is rewritten by the same command.

#![forbid(unsafe_code)]
#![deny(missing_docs)]

use std::collections::BTreeMap;
use std::path::Path;

use selvedge_core::{
    ChannelSet, DType, FORMAT_VERSION, ModelDigest, Sha256Digest, ShardId, ShardIdentity,
    StorageOrder, TensorRole, align_up, bytes_for, element_count,
};
use selvedge_format::{
    AuxExtent, FormatError, MANIFEST_SCHEMA, MANIFEST_SCHEMA_VERSION, Manifest, PublishedBundle,
    QuantizationSpec, ShardRecord, WriteOptions, encode_bundle_bytes,
};

/// Fixture bundle id.
pub const FIXTURE_BUNDLE_ID: &str = "tiny-synthetic";
/// File stem of the committed synthetic SwiGLU checkpoint.
pub const FIXTURE_SAFETENSORS_ID: &str = "tiny-swiglu";

/// Fixture alignment. Smaller than the 4096 default so the committed artifact
/// stays far below the 100 KiB budget; this is a fixture choice, not a claim
/// that 512 is a better runtime alignment.
pub const FIXTURE_ALIGNMENT: u64 = 512;

const HIDDEN: u64 = 16;
const GROUPS: u64 = 4;
const GROUP_SIZE: u64 = 8;
const LAYERS: u32 = 2;
const VOCAB: u64 = 32;

/// Deterministic xorshift64* PRNG.
#[derive(Clone, Debug)]
pub struct DeterministicRng {
    state: u64,
}

impl DeterministicRng {
    /// Create a generator with a fixed seed.
    #[must_use]
    pub fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 {
                0x9E37_79B9_7F4A_7C15
            } else {
                seed
            },
        }
    }

    /// Next raw 64-bit value.
    pub fn next_u64(&mut self) -> u64 {
        let mut state = self.state;
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        self.state = state;
        state.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Next value in `[-1, 1)`.
    pub fn next_f32(&mut self) -> f32 {
        let raw = (self.next_u64() >> 40) as u32;
        let unit = raw as f32 / 16_777_216.0;
        unit.mul_add(2.0, -1.0)
    }

    /// Next signed 8-bit value.
    pub fn next_i8(&mut self) -> i8 {
        let value = (self.next_u64() % 255) as u8;
        value.wrapping_sub(127).cast_signed()
    }
}

/// A generated manifest and its payload.
#[derive(Clone, Debug, PartialEq)]
pub struct FixtureBundle {
    /// Bundle manifest.
    pub manifest: Manifest,
    /// Payload bytes, aligned to [`FIXTURE_ALIGNMENT`].
    pub payload: Vec<u8>,
}

struct PlannedShard {
    label: String,
    role: TensorRole,
    layer: Option<u32>,
    channels: ChannelSet,
    channel_axis: u32,
    shape: Vec<u64>,
    dtype: DType,
    quant: Option<QuantizationSpec>,
    transform_id: String,
    calibration: Option<selvedge_format::CalibrationProvenance>,
    extent: Vec<u8>,
    offset: u64,
    shard_id: ShardId,
}

impl FixtureBundle {
    /// Encoded bundle bytes.
    ///
    /// # Errors
    ///
    /// Propagates encoding failure.
    pub fn encoded(&self) -> Result<Vec<u8>, FormatError> {
        encode_bundle_bytes(&self.manifest, &self.payload)
    }

    /// SHA-256 of the encoded bundle.
    ///
    /// # Errors
    ///
    /// Propagates encoding failure.
    pub fn sha256(&self) -> Result<Sha256Digest, FormatError> {
        Ok(Sha256Digest::of(&self.encoded()?))
    }
}

fn model_identity() -> Result<ModelDigest, FormatError> {
    Ok(ModelDigest::from_parts(
        Sha256Digest::of(b"selvedge tiny synthetic model v1"),
        Some("synthetic-fixture".to_owned()),
        Some(Sha256Digest::of(b"selvedge tiny synthetic tokenizer v1")),
    )?)
}

fn f32_bytes(rng: &mut DeterministicRng, elements: u64) -> Vec<u8> {
    let mut out = Vec::with_capacity((elements as usize) * 4);
    for _ in 0..elements {
        out.extend_from_slice(&rng.next_f32().to_le_bytes());
    }
    out
}

fn i8_quantized_bytes(
    rng: &mut DeterministicRng,
    elements: u64,
    block_size: u32,
) -> (Vec<u8>, QuantizationSpec) {
    let mut weights = Vec::with_capacity(elements as usize);
    for _ in 0..elements {
        weights.push(rng.next_i8().to_le_bytes()[0]);
    }
    let blocks = elements / u64::from(block_size);
    let mut scales = Vec::with_capacity((blocks as usize) * 4);
    for index in 0..blocks {
        let scale = 0.01_f32 + (index as f32) * 0.005;
        scales.extend_from_slice(&scale.to_le_bytes());
    }
    let scale_offset = weights.len() as u64;
    let scale_length = scales.len() as u64;
    weights.extend_from_slice(&scales);
    let spec = QuantizationSpec {
        block_size,
        scales: Some(AuxExtent {
            offset: scale_offset,
            length: scale_length,
            dtype: DType::F32,
        }),
        zero_points: None,
    };
    (weights, spec)
}

/// Build the tiny synthetic bundle.
///
/// # Errors
///
/// Propagates identifier, size and manifest errors.
pub fn tiny_bundle() -> Result<FixtureBundle, FormatError> {
    let model = model_identity()?;
    let mut rng = DeterministicRng::new(0x5E1A_ED6E_5E1A_ED6E);
    let mut planned: Vec<PlannedShard> = Vec::new();
    let mut ids: BTreeMap<String, ShardId> = BTreeMap::new();

    let push = |planned: &mut Vec<PlannedShard>,
                ids: &mut BTreeMap<String, ShardId>,
                label: &str,
                role: TensorRole,
                layer: Option<u32>,
                channels: ChannelSet,
                channel_axis: u32,
                shape: Vec<u64>,
                dtype: DType,
                quant: Option<QuantizationSpec>,
                transform_id: &str,
                calibration: Option<selvedge_format::CalibrationProvenance>,
                extent: Vec<u8>|
     -> Result<(), FormatError> {
        let identity = ShardIdentity {
            format_version: FORMAT_VERSION,
            model: model.clone(),
            role: role.clone(),
            layer,
            channels: channels.clone(),
            channel_axis,
            dtype: dtype.clone(),
            quant: quant.as_ref().map(QuantizationSpec::scheme),
            transform_id: transform_id.to_owned(),
        };
        let shard_id = ShardId::derive(&identity)?;
        ids.insert(label.to_owned(), shard_id);
        planned.push(PlannedShard {
            label: label.to_owned(),
            role,
            layer,
            channels,
            channel_axis,
            shape,
            dtype,
            quant,
            transform_id: transform_id.to_owned(),
            calibration,
            extent,
            offset: 0,
            shard_id,
        });
        Ok(())
    };

    let embed_extent = f32_bytes(&mut rng, VOCAB * HIDDEN);
    push(
        &mut planned,
        &mut ids,
        "embed_tokens",
        TensorRole::EmbedTokens,
        None,
        ChannelSet::range(0, VOCAB)?,
        0,
        vec![VOCAB, HIDDEN],
        DType::F32,
        None,
        "identity",
        None,
        embed_extent.clone(),
    )?;
    push(
        &mut planned,
        &mut ids,
        "lm_head",
        TensorRole::LmHead,
        None,
        ChannelSet::range(0, VOCAB)?,
        0,
        vec![VOCAB, HIDDEN],
        DType::F32,
        None,
        "identity",
        None,
        embed_extent,
    )?;

    for layer in 0..LAYERS {
        let layer_label = format!("l{layer}");
        let norm_extent = f32_bytes(&mut rng, HIDDEN);
        push(
            &mut planned,
            &mut ids,
            &format!("{layer_label}.input_norm"),
            TensorRole::InputNorm,
            Some(layer),
            ChannelSet::range(0, HIDDEN)?,
            0,
            vec![HIDDEN],
            DType::F32,
            None,
            "identity",
            None,
            norm_extent.clone(),
        )?;
        push(
            &mut planned,
            &mut ids,
            &format!("{layer_label}.post_attn_norm"),
            TensorRole::PostAttnNorm,
            Some(layer),
            ChannelSet::range(0, HIDDEN)?,
            0,
            vec![HIDDEN],
            DType::F32,
            None,
            "identity",
            None,
            norm_extent,
        )?;
        for (role, name) in [
            (TensorRole::AttnQ, "attn.q"),
            (TensorRole::AttnK, "attn.k"),
            (TensorRole::AttnV, "attn.v"),
            (TensorRole::AttnO, "attn.o"),
        ] {
            let extent = f32_bytes(&mut rng, HIDDEN * HIDDEN);
            push(
                &mut planned,
                &mut ids,
                &format!("{layer_label}.{name}"),
                role,
                Some(layer),
                ChannelSet::range(0, HIDDEN)?,
                0,
                vec![HIDDEN, HIDDEN],
                DType::F32,
                None,
                "identity",
                None,
                extent,
            )?;
        }
        for group in 0..GROUPS {
            let channels = ChannelSet::range(group * GROUP_SIZE, (group + 1) * GROUP_SIZE)?;
            let elements = GROUP_SIZE * HIDDEN;
            let quantized = layer == 1 && group == 3;
            let (gate_extent, gate_dtype, gate_quant, gate_transform, calibration) = if quantized {
                let (bytes, spec) = i8_quantized_bytes(&mut rng, elements, 32);
                (
                    bytes,
                    DType::I8,
                    Some(spec),
                    "i8-symmetric-block32",
                    Some(selvedge_format::CalibrationProvenance {
                        method: "synthetic-fixture".to_owned(),
                        dataset_digest: Some(Sha256Digest::of(b"synthetic calibration set")),
                        notes: Some(
                            "Synthetic scale values; not a calibrated model artifact.".to_owned(),
                        ),
                    }),
                )
            } else {
                (
                    f32_bytes(&mut rng, elements),
                    DType::F32,
                    None,
                    "identity",
                    None,
                )
            };
            push(
                &mut planned,
                &mut ids,
                &format!("{layer_label}.mlp.g{group}.gate"),
                TensorRole::MlpGate,
                Some(layer),
                channels.clone(),
                0,
                vec![GROUP_SIZE, HIDDEN],
                gate_dtype,
                gate_quant,
                gate_transform,
                calibration,
                gate_extent,
            )?;
            let up_extent = f32_bytes(&mut rng, elements);
            push(
                &mut planned,
                &mut ids,
                &format!("{layer_label}.mlp.g{group}.up"),
                TensorRole::MlpUp,
                Some(layer),
                channels.clone(),
                0,
                vec![GROUP_SIZE, HIDDEN],
                DType::F32,
                None,
                "identity",
                None,
                up_extent,
            )?;
            let down_extent = f32_bytes(&mut rng, elements);
            push(
                &mut planned,
                &mut ids,
                &format!("{layer_label}.mlp.g{group}.down"),
                TensorRole::MlpDown,
                Some(layer),
                channels,
                1,
                vec![HIDDEN, GROUP_SIZE],
                DType::F32,
                None,
                "identity",
                None,
                down_extent,
            )?;
        }
    }

    // Assign aligned offsets and assemble the payload.
    let mut payload: Vec<u8> = Vec::new();
    for shard in &mut planned {
        let offset = align_up(payload.len() as u64, FIXTURE_ALIGNMENT)?;
        let padding = usize::try_from(offset - payload.len() as u64).map_err(|_| {
            FormatError::SizeOverflow {
                context: "fixture padding",
            }
        })?;
        payload.resize(payload.len() + padding, 0);
        shard.offset = offset;
        payload.extend_from_slice(&shard.extent);
    }
    let payload_len = align_up(payload.len() as u64, FIXTURE_ALIGNMENT)?;
    payload.resize(
        usize::try_from(payload_len).map_err(|_| FormatError::SizeOverflow {
            context: "fixture payload length",
        })?,
        0,
    );

    let embed_id =
        ids.get("embed_tokens")
            .copied()
            .ok_or_else(|| FormatError::ManifestInvalid {
                reason: "fixture lost the embedding id".to_owned(),
            })?;

    let mut records = Vec::with_capacity(planned.len());
    for shard in &planned {
        let mut co_access = Vec::new();
        if let Some(rest) = shard.label.strip_suffix(".gate") {
            for suffix in [".up", ".down"] {
                if let Some(id) = ids.get(&format!("{rest}{suffix}")) {
                    co_access.push(*id);
                }
            }
        } else if let Some(rest) = shard.label.strip_suffix(".up") {
            for suffix in [".gate", ".down"] {
                if let Some(id) = ids.get(&format!("{rest}{suffix}")) {
                    co_access.push(*id);
                }
            }
        } else if let Some(rest) = shard.label.strip_suffix(".down") {
            for suffix in [".gate", ".up"] {
                if let Some(id) = ids.get(&format!("{rest}{suffix}")) {
                    co_access.push(*id);
                }
            }
        }
        co_access.sort_unstable();
        co_access.dedup();
        let depends_on = if shard.label == "lm_head" {
            vec![embed_id]
        } else {
            Vec::new()
        };
        let elements = element_count(&shard.shape)?;
        let expected = bytes_for(elements, &shard.dtype)?;
        let aux = shard
            .quant
            .as_ref()
            .map_or(Ok(0), QuantizationSpec::auxiliary_bytes)?;
        if expected + aux != shard.extent.len() as u64 {
            return Err(FormatError::ManifestInvalid {
                reason: format!(
                    "fixture shard {} has {} bytes, expected {}",
                    shard.label,
                    shard.extent.len(),
                    expected + aux
                ),
            });
        }
        records.push(ShardRecord {
            shard_id: shard.shard_id,
            role: shard.role.clone(),
            layer: shard.layer,
            channels: shard.channels.clone(),
            channel_axis: shard.channel_axis,
            shape: shard.shape.clone(),
            storage_order: StorageOrder::RowMajor,
            dtype: shard.dtype.clone(),
            quant: shard.quant.clone(),
            offset: shard.offset,
            length: shard.extent.len() as u64,
            sha256: Sha256Digest::of(&shard.extent),
            alignment: FIXTURE_ALIGNMENT,
            compatibility_id: "tiny-synthetic-v1".to_owned(),
            transform_id: shard.transform_id.clone(),
            co_access,
            depends_on,
            calibration: shard.calibration.clone(),
        });
    }
    records.sort_by_key(|record| record.shard_id);

    let manifest = Manifest {
        schema: MANIFEST_SCHEMA.to_owned(),
        schema_version: MANIFEST_SCHEMA_VERSION,
        bundle_id: FIXTURE_BUNDLE_ID.to_owned(),
        model,
        transforms: vec!["i8-symmetric-block32".to_owned(), "identity".to_owned()],
        alignment: FIXTURE_ALIGNMENT,
        payload_len,
        created_by: format!("selvedge-fixtures {}", env!("CARGO_PKG_VERSION")),
        shards: records,
    };
    manifest.validate()?;
    Ok(FixtureBundle { manifest, payload })
}

/// Publish the fixture bundle into `dir`.
///
/// # Errors
///
/// Propagates generation and publication failures.
pub fn publish_tiny_bundle(dir: &Path, force: bool) -> Result<PublishedBundle, FormatError> {
    let fixture = tiny_bundle()?;
    let options = WriteOptions {
        force,
        verify_before_publish: true,
        publish_sidecar: true,
        ..WriteOptions::default()
    };
    selvedge_format::publish_bundle(dir, &fixture.manifest, &fixture.payload, &options)
}

/// Round a length to the fixture alignment.
///
/// # Errors
///
/// Rejects overflow.
pub fn aligned(value: u64) -> Result<u64, FormatError> {
    Ok(align_up(value, FIXTURE_ALIGNMENT)?)
}

/// Build the deterministic synthetic SwiGLU checkpoint used by the converter
/// fixture and the offline demo.
///
/// # Errors
///
/// Propagates generator failures.
pub fn tiny_swiglu_model()
-> Result<selvedge_convert::SyntheticModel, selvedge_convert::ConvertError> {
    selvedge_convert::swiglu_model(
        selvedge_convert::SYNTHETIC_HIDDEN,
        selvedge_convert::SYNTHETIC_INTERMEDIATE,
        selvedge_convert::SYNTHETIC_SEED,
    )
}

/// Encode the deterministic synthetic SwiGLU checkpoint as safetensors bytes.
///
/// # Errors
///
/// Propagates generation and encoding failures.
pub fn tiny_swiglu_safetensors() -> Result<Vec<u8>, selvedge_convert::ConvertError> {
    let model = tiny_swiglu_model()?;
    selvedge_convert::encode_safetensors(&model, selvedge_convert::SyntheticDType::F32)
}

/// Write the deterministic synthetic SwiGLU checkpoint.
///
/// # Errors
///
/// Propagates generation, encoding and filesystem failures.
pub fn write_tiny_swiglu_safetensors(path: &Path) -> Result<(), selvedge_convert::ConvertError> {
    let bytes = tiny_swiglu_safetensors()?;
    std::fs::write(path, bytes)?;
    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn prng_is_deterministic_and_stable() {
        let mut first = DeterministicRng::new(42);
        let mut second = DeterministicRng::new(42);
        for _ in 0..16 {
            assert_eq!(first.next_u64(), second.next_u64());
        }
        assert!(first.next_f32() >= -1.0 && first.next_f32() < 1.0);
    }

    #[test]
    fn generation_is_reproducible() {
        let first = tiny_bundle().unwrap();
        let second = tiny_bundle().unwrap();
        assert_eq!(first, second);
        assert_eq!(first.sha256().unwrap(), second.sha256().unwrap());
    }

    #[test]
    fn fixture_has_expected_roles_and_one_quantized_shard() {
        let fixture = tiny_bundle().unwrap();
        let quantized = fixture
            .manifest
            .shards
            .iter()
            .filter(|shard| shard.quant.is_some())
            .count();
        assert_eq!(quantized, 1);
        assert!(
            fixture
                .manifest
                .shards
                .iter()
                .any(|shard| shard.role == TensorRole::MlpDown)
        );
        assert!(fixture.manifest.payload_len < 100 * 1024);
    }
}
