//! Manifest v1: the self-describing index of a bundle.
//!
//! Every shard id is recomputed from the semantic fields during validation, so
//! an attacker cannot rename or re-label a shard without invalidating it.

use std::collections::BTreeSet;

use selvedge_core::{
    ChannelSet, DType, FORMAT_VERSION, ModelDigest, QuantScheme, Sha256Digest, ShardId,
    ShardIdentity, StorageOrder, TensorRole, align_up, bytes_for, element_count,
};
use serde::{Deserialize, Serialize};

use crate::error::{FormatError, Result};

/// Bundle magic bytes.
pub const MAGIC: [u8; 8] = *b"SLVDGFMT";

/// Header length: 8 magic + 2 version + 8 manifest length.
pub const HEADER_LEN: u64 = 18;

/// Manifest schema identifier.
pub const MANIFEST_SCHEMA: &str = "selvedge/shard-manifest/v1";

/// Manifest schema version.
pub const MANIFEST_SCHEMA_VERSION: u16 = 1;

/// Default shard extent alignment. A documented baseline, not a claim of
/// optimality on any particular device.
pub const DEFAULT_ALIGNMENT: u64 = 4096;

/// Smallest alignment the format accepts.
pub const MIN_ALIGNMENT: u64 = 512;

/// Hard cap on parsed manifest bytes, independent of configuration.
pub const HARD_MAX_MANIFEST_BYTES: u64 = 64 * 1024 * 1024;

/// Hard cap on shard records, to bound validation work.
pub const MAX_SHARDS: usize = 1_000_000;

/// An auxiliary extent inside a shard's bytes (scales, zero points).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AuxExtent {
    /// Offset from the start of the shard.
    pub offset: u64,
    /// Length in bytes.
    pub length: u64,
    /// Element type of the auxiliary values.
    pub dtype: DType,
}

/// Block quantization description for a shard.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QuantizationSpec {
    /// Logical elements per block.
    pub block_size: u32,
    /// Scales extent (required for quantized shards).
    pub scales: Option<AuxExtent>,
    /// Zero-point extent; present only for affine quantization.
    pub zero_points: Option<AuxExtent>,
}

impl QuantizationSpec {
    /// Scheme used in the shard identity.
    #[must_use]
    pub fn scheme(&self) -> QuantScheme {
        if self.zero_points.is_some() {
            QuantScheme::Affine {
                block_size: self.block_size,
                zero_points: true,
            }
        } else {
            QuantScheme::Symmetric {
                block_size: self.block_size,
            }
        }
    }

    /// Total auxiliary bytes.
    ///
    /// # Errors
    ///
    /// Rejects overflow.
    pub fn auxiliary_bytes(&self) -> Result<u64> {
        let mut total = 0_u64;
        for extent in [&self.scales, &self.zero_points].into_iter().flatten() {
            total = total
                .checked_add(extent.length)
                .ok_or(FormatError::SizeOverflow {
                    context: "auxiliary bytes",
                })?;
        }
        Ok(total)
    }
}

/// Calibration/training provenance recorded with a shard.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CalibrationProvenance {
    /// Method name (for example `synthetic-fixture` or an activation
    /// calibration procedure).
    pub method: String,
    /// Digest of the calibration dataset, when one was used.
    pub dataset_digest: Option<Sha256Digest>,
    /// Free-form notes that never stand in for evidence.
    pub notes: Option<String>,
}

impl CalibrationProvenance {
    /// Validate the provenance block.
    ///
    /// # Errors
    ///
    /// Rejects an empty method name or one longer than 256 bytes.
    pub fn validate(&self) -> Result<()> {
        if self.method.is_empty() || self.method.len() > 256 {
            return Err(FormatError::ManifestInvalid {
                reason: "calibration method must be 1..=256 bytes".to_owned(),
            });
        }
        Ok(())
    }
}

/// One shard record.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ShardRecord {
    /// Content-derived shard id.
    pub shard_id: ShardId,
    /// Tensor role.
    pub role: TensorRole,
    /// Optional layer index.
    pub layer: Option<u32>,
    /// Channel range or explicit list.
    pub channels: ChannelSet,
    /// Logical axis the channel set applies to.
    pub channel_axis: u32,
    /// Logical shape of the stored slice.
    pub shape: Vec<u64>,
    /// Actual on-disk orientation.
    pub storage_order: StorageOrder,
    /// Element type.
    pub dtype: DType,
    /// Quantization description for quantized dtypes.
    pub quant: Option<QuantizationSpec>,
    /// Payload-relative offset; must be aligned.
    pub offset: u64,
    /// Extent length in bytes, including auxiliary extents.
    pub length: u64,
    /// SHA-256 of the whole extent (weights followed by auxiliary bytes).
    pub sha256: Sha256Digest,
    /// Alignment this extent was written with; must equal the bundle alignment.
    pub alignment: u64,
    /// Compatibility identifier shared by shards that may execute together.
    pub compatibility_id: String,
    /// Transform/repacking identifier; must be listed in the bundle.
    pub transform_id: String,
    /// Shards expected to be co-accessed.
    pub co_access: Vec<ShardId>,
    /// Shards that must be loaded first.
    pub depends_on: Vec<ShardId>,
    /// Optional calibration provenance.
    pub calibration: Option<CalibrationProvenance>,
}

impl ShardRecord {
    /// Logical element count.
    ///
    /// # Errors
    ///
    /// Rejects malformed shapes and overflow.
    pub fn element_count(&self) -> Result<u64> {
        element_count(&self.shape).map_err(FormatError::Core)
    }

    /// Bytes occupied by the weight values (excluding auxiliary bytes).
    ///
    /// # Errors
    ///
    /// Rejects unsupported dtypes and overflow.
    pub fn weight_bytes(&self) -> Result<u64> {
        let elements = self.element_count()?;
        bytes_for(elements, &self.dtype).map_err(|err| FormatError::ManifestInvalid {
            reason: format!("shard {}: {err}", self.shard_id.short()),
        })
    }

    /// Auxiliary bytes (scales and zero points).
    ///
    /// # Errors
    ///
    /// Rejects overflow.
    pub fn auxiliary_bytes(&self) -> Result<u64> {
        self.quant
            .as_ref()
            .map_or(Ok(0), QuantizationSpec::auxiliary_bytes)
    }

    /// Semantic identity of the shard.
    #[must_use]
    pub fn identity(&self, model: &ModelDigest) -> ShardIdentity {
        ShardIdentity {
            format_version: FORMAT_VERSION,
            model: model.clone(),
            role: self.role.clone(),
            layer: self.layer,
            channels: self.channels.clone(),
            channel_axis: self.channel_axis,
            dtype: self.dtype.clone(),
            quant: self.quant.as_ref().map(QuantizationSpec::scheme),
            transform_id: self.transform_id.clone(),
        }
    }

    /// Recompute the content-derived id.
    ///
    /// # Errors
    ///
    /// Rejects invalid identity fields.
    pub fn recompute_id(&self, model: &ModelDigest) -> Result<ShardId> {
        ShardId::derive(&self.identity(model)).map_err(FormatError::Core)
    }

    /// Validate the record against the bundle-level fields.
    ///
    /// # Errors
    ///
    /// Rejects every physical or semantic inconsistency described in
    /// `docs/SHARD-FORMAT.md`.
    pub fn validate(
        &self,
        model: &ModelDigest,
        manifest_alignment: u64,
        payload_len: u64,
        transforms: &[String],
    ) -> Result<()> {
        self.channels
            .validate()
            .map_err(|err| FormatError::ManifestInvalid {
                reason: format!("shard {}: {err}", self.shard_id.short()),
            })?;
        let channel_count = self.channels.count();
        if self.shape.is_empty() || self.shape.len() > 16 {
            return Err(FormatError::ManifestInvalid {
                reason: format!(
                    "shard {}: shape rank {} outside 1..=16",
                    self.shard_id.short(),
                    self.shape.len()
                ),
            });
        }
        if self.shape.contains(&0) {
            return Err(FormatError::ManifestInvalid {
                reason: format!("shard {}: zero dimension", self.shard_id.short()),
            });
        }
        let axis =
            usize::try_from(self.channel_axis).map_err(|_| FormatError::ManifestInvalid {
                reason: format!("shard {}: channel axis does not fit", self.shard_id.short()),
            })?;
        if axis >= self.shape.len() || self.shape[axis] != channel_count {
            return Err(FormatError::ManifestInvalid {
                reason: format!(
                    "shard {}: shape[{}]={} does not equal channel count {channel_count}",
                    self.shard_id.short(),
                    self.channel_axis,
                    self.shape.get(axis).copied().unwrap_or(0)
                ),
            });
        }
        self.storage_order
            .validate(self.shape.len())
            .map_err(|err| FormatError::ManifestInvalid {
                reason: format!("shard {}: {err}", self.shard_id.short()),
            })?;
        if let DType::Unsupported(raw) = &self.dtype {
            return Err(FormatError::UnsupportedDType {
                shard_id: self.shard_id,
                dtype: raw.clone(),
            });
        }
        if self.length == 0 {
            return Err(FormatError::ManifestInvalid {
                reason: format!("shard {}: zero length", self.shard_id.short()),
            });
        }
        let elements = self.element_count()?;
        let expected_weight_bytes = self.weight_bytes()?;
        match (&self.quant, self.dtype.is_quantized()) {
            (None, true) => {
                return Err(FormatError::QuantizationInvalid {
                    shard_id: self.shard_id,
                    reason: "quantized dtype without a quantization description".to_owned(),
                });
            }
            (Some(_), false) => {
                return Err(FormatError::QuantizationInvalid {
                    shard_id: self.shard_id,
                    reason: format!("dtype {} must not be quantized", self.dtype),
                });
            }
            (Some(spec), true) => {
                validate_quantization(self, spec, elements, expected_weight_bytes)?;
            }
            (None, false) => {
                if self.length != expected_weight_bytes {
                    return Err(FormatError::ByteLengthMismatch {
                        shard_id: self.shard_id,
                        expected: expected_weight_bytes,
                        actual: self.length,
                        elements,
                        dtype: self.dtype.stable_name(),
                    });
                }
            }
        }
        let end = self
            .offset
            .checked_add(self.length)
            .ok_or(FormatError::SizeOverflow {
                context: "shard extent end",
            })?;
        if end > payload_len {
            return Err(FormatError::ExtentOutOfBounds {
                shard_id: self.shard_id,
                offset: self.offset,
                length: self.length,
                end,
                payload_len,
            });
        }
        if self.alignment != manifest_alignment {
            return Err(FormatError::ManifestInvalid {
                reason: format!(
                    "shard {}: alignment {} differs from bundle alignment {manifest_alignment}",
                    self.shard_id.short(),
                    self.alignment
                ),
            });
        }
        if !self.offset.is_multiple_of(manifest_alignment) {
            return Err(FormatError::AlignmentViolation {
                shard_id: self.shard_id,
                offset: self.offset,
                alignment: manifest_alignment,
            });
        }
        validate_label("compatibility_id", &self.compatibility_id)?;
        if !transforms.iter().any(|id| id == &self.transform_id) {
            return Err(FormatError::ManifestInvalid {
                reason: format!(
                    "shard {}: transform {:?} is not listed in the bundle",
                    self.shard_id.short(),
                    self.transform_id
                ),
            });
        }
        if let Some(calibration) = &self.calibration {
            calibration.validate()?;
        }
        let recomputed = self.recompute_id(model)?;
        if recomputed != self.shard_id {
            return Err(FormatError::ShardIdMismatch {
                shard_id: self.shard_id,
                recomputed,
            });
        }
        Ok(())
    }
}

fn validate_quantization(
    shard: &ShardRecord,
    spec: &QuantizationSpec,
    elements: u64,
    weight_bytes: u64,
) -> Result<()> {
    let block = u64::from(spec.block_size);
    if block == 0 {
        return Err(FormatError::QuantizationInvalid {
            shard_id: shard.shard_id,
            reason: "block size is zero".to_owned(),
        });
    }
    if block > elements || !elements.is_multiple_of(block) {
        return Err(FormatError::QuantizationInvalid {
            shard_id: shard.shard_id,
            reason: format!("{elements} elements are not whole blocks of {block}"),
        });
    }
    if let DType::I4Packed { block_size } | DType::TernaryPacked { block_size } = shard.dtype
        && block_size != spec.block_size
    {
        return Err(FormatError::QuantizationInvalid {
            shard_id: shard.shard_id,
            reason: format!(
                "packed dtype block {block_size} does not match quantization block {}",
                spec.block_size
            ),
        });
    }
    let blocks = elements / block;
    let scales = spec
        .scales
        .as_ref()
        .ok_or(FormatError::QuantizationInvalid {
            shard_id: shard.shard_id,
            reason: "scales extent is required".to_owned(),
        })?;
    let scale_bytes = unpacked_bytes_per_element(&scales.dtype, shard.shard_id)?;
    let expected_scales_length =
        blocks
            .checked_mul(scale_bytes)
            .ok_or(FormatError::SizeOverflow {
                context: "scales length",
            })?;
    if scales.length != expected_scales_length {
        return Err(FormatError::QuantizationInvalid {
            shard_id: shard.shard_id,
            reason: format!(
                "scales length {} does not equal {expected_scales_length}",
                scales.length
            ),
        });
    }
    if scales.offset != weight_bytes {
        return Err(FormatError::QuantizationInvalid {
            shard_id: shard.shard_id,
            reason: format!(
                "scales offset {} must immediately follow {weight_bytes} weight bytes",
                scales.offset
            ),
        });
    }
    let mut expected_end =
        weight_bytes
            .checked_add(scales.length)
            .ok_or(FormatError::SizeOverflow {
                context: "scales end",
            })?;
    if let Some(zero_points) = &spec.zero_points {
        let zero_bytes = unpacked_bytes_per_element(&zero_points.dtype, shard.shard_id)?;
        let expected_zero_length =
            blocks
                .checked_mul(zero_bytes)
                .ok_or(FormatError::SizeOverflow {
                    context: "zero-point length",
                })?;
        if zero_points.length != expected_zero_length {
            return Err(FormatError::QuantizationInvalid {
                shard_id: shard.shard_id,
                reason: format!(
                    "zero-point length {} does not equal {expected_zero_length}",
                    zero_points.length
                ),
            });
        }
        if zero_points.offset != expected_end {
            return Err(FormatError::QuantizationInvalid {
                shard_id: shard.shard_id,
                reason: format!(
                    "zero-point offset {} must equal {expected_end}",
                    zero_points.offset
                ),
            });
        }
        expected_end =
            expected_end
                .checked_add(zero_points.length)
                .ok_or(FormatError::SizeOverflow {
                    context: "zero-point end",
                })?;
    }
    if shard.length != expected_end {
        return Err(FormatError::ByteLengthMismatch {
            shard_id: shard.shard_id,
            expected: expected_end,
            actual: shard.length,
            elements,
            dtype: shard.dtype.stable_name(),
        });
    }
    Ok(())
}

fn unpacked_bytes_per_element(dtype: &DType, shard_id: ShardId) -> Result<u64> {
    if dtype.is_packed() || !dtype.is_supported() {
        return Err(FormatError::QuantizationInvalid {
            shard_id,
            reason: format!("auxiliary dtype {dtype} must be an unpacked numeric type"),
        });
    }
    bytes_for(1, dtype).map_err(|err| FormatError::QuantizationInvalid {
        shard_id,
        reason: err.to_string(),
    })
}

fn validate_label(field: &str, value: &str) -> Result<()> {
    if value.is_empty() || value.len() > 128 {
        return Err(FormatError::ManifestInvalid {
            reason: format!("{field} must be 1..=128 bytes"),
        });
    }
    if !value.bytes().all(|byte| {
        byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b':' | b'/' | b'-')
    }) {
        return Err(FormatError::ManifestInvalid {
            reason: format!("{field} contains characters outside A-Za-z0-9._:/-"),
        });
    }
    Ok(())
}

/// The complete manifest.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
    /// Schema identifier, exactly [`MANIFEST_SCHEMA`].
    pub schema: String,
    /// Schema version, exactly [`MANIFEST_SCHEMA_VERSION`].
    pub schema_version: u16,
    /// Stable bundle identifier (also used in file names).
    pub bundle_id: String,
    /// Source model identity.
    pub model: ModelDigest,
    /// Transform ids that shards may reference; sorted and unique.
    pub transforms: Vec<String>,
    /// Extent alignment in bytes.
    pub alignment: u64,
    /// Payload length in bytes, a multiple of `alignment`.
    pub payload_len: u64,
    /// Producing tool and version.
    pub created_by: String,
    /// Shard records, sorted by shard id.
    pub shards: Vec<ShardRecord>,
}

impl Manifest {
    /// Compact JSON bytes of the manifest.
    ///
    /// # Errors
    ///
    /// Propagates serialization failure.
    pub fn manifest_bytes(&self) -> Result<Vec<u8>> {
        serde_json::to_vec(self).map_err(FormatError::ManifestParse)
    }

    /// Payload offset for a manifest of `manifest_len` bytes.
    ///
    /// # Errors
    ///
    /// Rejects overflow.
    pub fn payload_offset(&self, manifest_len: u64) -> Result<u64> {
        let end = HEADER_LEN
            .checked_add(manifest_len)
            .ok_or(FormatError::SizeOverflow {
                context: "manifest end",
            })?;
        Ok(align_up(end, self.alignment)?)
    }

    /// Find a shard by id.
    #[must_use]
    pub fn shard(&self, id: &ShardId) -> Option<&ShardRecord> {
        self.shards.iter().find(|shard| shard.shard_id == *id)
    }

    /// Total shard extent bytes (excluding alignment padding).
    ///
    /// # Errors
    ///
    /// Rejects overflow.
    pub fn total_shard_bytes(&self) -> Result<u64> {
        let mut total = 0_u64;
        for shard in &self.shards {
            total = total
                .checked_add(shard.length)
                .ok_or(FormatError::SizeOverflow {
                    context: "total shard bytes",
                })?;
        }
        Ok(total)
    }

    /// Sorted unique compatibility ids.
    #[must_use]
    pub fn compatibility_ids(&self) -> Vec<String> {
        let ids: BTreeSet<String> = self
            .shards
            .iter()
            .map(|shard| shard.compatibility_id.clone())
            .collect();
        ids.into_iter().collect()
    }

    /// Validate the whole manifest.
    ///
    /// # Errors
    ///
    /// Rejects every semantic, structural and physical inconsistency described
    /// in `docs/SHARD-FORMAT.md`.
    pub fn validate(&self) -> Result<()> {
        if self.schema != MANIFEST_SCHEMA {
            return Err(FormatError::ManifestInvalid {
                reason: format!("schema {:?} is not {MANIFEST_SCHEMA:?}", self.schema),
            });
        }
        if self.schema_version != MANIFEST_SCHEMA_VERSION {
            return Err(FormatError::ManifestInvalid {
                reason: format!(
                    "schema version {} is not {MANIFEST_SCHEMA_VERSION}",
                    self.schema_version
                ),
            });
        }
        validate_bundle_id(&self.bundle_id)?;
        if self.created_by.is_empty() || self.created_by.len() > 256 {
            return Err(FormatError::ManifestInvalid {
                reason: "created_by must be 1..=256 bytes".to_owned(),
            });
        }
        if !self.alignment.is_power_of_two() || self.alignment < MIN_ALIGNMENT {
            return Err(FormatError::AlignmentNotPowerOfTwo(self.alignment));
        }
        if self.payload_len == 0 || !self.payload_len.is_multiple_of(self.alignment) {
            return Err(FormatError::ManifestInvalid {
                reason: format!(
                    "payload length {} is zero or not a multiple of {}",
                    self.payload_len, self.alignment
                ),
            });
        }
        if self.transforms.is_empty() {
            return Err(FormatError::ManifestInvalid {
                reason: "transforms list is empty".to_owned(),
            });
        }
        for pair in self.transforms.windows(2) {
            if pair[0] >= pair[1] {
                return Err(FormatError::ManifestInvalid {
                    reason: "transforms must be sorted and unique".to_owned(),
                });
            }
        }
        for transform in &self.transforms {
            validate_label("transform_id", transform)?;
        }
        if self.shards.is_empty() {
            return Err(FormatError::ManifestInvalid {
                reason: "manifest has no shards".to_owned(),
            });
        }
        if self.shards.len() > MAX_SHARDS {
            return Err(FormatError::ManifestInvalid {
                reason: format!("shard count {} exceeds {MAX_SHARDS}", self.shards.len()),
            });
        }

        let mut ids = BTreeSet::new();
        for shard in &self.shards {
            if !ids.insert(shard.shard_id) {
                return Err(FormatError::DuplicateShardId(shard.shard_id));
            }
            shard.validate(
                &self.model,
                self.alignment,
                self.payload_len,
                &self.transforms,
            )?;
        }

        for shard in &self.shards {
            for (field, links) in [
                ("co_access", &shard.co_access),
                ("depends_on", &shard.depends_on),
            ] {
                for pair in links.windows(2) {
                    if pair[0] >= pair[1] {
                        return Err(FormatError::ManifestInvalid {
                            reason: format!(
                                "shard {}: {field} must be sorted and unique",
                                shard.shard_id.short()
                            ),
                        });
                    }
                }
                for target in links {
                    if *target == shard.shard_id {
                        return Err(FormatError::SelfLink {
                            shard_id: shard.shard_id,
                        });
                    }
                    if !ids.contains(target) {
                        return Err(FormatError::DanglingLink {
                            shard_id: shard.shard_id,
                            target: *target,
                        });
                    }
                }
            }
        }

        validate_extents(&self.shards, self.payload_len)?;
        Ok(())
    }
}

fn validate_extents(shards: &[ShardRecord], payload_len: u64) -> Result<()> {
    let mut extents: Vec<(u64, u64, ShardId)> = shards
        .iter()
        .map(|shard| {
            let end = shard
                .offset
                .checked_add(shard.length)
                .ok_or(FormatError::SizeOverflow {
                    context: "extent end",
                })?;
            if end > payload_len {
                return Err(FormatError::ExtentOutOfBounds {
                    shard_id: shard.shard_id,
                    offset: shard.offset,
                    length: shard.length,
                    end,
                    payload_len,
                });
            }
            Ok((shard.offset, end, shard.shard_id))
        })
        .collect::<Result<_>>()?;
    extents.sort_unstable();
    for pair in extents.windows(2) {
        if pair[0].1 > pair[1].0 {
            return Err(FormatError::OverlappingExtents {
                first: pair[0].2,
                second: pair[1].2,
            });
        }
    }
    Ok(())
}

fn validate_bundle_id(bundle_id: &str) -> Result<()> {
    if bundle_id.is_empty() || bundle_id.len() > 128 {
        return Err(FormatError::ManifestInvalid {
            reason: "bundle_id must be 1..=128 bytes".to_owned(),
        });
    }
    if bundle_id.starts_with('.') {
        return Err(FormatError::ManifestInvalid {
            reason: "bundle_id must not start with '.'".to_owned(),
        });
    }
    if !bundle_id
        .bytes()
        .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-'))
    {
        return Err(FormatError::ManifestInvalid {
            reason: "bundle_id contains characters outside A-Za-z0-9._-".to_owned(),
        });
    }
    Ok(())
}
