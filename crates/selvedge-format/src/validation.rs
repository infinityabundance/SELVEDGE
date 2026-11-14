//! Bundle header parsing and the validation pipeline.

use std::collections::BTreeSet;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use selvedge_core::{Sha256Builder, Sha256Digest, ShardId};
use serde::{Deserialize, Serialize};

use crate::error::{FormatError, Result};
use crate::manifest::{
    HARD_MAX_MANIFEST_BYTES, HEADER_LEN, MAGIC, MANIFEST_SCHEMA_VERSION, Manifest,
};

/// The 18-byte bundle header.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BundleHeader {
    /// Container version.
    pub version: u16,
    /// Manifest JSON length in bytes.
    pub manifest_len: u64,
}

impl BundleHeader {
    /// Encode the header.
    #[must_use]
    pub fn encode(&self) -> [u8; 18] {
        let mut out = [0_u8; 18];
        out[..8].copy_from_slice(&MAGIC);
        out[8..10].copy_from_slice(&self.version.to_le_bytes());
        out[10..18].copy_from_slice(&self.manifest_len.to_le_bytes());
        out
    }

    /// Read and check a header from a stream.
    ///
    /// # Errors
    ///
    /// Rejects truncated streams, bad magic and unsupported versions.
    pub fn read_from<R: Read>(reader: &mut R) -> Result<Self> {
        let mut header = [0_u8; 18];
        reader.read_exact(&mut header)?;
        if header[..8] != MAGIC {
            let mut found = [0_u8; 8];
            found.copy_from_slice(&header[..8]);
            return Err(FormatError::MagicMismatch { found });
        }
        let version = u16::from_le_bytes([header[8], header[9]]);
        if version != 1 {
            return Err(FormatError::UnsupportedVersion { found: version });
        }
        let mut length = [0_u8; 8];
        length.copy_from_slice(&header[10..18]);
        Ok(Self {
            version,
            manifest_len: u64::from_le_bytes(length),
        })
    }
}

/// Which shards receive SHA-256 verification.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum VerifyMode {
    /// Verify every shard extent.
    #[default]
    All,
    /// Verify only the listed shards; an unknown id is an error.
    Selected(Vec<ShardId>),
    /// Validate structure only. This is explicitly *not* integrity
    /// verification and is reported as skipped.
    StructuralOnly,
}

impl VerifyMode {
    /// Report string.
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::All => "all",
            Self::Selected(_) => "selected",
            Self::StructuralOnly => "structural_only",
        }
    }
}

/// Validation inputs and limits.
#[derive(Clone, Debug)]
pub struct ValidationOptions {
    /// Shard hash verification mode.
    pub verify: VerifyMode,
    /// Expected model digest; every shard's bundle must match it.
    pub expected_model_digest: Option<Sha256Digest>,
    /// Expected compatibility id; every shard must carry it.
    pub expected_compatibility_id: Option<String>,
    /// Manifest size cap, additionally clamped by a hard 64 MiB ceiling.
    pub max_manifest_bytes: u64,
    /// Bundle file size cap.
    pub max_bundle_bytes: u64,
    /// Optional cancellation flag, checked between chunks.
    pub cancel: Option<Arc<AtomicBool>>,
}

impl Default for ValidationOptions {
    fn default() -> Self {
        Self {
            verify: VerifyMode::All,
            expected_model_digest: None,
            expected_compatibility_id: None,
            max_manifest_bytes: 8 * 1024 * 1024,
            max_bundle_bytes: 8 * 1024 * 1024 * 1024,
            cancel: None,
        }
    }
}

impl ValidationOptions {
    /// Read-only options for structural validation only.
    #[must_use]
    pub fn structural() -> Self {
        Self {
            verify: VerifyMode::StructuralOnly,
            ..Self::default()
        }
    }

    /// Cancel flag for the validator and reader.
    #[must_use]
    pub fn cancel_flag(&self) -> Option<Arc<AtomicBool>> {
        self.cancel.clone()
    }

    /// Fail if cancellation was requested.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::Cancelled`].
    pub fn check_cancel(&self) -> Result<()> {
        if let Some(flag) = &self.cancel
            && flag.load(Ordering::Relaxed)
        {
            return Err(FormatError::Cancelled);
        }
        Ok(())
    }
}

/// Structured validation report.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ValidationReport {
    /// Report schema identifier.
    pub schema: String,
    /// Bundle id.
    pub bundle_id: String,
    /// Container version.
    pub format_version: u16,
    /// Manifest schema version.
    pub manifest_schema_version: u16,
    /// SHA-256 of the exact manifest bytes in the bundle.
    pub manifest_sha256: Sha256Digest,
    /// File length.
    pub file_len: u64,
    /// Manifest length.
    pub manifest_len: u64,
    /// Payload start offset.
    pub payload_offset: u64,
    /// Payload length (a multiple of the alignment).
    pub payload_len: u64,
    /// Extent alignment.
    pub alignment: u64,
    /// Model digest declared by the manifest.
    pub model_digest: Sha256Digest,
    /// Number of shard records.
    pub shard_count: u64,
    /// Sum of shard extent lengths (padding excluded).
    pub total_shard_bytes: u64,
    /// Sorted unique compatibility ids.
    pub compatibility_ids: Vec<String>,
    /// Shards whose hashes were verified.
    pub verified_shards: Vec<ShardId>,
    /// Shards whose hashes were not verified.
    pub skipped_shards: Vec<ShardId>,
    /// Verification mode used.
    pub verify_mode: String,
}

/// Internal parse result shared with the reader.
pub(crate) struct Preamble {
    pub(crate) manifest: Manifest,
    pub(crate) manifest_len: u64,
    pub(crate) manifest_sha256: Sha256Digest,
    pub(crate) payload_offset: u64,
}

pub(crate) fn read_preamble<R: Read + Seek>(
    reader: &mut R,
    file_len: u64,
    max_manifest_bytes: u64,
) -> Result<Preamble> {
    if file_len < HEADER_LEN {
        return Err(FormatError::HeaderTruncated { len: file_len });
    }
    reader.seek(SeekFrom::Start(0))?;
    let header = BundleHeader::read_from(reader)?;
    let manifest_cap = max_manifest_bytes.min(HARD_MAX_MANIFEST_BYTES);
    if header.manifest_len > manifest_cap {
        return Err(FormatError::ManifestTooLarge {
            len: header.manifest_len,
            cap: manifest_cap,
        });
    }
    if header.manifest_len > file_len - HEADER_LEN {
        return Err(FormatError::ManifestOutOfBounds {
            len: header.manifest_len,
            file_len,
        });
    }
    let manifest_len = header.manifest_len;
    let manifest_bytes = read_bounded(reader, manifest_len)?;
    let manifest_sha256 = Sha256Digest::of(&manifest_bytes);
    let text = std::str::from_utf8(&manifest_bytes).map_err(|_| FormatError::ManifestNotUtf8)?;
    let manifest: Manifest = serde_json::from_str(text).map_err(FormatError::ManifestParse)?;
    manifest.validate()?;
    let payload_offset = manifest.payload_offset(manifest_len)?;
    Ok(Preamble {
        manifest,
        manifest_len,
        manifest_sha256,
        payload_offset,
    })
}

fn read_bounded<R: Read>(reader: &mut R, length: u64) -> Result<Vec<u8>> {
    let capacity = usize::try_from(length).map_err(|_| FormatError::Allocation {
        bytes: length,
        reason: "length does not fit in memory".to_owned(),
    })?;
    let mut buffer = Vec::new();
    buffer
        .try_reserve_exact(capacity)
        .map_err(|err| FormatError::Allocation {
            bytes: length,
            reason: err.to_string(),
        })?;
    buffer.resize(capacity, 0);
    reader.read_exact(&mut buffer)?;
    Ok(buffer)
}

/// Validate a bundle file on disk.
///
/// # Errors
///
/// Returns the first typed failure: length bounds, magic, version, manifest cap
/// or parse, semantic validation, extents, alignment, hashes, expected digest
/// or compatibility id, or cancellation.
pub fn validate_bundle(path: &Path, options: &ValidationOptions) -> Result<ValidationReport> {
    let mut file = std::fs::File::open(path)?;
    let file_len = file.metadata()?.len();
    validate_reader(&mut file, file_len, options)
}

/// Validate a bundle held in memory.
///
/// # Errors
///
/// Same failures as [`validate_bundle`].
pub fn validate_bytes(bytes: &[u8], options: &ValidationOptions) -> Result<ValidationReport> {
    let len = bytes.len() as u64;
    let mut cursor = std::io::Cursor::new(bytes);
    validate_reader(&mut cursor, len, options)
}

/// Validate any `Read + Seek` source with a known length.
///
/// # Errors
///
/// Same failures as [`validate_bundle`].
pub fn validate_reader<R: Read + Seek>(
    reader: &mut R,
    file_len: u64,
    options: &ValidationOptions,
) -> Result<ValidationReport> {
    options.check_cancel()?;
    if file_len > options.max_bundle_bytes {
        return Err(FormatError::BundleTooLarge {
            len: file_len,
            cap: options.max_bundle_bytes,
        });
    }
    let preamble = read_preamble(reader, file_len, options.max_manifest_bytes)?;
    options.check_cancel()?;

    let expected_len = preamble
        .payload_offset
        .checked_add(preamble.manifest.payload_len)
        .ok_or(FormatError::SizeOverflow {
            context: "bundle layout length",
        })?;
    if expected_len != file_len {
        return Err(FormatError::FileLengthMismatch {
            expected: expected_len,
            actual: file_len,
        });
    }

    if let Some(expected) = options.expected_model_digest {
        let actual = preamble.manifest.model.digest();
        if actual != expected {
            return Err(FormatError::ModelDigestMismatch { expected, actual });
        }
    }
    if let Some(expected) = &options.expected_compatibility_id {
        for shard in &preamble.manifest.shards {
            if shard.compatibility_id != *expected {
                return Err(FormatError::CompatibilityMismatch {
                    expected: expected.clone(),
                    actual: shard.compatibility_id.clone(),
                });
            }
        }
    }

    let selected: Option<BTreeSet<ShardId>> = match &options.verify {
        VerifyMode::All => None,
        VerifyMode::StructuralOnly => Some(BTreeSet::new()),
        VerifyMode::Selected(ids) => {
            for id in ids {
                if preamble.manifest.shard(id).is_none() {
                    return Err(FormatError::UnknownSelectedShard(*id));
                }
            }
            Some(ids.iter().copied().collect())
        }
    };

    let mut verified = Vec::new();
    let mut skipped = Vec::new();
    let mut scratch = vec![0_u8; 64 * 1024];
    for shard in &preamble.manifest.shards {
        let should_verify = selected
            .as_ref()
            .is_none_or(|set| set.contains(&shard.shard_id));
        if should_verify {
            verify_extent(
                reader,
                preamble.payload_offset,
                shard,
                options,
                &mut scratch,
            )?;
            verified.push(shard.shard_id);
        } else {
            skipped.push(shard.shard_id);
        }
    }
    options.check_cancel()?;

    Ok(ValidationReport {
        schema: "selvedge/validation-report/v1".to_owned(),
        bundle_id: preamble.manifest.bundle_id.clone(),
        format_version: 1,
        manifest_schema_version: MANIFEST_SCHEMA_VERSION,
        manifest_sha256: preamble.manifest_sha256,
        file_len,
        manifest_len: preamble.manifest_len,
        payload_offset: preamble.payload_offset,
        payload_len: preamble.manifest.payload_len,
        alignment: preamble.manifest.alignment,
        model_digest: preamble.manifest.model.digest(),
        shard_count: preamble.manifest.shards.len() as u64,
        total_shard_bytes: preamble.manifest.total_shard_bytes()?,
        compatibility_ids: preamble.manifest.compatibility_ids(),
        verified_shards: verified,
        skipped_shards: skipped,
        verify_mode: options.verify.as_str().to_owned(),
    })
}

fn verify_extent<R: Read + Seek>(
    reader: &mut R,
    payload_offset: u64,
    shard: &crate::manifest::ShardRecord,
    options: &ValidationOptions,
    scratch: &mut [u8],
) -> Result<()> {
    let start = payload_offset
        .checked_add(shard.offset)
        .ok_or(FormatError::SizeOverflow {
            context: "extent file offset",
        })?;
    reader.seek(SeekFrom::Start(start))?;
    let mut remaining = shard.length;
    let mut builder = Sha256Builder::new();
    while remaining > 0 {
        options.check_cancel()?;
        let want = usize::try_from(remaining.min(scratch.len() as u64)).map_err(|_| {
            FormatError::SizeOverflow {
                context: "extent chunk length",
            }
        })?;
        let read = reader.read(&mut scratch[..want])?;
        if read == 0 {
            return Err(FormatError::PayloadShort {
                expected: shard.length,
                actual: shard.length - remaining,
            });
        }
        builder.update(&scratch[..read]);
        remaining -= read as u64;
    }
    let actual = builder.finish();
    if actual != shard.sha256 {
        return Err(FormatError::HashMismatch {
            shard_id: shard.shard_id,
            expected: shard.sha256,
            actual,
        });
    }
    Ok(())
}
