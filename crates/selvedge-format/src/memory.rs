//! In-memory bundle for toy courts.
//!
//! A `MemoryBundle` is encoded and validated by exactly the same code paths as
//! an on-disk bundle, then answers shard reads from its own bytes.

use std::io::Cursor;

use selvedge_core::{Sha256Digest, ShardId};

use crate::error::{FormatError, Result};
use crate::manifest::Manifest;
use crate::validation::{ValidationOptions, ValidationReport, read_preamble, validate_bytes};
use crate::writer::encode_bundle_bytes;

/// A fully encoded bundle held in memory.
#[derive(Debug)]
pub struct MemoryBundle {
    bytes: Vec<u8>,
    manifest: Manifest,
    payload_offset: u64,
    manifest_sha256: Sha256Digest,
}

impl MemoryBundle {
    /// Encode and validate a manifest plus payload.
    ///
    /// # Errors
    ///
    /// Propagates encoding and validation failures.
    pub fn from_parts(
        manifest: Manifest,
        payload: &[u8],
        options: &ValidationOptions,
    ) -> Result<Self> {
        let bytes = encode_bundle_bytes(&manifest, payload)?;
        let report = validate_bytes(&bytes, options)?;
        Ok(Self {
            bytes,
            manifest,
            payload_offset: report.payload_offset,
            manifest_sha256: report.manifest_sha256,
        })
    }

    /// Adopt already-encoded bundle bytes after full validation.
    ///
    /// # Errors
    ///
    /// Propagates every validator failure.
    pub fn from_validated_bytes(bytes: Vec<u8>, options: &ValidationOptions) -> Result<Self> {
        validate_bytes(&bytes, options)?;
        let len = bytes.len() as u64;
        let mut cursor = Cursor::new(&bytes);
        let preamble = read_preamble(&mut cursor, len, options.max_manifest_bytes)?;
        let expected = preamble
            .payload_offset
            .checked_add(preamble.manifest.payload_len)
            .ok_or(FormatError::SizeOverflow {
                context: "memory bundle length",
            })?;
        if expected != len {
            return Err(FormatError::FileLengthMismatch {
                expected,
                actual: len,
            });
        }
        Ok(Self {
            bytes,
            manifest: preamble.manifest,
            payload_offset: preamble.payload_offset,
            manifest_sha256: preamble.manifest_sha256,
        })
    }

    /// Encoded bundle bytes.
    #[must_use]
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Parsed manifest.
    #[must_use]
    pub fn manifest(&self) -> &Manifest {
        &self.manifest
    }

    /// Payload start offset.
    #[must_use]
    pub const fn payload_offset(&self) -> u64 {
        self.payload_offset
    }

    /// SHA-256 of the manifest bytes.
    #[must_use]
    pub const fn manifest_sha256(&self) -> Sha256Digest {
        self.manifest_sha256
    }

    /// Payload bytes without alignment padding.
    ///
    /// # Errors
    ///
    /// Rejects a layout inconsistency.
    pub fn payload(&self) -> Result<&[u8]> {
        let start =
            usize::try_from(self.payload_offset).map_err(|_| FormatError::SizeOverflow {
                context: "payload offset",
            })?;
        let length =
            usize::try_from(self.manifest.payload_len).map_err(|_| FormatError::SizeOverflow {
                context: "payload length",
            })?;
        let end = start.checked_add(length).ok_or(FormatError::SizeOverflow {
            context: "payload end",
        })?;
        self.bytes
            .get(start..end)
            .ok_or(FormatError::ManifestInvalid {
                reason: "payload slice outside encoded bytes".to_owned(),
            })
    }

    /// Borrow one shard's bytes after bounds checking.
    ///
    /// # Errors
    ///
    /// Rejects unknown shards and out-of-bounds extents.
    pub fn read_shard(&self, shard_id: &ShardId) -> Result<&[u8]> {
        let shard = self
            .manifest
            .shard(shard_id)
            .ok_or(FormatError::UnknownShard(*shard_id))?;
        let payload = self.payload()?;
        let start = usize::try_from(shard.offset).map_err(|_| FormatError::SizeOverflow {
            context: "shard offset",
        })?;
        let end_u64 = shard
            .offset
            .checked_add(shard.length)
            .ok_or(FormatError::SizeOverflow {
                context: "shard end",
            })?;
        let end = usize::try_from(end_u64).map_err(|_| FormatError::SizeOverflow {
            context: "shard end",
        })?;
        payload
            .get(start..end)
            .ok_or(FormatError::ExtentOutOfBounds {
                shard_id: shard.shard_id,
                offset: shard.offset,
                length: shard.length,
                end: end_u64,
                payload_len: self.manifest.payload_len,
            })
    }

    /// Re-run full validation.
    ///
    /// # Errors
    ///
    /// Propagates every validator failure.
    pub fn validate(&self, options: &ValidationOptions) -> Result<ValidationReport> {
        validate_bytes(&self.bytes, options)
    }
}
