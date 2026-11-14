//! Random-access bundle reader.
//!
//! Reads re-verify the shard hash on every access and fail closed on mismatch.
//! Cancellation is checked between chunks and leaves the reader usable: each
//! read seeks explicitly before it starts.

use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;

use selvedge_core::{Sha256Builder, Sha256Digest, ShardId};

use crate::error::{FormatError, Result};
use crate::manifest::{Manifest, ShardRecord};
use crate::validation::{ValidationOptions, read_preamble};

/// An open, structurally validated bundle.
#[derive(Debug)]
pub struct BundleFile {
    file: std::fs::File,
    manifest: Manifest,
    payload_offset: u64,
    file_len: u64,
    manifest_len: u64,
    manifest_sha256: Sha256Digest,
    path: PathBuf,
}

impl BundleFile {
    /// Open a bundle with the default 8 MiB manifest cap.
    ///
    /// # Errors
    ///
    /// Rejects missing/truncated files, bad magic or version, an oversized or
    /// malformed manifest, a layout/length mismatch, and manifest semantic
    /// failures. Payload hashes are verified on shard access, not at open.
    pub fn open(path: &Path) -> Result<Self> {
        Self::open_with_manifest_cap(path, 8 * 1024 * 1024)
    }

    /// Open a bundle with an explicit manifest cap.
    ///
    /// # Errors
    ///
    /// Same failures as [`BundleFile::open`].
    pub fn open_with_manifest_cap(path: &Path, max_manifest_bytes: u64) -> Result<Self> {
        let mut file = std::fs::File::open(path)?;
        let file_len = file.metadata()?.len();
        let preamble = read_preamble(&mut file, file_len, max_manifest_bytes)?;
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
        Ok(Self {
            file,
            manifest: preamble.manifest,
            payload_offset: preamble.payload_offset,
            file_len,
            manifest_len: preamble.manifest_len,
            manifest_sha256: preamble.manifest_sha256,
            path: path.to_path_buf(),
        })
    }

    /// Parsed manifest.
    #[must_use]
    pub fn manifest(&self) -> &Manifest {
        &self.manifest
    }

    /// Payload start offset in the file.
    #[must_use]
    pub const fn payload_offset(&self) -> u64 {
        self.payload_offset
    }

    /// File length.
    #[must_use]
    pub const fn file_len(&self) -> u64 {
        self.file_len
    }

    /// Manifest length.
    #[must_use]
    pub const fn manifest_len(&self) -> u64 {
        self.manifest_len
    }

    /// SHA-256 of the exact manifest bytes.
    #[must_use]
    pub const fn manifest_sha256(&self) -> Sha256Digest {
        self.manifest_sha256
    }

    /// Path this bundle was opened from.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Read and verify one shard.
    ///
    /// # Errors
    ///
    /// Rejects unknown shards, a shard larger than `max_bytes`, allocation
    /// refusal, short reads, hash mismatch, and overflow.
    pub fn read_shard(&mut self, shard_id: &ShardId, max_bytes: u64) -> Result<Vec<u8>> {
        self.read_shard_cancellable(shard_id, max_bytes, None)
    }

    /// Read and verify one shard, aborting when `cancel` is set.
    ///
    /// # Errors
    ///
    /// Adds [`FormatError::Cancelled`] to the failures of
    /// [`BundleFile::read_shard`].
    pub fn read_shard_cancellable(
        &mut self,
        shard_id: &ShardId,
        max_bytes: u64,
        cancel: Option<&AtomicBool>,
    ) -> Result<Vec<u8>> {
        let shard = self
            .manifest
            .shard(shard_id)
            .cloned()
            .ok_or(FormatError::UnknownShard(*shard_id))?;
        check_cancel(cancel)?;
        if shard.length > max_bytes {
            return Err(FormatError::Allocation {
                bytes: shard.length,
                reason: format!("exceeds caller cap of {max_bytes} bytes"),
            });
        }
        let length = usize::try_from(shard.length).map_err(|_| FormatError::Allocation {
            bytes: shard.length,
            reason: "length does not fit in memory".to_owned(),
        })?;
        let mut buffer = Vec::new();
        buffer
            .try_reserve_exact(length)
            .map_err(|err| FormatError::Allocation {
                bytes: shard.length,
                reason: err.to_string(),
            })?;
        buffer.resize(length, 0);
        let start =
            self.payload_offset
                .checked_add(shard.offset)
                .ok_or(FormatError::SizeOverflow {
                    context: "shard file offset",
                })?;
        self.file.seek(SeekFrom::Start(start))?;
        let mut read = 0_usize;
        let mut hasher = Sha256Builder::new();
        while read < length {
            check_cancel(cancel)?;
            let count = self.file.read(&mut buffer[read..])?;
            if count == 0 {
                return Err(FormatError::PayloadShort {
                    expected: shard.length,
                    actual: read as u64,
                });
            }
            hasher.update(&buffer[read..read + count]);
            read += count;
        }
        let actual = hasher.finish();
        if actual != shard.sha256 {
            return Err(FormatError::HashMismatch {
                shard_id: shard.shard_id,
                expected: shard.sha256,
                actual,
            });
        }
        Ok(buffer)
    }

    /// Verify one shard without retaining its bytes.
    ///
    /// # Errors
    ///
    /// Same failures as [`BundleFile::read_shard`], plus cancellation.
    pub fn verify_shard(&mut self, shard_id: &ShardId, cancel: Option<&AtomicBool>) -> Result<()> {
        let shard = self
            .manifest
            .shard(shard_id)
            .cloned()
            .ok_or(FormatError::UnknownShard(*shard_id))?;
        let start =
            self.payload_offset
                .checked_add(shard.offset)
                .ok_or(FormatError::SizeOverflow {
                    context: "shard file offset",
                })?;
        self.file.seek(SeekFrom::Start(start))?;
        let mut scratch = vec![0_u8; 64 * 1024];
        let mut remaining = shard.length;
        let mut hasher = Sha256Builder::new();
        while remaining > 0 {
            check_cancel(cancel)?;
            let want = usize::try_from(remaining.min(scratch.len() as u64)).map_err(|_| {
                FormatError::SizeOverflow {
                    context: "verify chunk",
                }
            })?;
            let count = self.file.read(&mut scratch[..want])?;
            if count == 0 {
                return Err(FormatError::PayloadShort {
                    expected: shard.length,
                    actual: shard.length - remaining,
                });
            }
            hasher.update(&scratch[..count]);
            remaining -= count as u64;
        }
        let actual = hasher.finish();
        if actual != shard.sha256 {
            return Err(FormatError::HashMismatch {
                shard_id: shard.shard_id,
                expected: shard.sha256,
                actual,
            });
        }
        Ok(())
    }

    /// Look up a shard record.
    #[must_use]
    pub fn shard(&self, shard_id: &ShardId) -> Option<&ShardRecord> {
        self.manifest.shard(shard_id)
    }

    /// Validate the complete bundle using the shared validator.
    ///
    /// # Errors
    ///
    /// Propagates every validator failure.
    pub fn validate(&self, options: &ValidationOptions) -> Result<crate::ValidationReport> {
        crate::validate_bundle(&self.path, options)
    }
}

fn check_cancel(cancel: Option<&AtomicBool>) -> Result<()> {
    if let Some(flag) = cancel
        && flag.load(std::sync::atomic::Ordering::Relaxed)
    {
        return Err(FormatError::Cancelled);
    }
    Ok(())
}
