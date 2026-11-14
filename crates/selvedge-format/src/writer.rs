//! Atomic bundle publication.
//!
//! The writer never makes a partial file visible: it writes a temporary file in
//! the destination directory, fsyncs it, optionally validates the complete
//! temporary file with the same validator used by readers, and only then
//! hard-links (no-overwrite) or renames it into place. The directory is fsynced
//! after publication.

use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use selvedge_core::{FORMAT_VERSION, Sha256Builder, Sha256Digest, align_up};
use serde::{Deserialize, Serialize};

use crate::error::{FormatError, Result};
use crate::manifest::{
    HARD_MAX_MANIFEST_BYTES, HEADER_LEN, MAGIC, MANIFEST_SCHEMA_VERSION, Manifest,
};
use crate::sidecar::{
    Sidecar, TempGuard, bundle_file_name, publish_file, publish_sidecar, sidecar_file_name,
    sync_dir, temp_path,
};
use crate::validation::{
    BundleHeader, ValidationOptions, ValidationReport, VerifyMode, validate_bundle,
};

/// Publication behavior.
#[derive(Clone, Debug)]
pub struct WriteOptions {
    /// Replace an existing bundle/sidecar with the same id.
    pub force: bool,
    /// Validate the complete temporary file before publishing it.
    pub verify_before_publish: bool,
    /// Publish the advisory manifest/index sidecar.
    pub publish_sidecar: bool,
    /// Manifest size cap.
    pub max_manifest_bytes: u64,
    /// Bundle size cap.
    pub max_bundle_bytes: u64,
}

impl Default for WriteOptions {
    fn default() -> Self {
        Self {
            force: false,
            verify_before_publish: true,
            publish_sidecar: true,
            max_manifest_bytes: 8 * 1024 * 1024,
            max_bundle_bytes: 8 * 1024 * 1024 * 1024,
        }
    }
}

/// What publication produced.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PublishedBundle {
    /// Final bundle path.
    pub path: PathBuf,
    /// Sidecar path, when published.
    pub sidecar: Option<PathBuf>,
    /// SHA-256 of the complete bundle file as written.
    pub bundle_sha256: Sha256Digest,
    /// SHA-256 of the exact manifest bytes.
    pub manifest_sha256: Sha256Digest,
    /// Validation report from the temporary file, when verification was on.
    pub report: Option<ValidationReport>,
}

/// Encode a complete bundle in memory.
///
/// # Errors
///
/// Rejects an invalid manifest, a payload length mismatch and allocation
/// failure.
pub fn encode_bundle_bytes(manifest: &Manifest, payload: &[u8]) -> Result<Vec<u8>> {
    manifest.validate()?;
    let payload_len = payload.len() as u64;
    if payload_len != manifest.payload_len {
        return Err(FormatError::ManifestInvalid {
            reason: format!(
                "payload length {payload_len} does not equal manifest payload_len {}",
                manifest.payload_len
            ),
        });
    }
    let manifest_bytes = manifest.manifest_bytes()?;
    let payload_offset = manifest.payload_offset(manifest_bytes.len() as u64)?;
    let total = payload_offset
        .checked_add(payload_len)
        .ok_or(FormatError::SizeOverflow {
            context: "encoded bundle length",
        })?;
    let total_usize = usize::try_from(total).map_err(|_| FormatError::Allocation {
        bytes: total,
        reason: "bundle length does not fit in memory".to_owned(),
    })?;
    let mut out = Vec::new();
    out.try_reserve_exact(total_usize)
        .map_err(|err| FormatError::Allocation {
            bytes: total,
            reason: err.to_string(),
        })?;
    let header = BundleHeader {
        version: FORMAT_VERSION,
        manifest_len: manifest_bytes.len() as u64,
    };
    out.extend_from_slice(&header.encode());
    out.extend_from_slice(&manifest_bytes);
    let padding = usize::try_from(payload_offset - HEADER_LEN - manifest_bytes.len() as u64)
        .map_err(|_| FormatError::SizeOverflow {
            context: "alignment padding",
        })?;
    out.resize(out.len() + padding, 0);
    out.extend_from_slice(payload);
    Ok(out)
}

/// Publish a bundle from an in-memory payload.
///
/// # Errors
///
/// Propagates [`publish_bundle_from_reader`] failures.
pub fn publish_bundle(
    dir: &Path,
    manifest: &Manifest,
    payload: &[u8],
    options: &WriteOptions,
) -> Result<PublishedBundle> {
    let mut cursor = std::io::Cursor::new(payload);
    publish_bundle_from_reader(dir, manifest, &mut cursor, options)
}

/// Publish a bundle, streaming exactly `manifest.payload_len` bytes from
/// `reader`.
///
/// # Errors
///
/// Rejects an invalid manifest, a payload that is shorter or longer than
/// declared, an existing bundle/sidecar when `force` is false, verification
/// failure, allocation failure and filesystem failures. Every failure before
/// the final link/rename removes the temporary file and leaves an existing
/// bundle untouched.
pub fn publish_bundle_from_reader<R: Read + ?Sized>(
    dir: &Path,
    manifest: &Manifest,
    reader: &mut R,
    options: &WriteOptions,
) -> Result<PublishedBundle> {
    manifest.validate()?;
    let manifest_bytes = manifest.manifest_bytes()?;
    let manifest_cap = options.max_manifest_bytes.min(HARD_MAX_MANIFEST_BYTES);
    if manifest_bytes.len() as u64 > manifest_cap {
        return Err(FormatError::ManifestTooLarge {
            len: manifest_bytes.len() as u64,
            cap: manifest_cap,
        });
    }
    let payload_offset = manifest.payload_offset(manifest_bytes.len() as u64)?;
    let total_len =
        payload_offset
            .checked_add(manifest.payload_len)
            .ok_or(FormatError::SizeOverflow {
                context: "bundle length",
            })?;
    if total_len > options.max_bundle_bytes {
        return Err(FormatError::BundleTooLarge {
            len: total_len,
            cap: options.max_bundle_bytes,
        });
    }

    std::fs::create_dir_all(dir)?;
    let final_path = dir.join(bundle_file_name(&manifest.bundle_id));
    let sidecar_path = dir.join(sidecar_file_name(&manifest.bundle_id));
    if !options.force && final_path.exists() {
        return Err(FormatError::ImmutableBundleExists(final_path));
    }
    if options.publish_sidecar && !options.force && sidecar_path.exists() {
        return Err(FormatError::ImmutableSidecarExists(sidecar_path));
    }

    let temp = temp_path(dir, "bundle");
    let mut guard = TempGuard::new(&temp);
    let mut hashing = HashingWriter::new(
        std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temp)?,
    );

    let header = BundleHeader {
        version: FORMAT_VERSION,
        manifest_len: manifest_bytes.len() as u64,
    };
    hashing.write_all(&header.encode())?;
    hashing.write_all(&manifest_bytes)?;
    write_zeros(
        &mut hashing,
        payload_offset - HEADER_LEN - manifest_bytes.len() as u64,
    )?;
    stream_payload(&mut hashing, reader, manifest.payload_len)?;
    let (file, written, bundle_sha256) = hashing.finish();
    if written != total_len {
        return Err(FormatError::SizeOverflow {
            context: "written bundle length",
        });
    }
    file.sync_all()?;
    drop(file);

    let report = if options.verify_before_publish {
        let validation = ValidationOptions {
            verify: VerifyMode::All,
            max_manifest_bytes: options.max_manifest_bytes,
            max_bundle_bytes: options.max_bundle_bytes,
            ..ValidationOptions::default()
        };
        Some(validate_bundle(&temp, &validation)?)
    } else {
        None
    };

    publish_file(&temp, &final_path, options.force)?;
    guard.disarm();

    let mut published_sidecar = None;
    if options.publish_sidecar {
        let manifest_sha256 = Sha256Digest::of(&manifest_bytes);
        let sidecar = Sidecar::for_manifest(manifest, manifest_sha256, bundle_sha256)?;
        published_sidecar = Some(publish_sidecar(dir, &sidecar, options.force)?);
    }
    sync_dir(dir)?;

    Ok(PublishedBundle {
        path: final_path,
        sidecar: published_sidecar,
        bundle_sha256,
        manifest_sha256: Sha256Digest::of(&manifest_bytes),
        report,
    })
}

fn write_zeros<W: Write>(writer: &mut W, mut count: u64) -> Result<()> {
    let zeros = [0_u8; 8192];
    while count > 0 {
        let chunk = usize::try_from(count.min(zeros.len() as u64)).map_err(|_| {
            FormatError::SizeOverflow {
                context: "padding chunk",
            }
        })?;
        writer.write_all(&zeros[..chunk])?;
        count -= chunk as u64;
    }
    Ok(())
}

fn stream_payload<R: Read + ?Sized, W: Write>(
    writer: &mut W,
    reader: &mut R,
    payload_len: u64,
) -> Result<()> {
    let mut scratch = vec![0_u8; 64 * 1024];
    let mut remaining = payload_len;
    while remaining > 0 {
        let want = usize::try_from(remaining.min(scratch.len() as u64)).map_err(|_| {
            FormatError::SizeOverflow {
                context: "payload chunk",
            }
        })?;
        let read = reader.read(&mut scratch[..want])?;
        if read == 0 {
            return Err(FormatError::PayloadShort {
                expected: payload_len,
                actual: payload_len - remaining,
            });
        }
        writer.write_all(&scratch[..read])?;
        remaining -= read as u64;
    }
    let mut probe = [0_u8; 1];
    if reader.read(&mut probe)? > 0 {
        return Err(FormatError::PayloadLong {
            expected: payload_len,
        });
    }
    Ok(())
}

struct HashingWriter<W: Write> {
    inner: W,
    hasher: Sha256Builder,
    written: u64,
}

impl<W: Write> HashingWriter<W> {
    fn new(inner: W) -> Self {
        Self {
            inner,
            hasher: Sha256Builder::new(),
            written: 0,
        }
    }

    fn finish(self) -> (W, u64, Sha256Digest) {
        let Self {
            inner,
            hasher,
            written,
        } = self;
        (inner, written, hasher.finish())
    }
}

impl<W: Write> Write for HashingWriter<W> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let written = self.inner.write(buf)?;
        self.hasher.update(&buf[..written]);
        self.written += written as u64;
        Ok(written)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.inner.flush()
    }
}

/// The manifest schema version this build writes.
#[must_use]
pub const fn written_schema_version() -> u16 {
    MANIFEST_SCHEMA_VERSION
}

/// The magic bytes this build writes.
#[must_use]
pub const fn written_magic() -> [u8; 8] {
    MAGIC
}

/// Round a length up to the manifest alignment.
///
/// # Errors
///
/// Rejects a zero alignment or overflow.
pub fn round_up(value: u64, alignment: u64) -> Result<u64> {
    Ok(align_up(value, alignment)?)
}
