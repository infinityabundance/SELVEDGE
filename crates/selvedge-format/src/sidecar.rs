//! Manifest/index sidecar publication and staleness detection.
//!
//! The sidecar is advisory and redundant with the in-file manifest, but a stale
//! sidecar is a real failure mode: it must not be trusted to describe a bundle
//! it did not come from.

use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use selvedge_core::{Sha256Builder, Sha256Digest};
use serde::{Deserialize, Serialize};

use crate::error::{FormatError, Result};
use crate::manifest::{HEADER_LEN, Manifest};
use crate::validation::read_preamble;

/// Sidecar schema identifier.
pub const SIDECAR_SCHEMA: &str = "selvedge/bundle-sidecar/v1";

/// Maximum sidecar size this build will read.
pub const MAX_SIDECAR_BYTES: u64 = 1024 * 1024;

/// Advisory per-bundle index published beside the bundle.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Sidecar {
    /// Schema identifier.
    pub schema: String,
    /// Bundle id.
    pub bundle_id: String,
    /// Bundle file name (not a full path).
    pub bundle_file: String,
    /// SHA-256 of the complete bundle file.
    pub bundle_sha256: Sha256Digest,
    /// SHA-256 of the exact manifest bytes inside the bundle.
    pub manifest_sha256: Sha256Digest,
    /// Model digest declared by the manifest.
    pub model_digest: Sha256Digest,
    /// Declared payload length.
    pub payload_len: u64,
    /// Declared alignment.
    pub alignment: u64,
    /// Number of shard records.
    pub shard_count: u64,
}

impl Sidecar {
    /// Build a sidecar for a manifest being published.
    ///
    /// # Errors
    ///
    /// Rejects a malformed manifest.
    pub fn for_manifest(
        manifest: &Manifest,
        manifest_sha256: Sha256Digest,
        bundle_sha256: Sha256Digest,
    ) -> Result<Self> {
        manifest.validate()?;
        Ok(Self {
            schema: SIDECAR_SCHEMA.to_owned(),
            bundle_id: manifest.bundle_id.clone(),
            bundle_file: bundle_file_name(&manifest.bundle_id),
            bundle_sha256,
            manifest_sha256,
            model_digest: manifest.model.digest(),
            payload_len: manifest.payload_len,
            alignment: manifest.alignment,
            shard_count: manifest.shards.len() as u64,
        })
    }

    /// Check the sidecar's internal schema.
    ///
    /// # Errors
    ///
    /// Rejects an unknown schema, a mismatched file name or zero/unaligned
    /// lengths.
    pub fn validate(&self) -> Result<()> {
        if self.schema != SIDECAR_SCHEMA {
            return Err(FormatError::SidecarMismatch {
                reason: format!("schema {:?} is not {SIDECAR_SCHEMA:?}", self.schema),
            });
        }
        if self.bundle_file != bundle_file_name(&self.bundle_id) {
            return Err(FormatError::SidecarMismatch {
                reason: "bundle_file does not match bundle_id".to_owned(),
            });
        }
        if !self.alignment.is_power_of_two()
            || self.payload_len == 0
            || !self.payload_len.is_multiple_of(self.alignment)
        {
            return Err(FormatError::SidecarMismatch {
                reason: "alignment or payload length is invalid".to_owned(),
            });
        }
        Ok(())
    }
}

/// Bundle file name for an id.
#[must_use]
pub fn bundle_file_name(bundle_id: &str) -> String {
    format!("{bundle_id}.slv")
}

/// Sidecar file name for an id.
#[must_use]
pub fn sidecar_file_name(bundle_id: &str) -> String {
    format!("{bundle_id}.manifest.json")
}

/// Publish a sidecar atomically in `dir`.
///
/// # Errors
///
/// Rejects an invalid sidecar, an existing sidecar when `force` is false, and
/// filesystem failures. The temporary file is removed on every failure path.
pub fn publish_sidecar(dir: &Path, sidecar: &Sidecar, force: bool) -> Result<PathBuf> {
    sidecar.validate()?;
    std::fs::create_dir_all(dir)?;
    let final_path = dir.join(sidecar_file_name(&sidecar.bundle_id));
    if !force && final_path.exists() {
        return Err(FormatError::ImmutableSidecarExists(final_path));
    }
    let bytes = serde_json::to_vec(sidecar).map_err(FormatError::ManifestParse)?;
    let temp_path = temp_path(dir, "sidecar");
    let mut guard = TempGuard::new(&temp_path);
    {
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temp_path)?;
        use std::io::Write;
        file.write_all(&bytes)?;
        file.sync_all()?;
    }
    publish_file(&temp_path, &final_path, force)?;
    guard.disarm();
    sync_dir(dir)?;
    Ok(final_path)
}

/// Read and validate a sidecar from disk.
///
/// # Errors
///
/// Rejects oversized sidecars, I/O failures, malformed JSON, unknown fields and
/// invalid contents.
pub fn read_sidecar(path: &Path) -> Result<Sidecar> {
    let mut file = std::fs::File::open(path)?;
    let length = file.metadata()?.len();
    if length > MAX_SIDECAR_BYTES {
        return Err(FormatError::ManifestTooLarge {
            len: length,
            cap: MAX_SIDECAR_BYTES,
        });
    }
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(
            usize::try_from(length).map_err(|_| FormatError::Allocation {
                bytes: length,
                reason: "sidecar length does not fit in memory".to_owned(),
            })?,
        )
        .map_err(|err| FormatError::Allocation {
            bytes: length,
            reason: err.to_string(),
        })?;
    file.read_to_end(&mut bytes)?;
    let sidecar: Sidecar = serde_json::from_slice(&bytes).map_err(FormatError::ManifestParse)?;
    sidecar.validate()?;
    Ok(sidecar)
}

/// Check that a sidecar describes the bundle beside it.
///
/// # Errors
///
/// Rejects a different bundle file name, manifest digest, whole-file digest,
/// model digest, payload length, alignment, shard count or bundle id.
pub fn validate_sidecar(bundle_path: &Path, sidecar_path: &Path) -> Result<()> {
    let sidecar = read_sidecar(sidecar_path)?;
    let actual_name = bundle_path
        .file_name()
        .ok_or_else(|| FormatError::SidecarMismatch {
            reason: "bundle path has no file name".to_owned(),
        })?
        .to_string_lossy()
        .into_owned();
    if actual_name != sidecar.bundle_file {
        return Err(FormatError::SidecarMismatch {
            reason: format!(
                "sidecar names {:?} but bundle is {:?}",
                sidecar.bundle_file, actual_name
            ),
        });
    }
    let mut file = std::fs::File::open(bundle_path)?;
    let file_len = file.metadata()?.len();
    let preamble = read_preamble(&mut file, file_len, 8 * 1024 * 1024)?;
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
    if sidecar.bundle_id != preamble.manifest.bundle_id {
        return Err(FormatError::SidecarMismatch {
            reason: "bundle id differs".to_owned(),
        });
    }
    if sidecar.manifest_sha256 != preamble.manifest_sha256 {
        return Err(FormatError::SidecarMismatch {
            reason: "manifest digest differs".to_owned(),
        });
    }
    let bundle_sha256 = hash_file(bundle_path)?;
    if sidecar.bundle_sha256 != bundle_sha256 {
        return Err(FormatError::SidecarMismatch {
            reason: "bundle file digest differs".to_owned(),
        });
    }
    if sidecar.model_digest != preamble.manifest.model.digest()
        || sidecar.payload_len != preamble.manifest.payload_len
        || sidecar.alignment != preamble.manifest.alignment
        || sidecar.shard_count != preamble.manifest.shards.len() as u64
    {
        return Err(FormatError::SidecarMismatch {
            reason: "sidecar counters differ from the manifest".to_owned(),
        });
    }
    Ok(())
}

/// SHA-256 of a whole file, streamed with a bounded buffer.
///
/// # Errors
///
/// Propagates I/O failures.
pub fn hash_file(path: &Path) -> Result<Sha256Digest> {
    let mut file = std::fs::File::open(path)?;
    let mut hasher = Sha256Builder::new();
    let mut buffer = vec![0_u8; 64 * 1024];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(hasher.finish())
}

pub(crate) fn publish_file(temp_path: &Path, final_path: &Path, force: bool) -> Result<()> {
    if force {
        std::fs::rename(temp_path, final_path)?;
        return Ok(());
    }
    match std::fs::hard_link(temp_path, final_path) {
        Ok(()) => {
            std::fs::remove_file(temp_path)?;
            Ok(())
        }
        Err(err) if err.kind() == std::io::ErrorKind::AlreadyExists => {
            Err(FormatError::ImmutableBundleExists(final_path.to_path_buf()))
        }
        Err(err) => Err(FormatError::Io(err)),
    }
}

pub(crate) fn sync_dir(dir: &Path) -> Result<()> {
    // Directory fsync makes the rename/link durable. On platforms where a
    // directory cannot be opened for sync this is reported, not ignored.
    let handle = std::fs::File::open(dir)?;
    handle.sync_all()?;
    Ok(())
}

pub(crate) struct TempGuard {
    path: PathBuf,
    armed: bool,
}

impl TempGuard {
    pub(crate) fn new(path: &Path) -> Self {
        Self {
            path: path.to_path_buf(),
            armed: true,
        }
    }

    pub(crate) fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for TempGuard {
    fn drop(&mut self) {
        if self.armed {
            let _ = std::fs::remove_file(&self.path);
        }
    }
}

static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

pub(crate) fn temp_path(dir: &Path, kind: &str) -> PathBuf {
    let id = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    dir.join(format!(".selvedge-tmp-{kind}-{}-{id}", std::process::id()))
}

/// Expose the header length for callers that build their own diagnostics.
#[must_use]
pub const fn header_len() -> u64 {
    HEADER_LEN
}
