//! Typed format failures.

use std::path::PathBuf;

use selvedge_core::{CoreError, Sha256Digest, ShardId};
use thiserror::Error;

/// Format-crate result alias.
pub type Result<T> = core::result::Result<T, FormatError>;

/// Every failure mode of the container format.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum FormatError {
    /// Filesystem or stream failure.
    #[error("i/o error: {0}")]
    Io(#[from] std::io::Error),
    /// A core contract rejected a value.
    #[error("core contract error: {0}")]
    Core(#[from] CoreError),
    /// The bundle exceeds the configured byte cap.
    #[error("bundle is too large: {len} bytes exceeds configured cap {cap}")]
    BundleTooLarge {
        /// Observed length.
        len: u64,
        /// Configured cap.
        cap: u64,
    },
    /// The file is shorter than the 18-byte header.
    #[error("bundle header is truncated: {len} bytes, need 18")]
    HeaderTruncated {
        /// Observed length.
        len: u64,
    },
    /// The first eight bytes are not `SLVDGFMT`.
    #[error("bad bundle magic: {found:?}")]
    MagicMismatch {
        /// Observed magic bytes.
        found: [u8; 8],
    },
    /// The container version is not supported by this build.
    #[error("unsupported container version {found} (this build supports 1)")]
    UnsupportedVersion {
        /// Observed version.
        found: u16,
    },
    /// The manifest length exceeds the configured cap.
    #[error("manifest length {len} exceeds cap {cap}")]
    ManifestTooLarge {
        /// Declared length.
        len: u64,
        /// Configured cap.
        cap: u64,
    },
    /// The declared manifest length does not fit inside the file.
    #[error("manifest length {len} does not fit in a bundle of {file_len} bytes")]
    ManifestOutOfBounds {
        /// Declared length.
        len: u64,
        /// File length.
        file_len: u64,
    },
    /// The manifest bytes are not valid UTF-8.
    #[error("manifest is not valid UTF-8")]
    ManifestNotUtf8,
    /// The manifest JSON is malformed.
    #[error("manifest parse error: {0}")]
    ManifestParse(serde_json::Error),
    /// A manifest invariant failed.
    #[error("manifest is invalid: {reason}")]
    ManifestInvalid {
        /// Why the manifest was rejected.
        reason: String,
    },
    /// A stored shard id does not match the id recomputed from its fields.
    #[error("shard {shard_id}: stored id does not match recomputed id {recomputed}")]
    ShardIdMismatch {
        /// Stored id.
        shard_id: ShardId,
        /// Recomputed id.
        recomputed: ShardId,
    },
    /// Two shard records carry the same id.
    #[error("duplicate shard id {0}")]
    DuplicateShardId(ShardId),
    /// A shard extent is outside the payload.
    #[error("shard {shard_id}: extent {offset}..{end} outside payload of {payload_len} bytes")]
    ExtentOutOfBounds {
        /// Shard id.
        shard_id: ShardId,
        /// Payload-relative offset.
        offset: u64,
        /// Extent length.
        length: u64,
        /// One past the last byte of the offset.
        end: u64,
        /// Payload length.
        payload_len: u64,
    },
    /// Two shard extents overlap.
    #[error("shard extents overlap: {first} and {second}")]
    OverlappingExtents {
        /// First shard.
        first: ShardId,
        /// Second shard.
        second: ShardId,
    },
    /// A payload hash does not match the manifest.
    #[error("shard {shard_id}: sha256 mismatch (expected {expected}, actual {actual})")]
    HashMismatch {
        /// Shard id.
        shard_id: ShardId,
        /// Manifest digest.
        expected: Sha256Digest,
        /// Observed digest.
        actual: Sha256Digest,
    },
    /// The shard byte length is inconsistent with its shape and dtype.
    #[error(
        "shard {shard_id}: manifest length {actual} is not the computed {expected} bytes for {elements} elements of {dtype}"
    )]
    ByteLengthMismatch {
        /// Shard id.
        shard_id: ShardId,
        /// Computed length.
        expected: u64,
        /// Manifest length.
        actual: u64,
        /// Element count.
        elements: u64,
        /// Dtype name.
        dtype: String,
    },
    /// An extent is not aligned as required.
    #[error("shard {shard_id}: offset {offset} is not aligned to {alignment}")]
    AlignmentViolation {
        /// Shard id.
        shard_id: ShardId,
        /// Observed offset.
        offset: u64,
        /// Required alignment.
        alignment: u64,
    },
    /// The manifest alignment is not a power of two.
    #[error("manifest alignment {0} is not a power of two")]
    AlignmentNotPowerOfTwo(u64),
    /// The model digest differs from the expected digest.
    #[error("bundle model digest mismatch: expected {expected}, found {actual}")]
    ModelDigestMismatch {
        /// Expected digest.
        expected: Sha256Digest,
        /// Observed digest.
        actual: Sha256Digest,
    },
    /// A shard compatibility id differs from the expected id.
    #[error("compatibility id mismatch: expected {expected:?}, found {actual:?}")]
    CompatibilityMismatch {
        /// Expected id.
        expected: String,
        /// Observed id.
        actual: String,
    },
    /// A shard uses a dtype this build cannot decode.
    #[error("shard {shard_id}: unsupported dtype {dtype}")]
    UnsupportedDType {
        /// Shard id.
        shard_id: ShardId,
        /// Dtype name.
        dtype: String,
    },
    /// A shard's quantization description is inconsistent.
    #[error("shard {shard_id}: invalid quantization: {reason}")]
    QuantizationInvalid {
        /// Shard id.
        shard_id: ShardId,
        /// Why it was rejected.
        reason: String,
    },
    /// A selected or requested shard does not exist in the manifest.
    #[error("unknown selected shard {0}")]
    UnknownSelectedShard(ShardId),
    /// A requested shard does not exist in the manifest.
    #[error("unknown shard {0}")]
    UnknownShard(ShardId),
    /// The file length does not match the header/manifest layout.
    #[error("file length {actual} does not match layout length {expected}")]
    FileLengthMismatch {
        /// Computed length.
        expected: u64,
        /// Actual length.
        actual: u64,
    },
    /// Validation was cancelled by the caller.
    #[error("validation cancelled")]
    Cancelled,
    /// Refusing to overwrite an immutable bundle.
    #[error("refusing to overwrite existing bundle {0}")]
    ImmutableBundleExists(PathBuf),
    /// Refusing to overwrite an immutable sidecar.
    #[error("refusing to overwrite existing sidecar {0}")]
    ImmutableSidecarExists(PathBuf),
    /// A sidecar does not describe its bundle.
    #[error("sidecar does not match bundle: {reason}")]
    SidecarMismatch {
        /// Why the sidecar was rejected.
        reason: String,
    },
    /// Size arithmetic overflowed.
    #[error("size arithmetic overflow: {context}")]
    SizeOverflow {
        /// Failed operation.
        context: &'static str,
    },
    /// A bounded allocation was refused.
    #[error("bounded allocation of {bytes} bytes refused: {reason}")]
    Allocation {
        /// Requested length.
        bytes: u64,
        /// Why it was refused.
        reason: String,
    },
    /// The payload stream ended early.
    #[error("payload too short: expected {expected} bytes, read {actual}")]
    PayloadShort {
        /// Expected length.
        expected: u64,
        /// Bytes actually read.
        actual: u64,
    },
    /// The payload stream contained more bytes than declared.
    #[error("payload too long: expected {expected} bytes")]
    PayloadLong {
        /// Expected length.
        expected: u64,
    },
    /// A shard record references a shard that is not in the manifest.
    #[error("shard {shard_id}: link to unknown shard {target}")]
    DanglingLink {
        /// Source shard.
        shard_id: ShardId,
        /// Missing target.
        target: ShardId,
    },
    /// A shard record links to itself.
    #[error("shard {shard_id} links to itself")]
    SelfLink {
        /// Offending shard.
        shard_id: ShardId,
    },
}

/// Header length constant exposed in error messages.
pub const HEADER_LEN: u64 = 18;
