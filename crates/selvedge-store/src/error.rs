//! Typed failures for the storage engine.
//!
//! Every rejection path is an error, never a warning or a silent fallback: a
//! checksum mismatch, a read outside the execution scope, a cache budget that
//! cannot be honoured, or a full transfer queue all fail closed.

use selvedge_core::Sha256Digest;
use selvedge_core::ShardId;
use thiserror::Error;

/// Crate-wide result alias.
pub type Result<T> = core::result::Result<T, StoreError>;

/// Every failure mode of the storage engine and the I/O measurement commands.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum StoreError {
    /// A configuration value or combination was rejected.
    #[error("invalid store configuration: {reason}")]
    Config {
        /// Why the configuration was refused.
        reason: String,
    },
    /// The underlying bundle could not be opened, read or validated.
    #[error("bundle error: {0}")]
    Format(#[from] selvedge_format::FormatError),
    /// A filesystem or system call failed.
    #[error("i/o error: {0}")]
    Io(#[from] std::io::Error),
    /// The shard id is not present in the opened bundle.
    #[error("unknown shard {0:?}")]
    UnknownShard(ShardId),
    /// A required read named a shard outside the execution scope.
    ///
    /// Exact mode executes exactly the planned set; a read of anything else is
    /// a caller bug and is refused rather than silently widening execution.
    #[error("shard {shard:?} is not in the execution scope")]
    NotInScope {
        /// The rejected shard.
        shard: ShardId,
    },
    /// A caller-provided buffer has the wrong length for the shard.
    #[error("buffer for {shard:?} is {actual} bytes, shard is {expected} bytes")]
    BufferSize {
        /// Shard being read.
        shard: ShardId,
        /// Required length.
        expected: u64,
        /// Supplied length.
        actual: u64,
    },
    /// The bytes read did not match the manifest digest.
    #[error("checksum mismatch for {shard:?}: expected {expected}, computed {actual}")]
    Checksum {
        /// Shard whose bytes failed verification.
        shard: ShardId,
        /// Digest recorded in the validated manifest.
        expected: Sha256Digest,
        /// Digest of the bytes actually read.
        actual: Sha256Digest,
    },
    /// The transfer queue is full or over its byte budget.
    #[error("backpressure on {resource}: {reason}")]
    Backpressure {
        /// Resource that refused the work.
        resource: &'static str,
        /// Measured state at refusal time.
        reason: String,
    },
    /// A transfer was cancelled before it completed.
    #[error("transfer {0:?} was cancelled")]
    Cancelled(selvedge_core::TransferId),
    /// A cache insertion would exceed the configured byte budget and no entry
    /// could be evicted to make room.
    #[error("cache budget exceeded: need {requested} bytes, {used} of {capacity} used")]
    CacheBudgetExceeded {
        /// Bytes the insertion needed.
        requested: u64,
        /// Bytes currently resident.
        used: u64,
        /// Configured capacity.
        capacity: u64,
    },
    /// A verification proof was missing, stale or did not describe this bundle.
    #[error("verification proof rejected: {reason}")]
    ProofRejected {
        /// Why the proof could not be used.
        reason: String,
    },
    /// A metrics record could not be serialized or written.
    #[error("metrics error: {reason}")]
    Metrics {
        /// Why the record failed.
        reason: String,
    },
    /// Checked arithmetic overflowed.
    #[error("arithmetic overflow while computing {context}")]
    Overflow {
        /// Name of the overflowing computation.
        context: &'static str,
    },
    /// A measurement precondition was not met (missing file, no free space,
    /// unsupported device counter path).
    #[error("measurement precondition failed: {reason}")]
    Precondition {
        /// What was missing and why the run refused to continue.
        reason: String,
    },
}

impl StoreError {
    /// Build a [`StoreError::Config`].
    pub(crate) fn config(reason: impl Into<String>) -> Self {
        Self::Config {
            reason: reason.into(),
        }
    }

    /// Build a [`StoreError::Precondition`].
    pub(crate) fn precondition(reason: impl Into<String>) -> Self {
        Self::Precondition {
            reason: reason.into(),
        }
    }
}

impl From<rustix::io::Errno> for StoreError {
    fn from(err: rustix::io::Errno) -> Self {
        Self::Io(std::io::Error::from_raw_os_error(err.raw_os_error()))
    }
}

impl From<std::num::TryFromIntError> for StoreError {
    fn from(_: std::num::TryFromIntError) -> Self {
        Self::Overflow {
            context: "integer conversion to usize",
        }
    }
}
