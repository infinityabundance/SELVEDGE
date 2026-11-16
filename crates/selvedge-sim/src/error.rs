//! Typed errors for the trace simulator.
//!
//! Every rejection carries the field or limit that failed. There is no
//! "best-effort" path: a malformed trace, an out-of-table shard or an
//! unrepresentable configuration fails closed, because a simulator that guesses
//! produces numbers nobody can check.

/// Failures produced by `selvedge-sim`.
#[derive(Debug, thiserror::Error)]
pub enum SimError {
    /// A field or configuration combination was rejected.
    #[error("invalid {field}: {reason}")]
    Invalid {
        /// Name of the rejected field.
        field: &'static str,
        /// Why it was rejected.
        reason: String,
    },
    /// A bounded input or allocation limit was exceeded.
    #[error("{what} limit exceeded: {value} > {limit}")]
    Limit {
        /// Limited quantity.
        what: &'static str,
        /// Observed value.
        value: u64,
        /// Configured cap.
        limit: u64,
    },
    /// A trace file could not be read or written.
    #[error("trace i/o error: {0}")]
    Io(#[from] std::io::Error),
    /// A trace line or receipt was not valid JSON.
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),
    /// A bundle or manifest failed validation.
    #[error("format error: {0}")]
    Format(#[from] selvedge_format::FormatError),
    /// A core contract was violated.
    #[error("core error: {0}")]
    Core(#[from] selvedge_core::CoreError),
    /// The real store refused a read.
    #[error("store error: {0}")]
    Store(#[from] selvedge_store::StoreError),
}

impl SimError {
    /// Construct an [`SimError::Invalid`].
    #[must_use]
    pub fn invalid(field: &'static str, reason: impl Into<String>) -> Self {
        Self::Invalid {
            field,
            reason: reason.into(),
        }
    }

    /// Construct an [`SimError::Limit`].
    #[must_use]
    pub const fn limit(what: &'static str, value: u64, limit: u64) -> Self {
        Self::Limit { what, value, limit }
    }
}

/// Result alias for the simulator.
pub type Result<T> = std::result::Result<T, SimError>;
