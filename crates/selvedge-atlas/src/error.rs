//! Typed errors for the Atlas crate. Every public constructor returns a `Result`.

/// Errors produced by the Atlas crate.
#[derive(Debug, thiserror::Error)]
pub enum AtlasError {
    /// A field failed a typed invariant.
    #[error("invalid {field}: {reason}")]
    Invalid {
        /// Field name.
        field: &'static str,
        /// Human-readable reason.
        reason: String,
    },
    /// A graph invariant was violated.
    #[error("graph invariant violated: {0}")]
    Invariant(String),
    /// Filesystem error.
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    /// JSON error.
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),
    /// The plan exceeds the active-parameter ceiling.
    #[error("budget exceeded: {0}")]
    Budget(String),
    /// A model or input has not been fitted or is dimensionally wrong.
    #[error("not fitted: {0}")]
    NotFitted(String),
}

impl AtlasError {
    /// Construct an invalid-field error.
    pub fn invalid(field: &'static str, reason: impl Into<String>) -> Self {
        Self::Invalid {
            field,
            reason: reason.into(),
        }
    }
}

/// Result alias used throughout the crate.
pub type Result<T> = std::result::Result<T, AtlasError>;
