//! Error type shared by the core data contracts.

use thiserror::Error;

/// Crate-wide result alias.
pub type Result<T> = core::result::Result<T, CoreError>;

/// Every failure mode of the core contracts.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum CoreError {
    /// A caller-supplied value failed validation.
    #[error("invalid {field}: {reason}")]
    Invalid {
        /// Name of the rejected field.
        field: &'static str,
        /// Human-readable reason.
        reason: String,
    },
    /// Checked arithmetic overflowed.
    #[error("arithmetic overflow while computing {operation}")]
    Overflow {
        /// Name of the failed operation.
        operation: &'static str,
    },
    /// A digest string was not 64 lowercase/uppercase hexadecimal characters.
    #[error("invalid sha256 digest: {reason}")]
    InvalidDigest {
        /// Why the digest was rejected.
        reason: String,
    },
    /// A configuration document was malformed or violated a hard limit.
    #[error("configuration error: {reason}")]
    Config {
        /// Why the configuration was rejected.
        reason: String,
    },
    /// A storage or numeric type is not supported by this build.
    #[error("unsupported dtype: {dtype}")]
    UnsupportedDType {
        /// The stable name of the unsupported type.
        dtype: String,
    },
    /// A bounded allocation could not be reserved.
    #[error("bounded allocation failed: {reason}")]
    Allocation {
        /// Why the allocation was refused.
        reason: String,
    },
    /// Only returned by test-support helpers; library accounting never does I/O.
    #[error("i/o error: {0}")]
    Io(#[from] std::io::Error),
}

impl CoreError {
    /// Construct an [`CoreError::Invalid`] without repeating the field name.
    #[must_use]
    pub fn invalid(field: &'static str, reason: impl Into<String>) -> Self {
        Self::Invalid {
            field,
            reason: reason.into(),
        }
    }

    /// Construct an [`CoreError::Overflow`].
    #[must_use]
    pub fn overflow(operation: &'static str) -> Self {
        Self::Overflow { operation }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::CoreError;

    #[test]
    fn display_names_field_and_reason() {
        let err = CoreError::invalid("range", "start must be below end");
        assert_eq!(err.to_string(), "invalid range: start must be below end");
    }
}
