//! Typed errors for the runtime.
//!
//! Every failure is explicit. There is no fallback dtype, no silent
//! re-quantization and no "best effort" partial execution: a shape mismatch, a
//! missing tensor, an over-budget path, a failed witness or a manifest that does
//! not declare a complete SwiGLU partition stops the run.

use selvedge_core::CoreError;

/// Result alias for the runtime.
pub type Result<T> = core::result::Result<T, RuntimeError>;

/// Why a run could not proceed.
#[derive(Debug, thiserror::Error)]
pub enum RuntimeError {
    /// A core contract rejected a value.
    #[error("core: {0}")]
    Core(#[from] CoreError),
    /// The bundle container or manifest was rejected.
    #[error("format: {0}")]
    Format(#[from] selvedge_format::FormatError),
    /// The tiered store refused a read.
    #[error("store: {0}")]
    Store(#[from] selvedge_store::StoreError),
    /// The checkpoint reader or converter refused a tensor.
    #[error("convert: {0}")]
    Convert(#[from] selvedge_convert::ConvertError),
    /// Filesystem failure.
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    /// JSON serialization failure.
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
    /// The tokenizer file could not be loaded or used.
    #[error("tokenizer: {0}")]
    Tokenizer(String),
    /// A configuration value is missing, inconsistent or unsupported.
    #[error("config {field}: {reason}")]
    Config {
        /// Field name.
        field: &'static str,
        /// Why it was rejected.
        reason: String,
    },
    /// The checkpoint and the bundle disagree, or a tensor does not match the
    /// configuration.
    #[error("model: {reason}")]
    Model {
        /// What is inconsistent.
        reason: String,
    },
    /// The bundle does not declare a complete native SwiGLU partition.
    #[error("bundle: {reason}")]
    Bundle {
        /// What the manifest declares instead.
        reason: String,
    },
    /// The budget gate refused the path before execution.
    #[error("budget: {reason}")]
    Budget {
        /// The gate's message.
        reason: String,
    },
    /// A semantic-exactness claim could not be made.
    #[error("exactness: {reason}")]
    Exactness {
        /// Why the witness or classification failed.
        reason: String,
    },
    /// A requested mode is not implemented, or was asked for with arguments it
    /// cannot honour.
    #[error("mode: {reason}")]
    Mode {
        /// Why the request is unsupported.
        reason: String,
    },
    /// A bounded allocation was refused.
    #[error("memory: {reason}")]
    Memory {
        /// Which buffer, and its size.
        reason: String,
    },
}

impl RuntimeError {
    /// Build a configuration error.
    pub(crate) fn config(field: &'static str, reason: impl Into<String>) -> Self {
        Self::Config {
            field,
            reason: reason.into(),
        }
    }

    /// Build a model-consistency error.
    pub(crate) fn model(reason: impl Into<String>) -> Self {
        Self::Model {
            reason: reason.into(),
        }
    }

    /// Build a bundle-declaration error.
    pub(crate) fn bundle(reason: impl Into<String>) -> Self {
        Self::Bundle {
            reason: reason.into(),
        }
    }

    /// Build a budget refusal.
    pub(crate) fn budget(reason: impl Into<String>) -> Self {
        Self::Budget {
            reason: reason.into(),
        }
    }

    /// Build an exactness failure.
    pub(crate) fn exactness(reason: impl Into<String>) -> Self {
        Self::Exactness {
            reason: reason.into(),
        }
    }

    /// Build an unsupported-mode error.
    pub(crate) fn mode(reason: impl Into<String>) -> Self {
        Self::Mode {
            reason: reason.into(),
        }
    }

    /// Build a bounded-allocation refusal.
    pub(crate) fn memory(reason: impl Into<String>) -> Self {
        Self::Memory {
            reason: reason.into(),
        }
    }
}
