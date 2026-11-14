//! Typed failures for safetensors parsing, conversion and reconstruction.

use selvedge_core::CoreError;
use selvedge_format::FormatError;
use thiserror::Error;

/// Crate-wide result alias.
pub type Result<T> = core::result::Result<T, ConvertError>;

/// Every failure mode of tensor reading, conversion and verification.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum ConvertError {
    /// The 8-byte header length is outside the configured or hard cap.
    #[error("safetensors header length {len} exceeds the cap of {cap} bytes")]
    HeaderTooLarge {
        /// Declared header length.
        len: u64,
        /// Configured or hard cap.
        cap: u64,
    },
    /// The declared header does not fit in the file.
    #[error("safetensors header of {len} bytes does not fit in a file of {file_len} bytes")]
    HeaderOutOfBounds {
        /// Declared header length.
        len: u64,
        /// Physical file length.
        file_len: u64,
    },
    /// The header is not aligned to the format's 8-byte requirement.
    #[error("safetensors header length {len} is not a multiple of 8")]
    HeaderMisaligned {
        /// Declared header length.
        len: u64,
    },
    /// The header bytes are not UTF-8.
    #[error("safetensors header is not valid UTF-8")]
    HeaderNotUtf8,
    /// The header is not valid JSON of the required shape.
    #[error("safetensors header is malformed: {reason}")]
    HeaderMalformed {
        /// Parser-provided reason.
        reason: String,
    },
    /// A tensor name appears more than once in the header.
    #[error("safetensors header contains duplicate key {key:?}")]
    DuplicateKey {
        /// Duplicated name.
        key: String,
    },
    /// The header declares more tensors than this build accepts.
    #[error("safetensors header declares {count} tensors, above the cap of {cap}")]
    TooManyTensors {
        /// Declared count.
        count: u64,
        /// Hard cap.
        cap: u64,
    },
    /// A tensor name is empty or unreasonably long.
    #[error("safetensors tensor name is empty or longer than {cap} bytes")]
    InvalidTensorName {
        /// Length cap.
        cap: usize,
    },
    /// A dtype this build does not decode.
    #[error("unsupported safetensors dtype {dtype:?} for tensor {tensor:?}")]
    UnsupportedDType {
        /// Tensor name.
        tensor: String,
        /// Original dtype string.
        dtype: String,
    },
    /// A rank or dimension is outside the accepted bounds.
    #[error("tensor {tensor:?} has an invalid shape: {reason}")]
    InvalidShape {
        /// Tensor name.
        tensor: String,
        /// Why the shape was rejected.
        reason: String,
    },
    /// The declared byte extent disagrees with the shape and dtype.
    #[error(
        "tensor {tensor:?}: data extent {extent} bytes disagrees with {expected} bytes for {elements} elements of {dtype}"
    )]
    ByteLengthMismatch {
        /// Tensor name.
        tensor: String,
        /// Declared extent length.
        extent: u64,
        /// Expected length.
        expected: u64,
        /// Element count.
        elements: u64,
        /// Stable dtype name.
        dtype: String,
    },
    /// The data extent is reversed or outside the data section.
    #[error(
        "tensor {tensor:?}: data offsets {begin}..{end} are outside the {data_len}-byte data section"
    )]
    OffsetOutOfBounds {
        /// Tensor name.
        tensor: String,
        /// First offset.
        begin: u64,
        /// One-past-last offset.
        end: u64,
        /// Data section length.
        data_len: u64,
    },
    /// Two tensor extents overlap.
    #[error("tensors {first:?} and {second:?} have overlapping data extents")]
    OverlappingExtents {
        /// First tensor.
        first: String,
        /// Second tensor.
        second: String,
    },
    /// A positional read ended before the requested bytes.
    #[error("short read at offset {offset}: wanted {wanted} bytes, got {got}")]
    ShortRead {
        /// Absolute offset.
        offset: u64,
        /// Requested byte count.
        wanted: u64,
        /// Actual byte count.
        got: u64,
    },
    /// The requested tensor is not present.
    #[error("tensor {0:?} is not present in the checkpoint")]
    MissingTensor(String),
    /// A tensor is not rank 2 where the FFN path requires it.
    #[error("tensor {tensor:?} has rank {rank}; the FFN path requires rank 2")]
    RankMismatch {
        /// Tensor name.
        tensor: String,
        /// Observed rank.
        rank: usize,
    },
    /// The tensor shapes do not describe a valid SwiGLU FFN.
    #[error("FFN tensor map is inconsistent: {reason}")]
    InconsistentFfn {
        /// Why the map was rejected.
        reason: String,
    },
    /// A source dtype cannot be used for the floating-point FFN path.
    #[error("tensor {tensor:?} has dtype {dtype}; SwiGLU conversion accepts f32, f16 and bf16")]
    SourceDTypeUnsupported {
        /// Tensor name.
        tensor: String,
        /// Stable dtype name.
        dtype: String,
    },
    /// A requested channel or region is outside the tensor.
    #[error("region {rows_start}..{rows_end} x {cols_start}..{cols_end} is outside {tensor:?}")]
    RegionOutOfBounds {
        /// Tensor name.
        tensor: String,
        /// First row.
        rows_start: u64,
        /// One-past-last row.
        rows_end: u64,
        /// First column.
        cols_start: u64,
        /// One-past-last column.
        cols_end: u64,
    },
    /// The grouping specification is invalid.
    #[error("invalid grouping: {reason}")]
    InvalidGrouping {
        /// Why the grouping was rejected.
        reason: String,
    },
    /// A read or output would exceed the in-flight byte budget.
    #[error(
        "memory budget exceeded: requested {requested} bytes with {used} in flight, limit {limit}"
    )]
    MemoryBudgetExceeded {
        /// Bytes requested.
        requested: u64,
        /// Bytes already in flight.
        used: u64,
        /// Configured limit.
        limit: u64,
    },
    /// A bounded allocation could not be reserved.
    #[error("bounded allocation of {bytes} bytes failed: {reason}")]
    Allocation {
        /// Requested bytes.
        bytes: u64,
        /// Allocator-provided reason.
        reason: String,
    },
    /// The conversion plan cannot be assembled.
    #[error("conversion plan is invalid: {reason}")]
    PlanInvalid {
        /// Why the plan was rejected.
        reason: String,
    },
    /// No FFN tensor sets were found.
    #[error("no SwiGLU FFN tensor sets matching gate/up/down suffixes were found")]
    NoFfnTensors,
    /// A bundle read failed.
    #[error("bundle read failed: {0}")]
    Format(#[from] FormatError),
    /// A core contract rejected a value.
    #[error("core contract failed: {0}")]
    Core(#[from] CoreError),
    /// Filesystem or stream I/O failed.
    #[error("i/o error: {0}")]
    Io(#[from] std::io::Error),
    /// A verification tolerance was exceeded.
    #[error(
        "reconstruction tolerance exceeded: max abs {max_abs} > {abs_tol} or relative error {max_rel} > {rel_tol}"
    )]
    ToleranceExceeded {
        /// Measured maximum absolute error.
        max_abs: f64,
        /// Measured maximum relative error.
        max_rel: f64,
        /// Absolute tolerance.
        abs_tol: f64,
        /// Relative tolerance.
        rel_tol: f64,
    },
}

impl ConvertError {
    /// Construct a malformed-header error.
    #[must_use]
    pub fn malformed(reason: impl Into<String>) -> Self {
        Self::HeaderMalformed {
            reason: reason.into(),
        }
    }

    /// Construct a generic inconsistency error.
    #[must_use]
    pub fn inconsistent(reason: impl Into<String>) -> Self {
        Self::InconsistentFfn {
            reason: reason.into(),
        }
    }

    /// Construct a grouping error.
    #[must_use]
    pub fn grouping(reason: impl Into<String>) -> Self {
        Self::InvalidGrouping {
            reason: reason.into(),
        }
    }

    /// Construct a plan error.
    #[must_use]
    pub fn plan(reason: impl Into<String>) -> Self {
        Self::PlanInvalid {
            reason: reason.into(),
        }
    }
}
