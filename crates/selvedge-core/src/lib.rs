//! Deterministic data contracts for SELVEDGE.
//!
//! This crate owns everything later phases must agree on: content-derived shard
//! identities, the model digest definition, tensor slice descriptions, the
//! learned-parameter ledger and its budget gate, selection provenance, and the
//! typed configuration. Its accounting paths perform no I/O and it contains no
//! unsafe code. The only I/O helpers are in [`testutil`], which is explicitly
//! test support.
//!
//! Invariants callers can rely on:
//!
//! - every identifier is a SHA-256 digest over a documented canonical byte
//!   encoding with explicit little-endian integers; no map iteration order or
//!   platform-dependent layout is ever hashed;
//! - every constructor/parser that accepts untrusted values returns a `Result`
//!   and rejects malformed input instead of panicking;
//! - the budget gate returns `Err(BudgetError::Exceeded)` for an over-budget
//!   plan in compliance mode, so a caller cannot obtain a completed decision by
//!   ignoring a boolean.

#![forbid(unsafe_code)]
#![deny(missing_docs)]

pub mod canonical;
pub mod config;
pub mod courts;
pub mod error;
pub mod hash;
pub mod ids;
pub mod ledger;
pub mod residency;
pub mod selection;
pub mod tensor;
pub mod testutil;
pub mod traits;
pub mod units;
pub mod verdict;

pub use canonical::{canonical_json, canonical_json_sha256, canonical_json_value};
pub use config::{Config, HardLimits};
pub use error::{CoreError, Result};
pub use hash::{Sha256Builder, Sha256Digest};
pub use ids::{CheckpointFile, FORMAT_VERSION, ModelDigest, ShardId, ShardIdentity};
pub use ledger::{
    AdmittedSelectionPath, AuxiliaryKind, Budget, BudgetError, BudgetVerdict, ComplianceMode,
    DecisionPlan, FlopEstimate, ParamUse, ParameterLedger, QuantDequantWork, ResolvedPlan,
    SelectionPath, TokenAccounting, WithinBudgetFallback, union_element_count,
};
pub use residency::{
    ResidencySnapshot, ResidentShard, StorageTier, TransferCompletion, TransferId, TransferOutcome,
    TransferRequest,
};
pub use selection::{
    ExactSelection, NATIVE_SET_DOMAIN, NativeComputationSet, NativeSetWitness, SelectionDecision,
    SelectionMode, SelectionProvenance, SemanticExactness, SemanticGap, ShardScore,
    native_set_digest,
};
pub use tensor::{
    ChannelSet, DType, ElementCount, ParamIdentity, ParamTensorId, QuantScheme, StorageOrder,
    TensorRole, TensorSlice,
};
pub use traits::{Backend, CostEstimate, CostModel};
pub use units::{align_up, bytes_for, element_count};
pub use verdict::{
    ArmVerdict, CausalAvailability, CausalEvidence, CausalFailingReason, DecisionInput,
    PhysicalCompliance, PhysicalEvidence, PhysicalFailingReason, ReadMeasurement, VERDICT_SCHEMA,
};
