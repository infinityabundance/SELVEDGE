//! Async-free capability and cost traits.
//!
//! These are deliberately small seams: core has no I/O and no device handles.
//! Later phases implement [`CostModel`] with measured calibrations and
//! [`Backend`] with a real execution backend.

use crate::error::Result;
use crate::ids::ShardId;
use crate::residency::{StorageTier, TransferRequest};
use crate::tensor::DType;

/// A deterministic cost estimate in nanoseconds.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CostEstimate {
    /// Estimated transfer time.
    pub transfer_ns: u64,
    /// Estimated compute time.
    pub compute_ns: u64,
}

impl CostEstimate {
    /// Total estimated time.
    #[must_use]
    pub const fn total_ns(self) -> u64 {
        self.transfer_ns.saturating_add(self.compute_ns)
    }
}

/// Estimates physical costs for scheduling experiments.
///
/// Implementations must state whether a value is measured or modelled. Core
/// never treats an estimate as a benchmark.
pub trait CostModel {
    /// Stable model name.
    fn name(&self) -> &str;

    /// Estimate a transfer request.
    ///
    /// # Errors
    ///
    /// Rejects requests the model cannot represent.
    fn estimate_transfer(&self, request: &TransferRequest) -> Result<CostEstimate>;

    /// Estimate compute for a given FLOP count.
    ///
    /// # Errors
    ///
    /// Rejects impossible counts if the model imposes limits.
    fn estimate_compute(&self, flops: u64) -> Result<CostEstimate>;

    /// Capacity of a tier in bytes.
    fn capacity_bytes(&self, tier: StorageTier) -> u64;
}

/// Capability seam for a future execution backend.
///
/// Execution itself is intentionally absent: this increment implements no
/// tensor kernels. The trait exists so the runtime can be written against a
/// stable contract without core depending on a device library.
pub trait Backend {
    /// Stable backend name, including its version.
    fn name(&self) -> &str;

    /// Whether this build can decode and compute the dtype exactly.
    fn supports_dtype(&self, dtype: &DType) -> bool;

    /// Capacity of a tier in bytes.
    fn capacity_bytes(&self, tier: StorageTier) -> u64;

    /// Whether the backend can read a shard from a tier without an extra copy.
    fn can_read_directly(&self, shard: &ShardId, tier: StorageTier) -> bool;
}
