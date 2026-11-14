//! Residency and transfer records.
//!
//! These are pure data contracts. No cache policy, thread or device handle is
//! hidden here; later phases implement behaviour against them.

use core::fmt;

use serde::{Deserialize, Serialize};

use crate::error::{CoreError, Result};
use crate::ids::ShardId;

/// One physical storage tier.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StorageTier {
    /// Block device or filesystem-backed cold storage.
    Nvme,
    /// Host DRAM.
    HostRam,
    /// Device memory.
    Gpu,
}

impl fmt::Display for StorageTier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Nvme => f.write_str("nvme"),
            Self::HostRam => f.write_str("host_ram"),
            Self::Gpu => f.write_str("gpu"),
        }
    }
}

/// One resident shard in a snapshot.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResidentShard {
    /// Shard id.
    pub shard: ShardId,
    /// Tier holding the bytes.
    pub tier: StorageTier,
    /// Resident bytes.
    pub bytes: u64,
    /// Whether the shard may not be evicted.
    pub pinned: bool,
    /// Last step at which the shard was used.
    pub last_used_step: Option<u64>,
}

/// A point-in-time residency observation.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResidencySnapshot {
    /// Scheduler step at which the snapshot was taken.
    pub step: u64,
    /// Capacity per tier in bytes.
    pub capacity_bytes: std::collections::BTreeMap<StorageTier, u64>,
    /// Used bytes per tier.
    pub used_bytes: std::collections::BTreeMap<StorageTier, u64>,
    /// Resident shards.
    pub resident: Vec<ResidentShard>,
}

impl ResidencySnapshot {
    /// Validate internal consistency.
    ///
    /// # Errors
    ///
    /// Rejects duplicate resident shards, resident bytes that exceed used bytes
    /// for a tier, and used bytes that exceed capacity.
    pub fn validate(&self) -> Result<()> {
        let mut seen = std::collections::BTreeSet::new();
        let mut resident_bytes: std::collections::BTreeMap<StorageTier, u64> =
            std::collections::BTreeMap::new();
        for shard in &self.resident {
            if !seen.insert(shard.shard) {
                return Err(CoreError::invalid(
                    "residency",
                    format!("shard {} listed twice", shard.shard.short()),
                ));
            }
            let total = resident_bytes.entry(shard.tier).or_insert(0);
            *total = crate::units::checked_add(*total, shard.bytes, "resident bytes")?;
        }
        for (tier, resident) in &resident_bytes {
            let used = self.used_bytes.get(tier).copied().unwrap_or(0);
            if *resident > used {
                return Err(CoreError::invalid(
                    "residency",
                    format!("resident bytes {resident} exceed used bytes {used} on {tier}"),
                ));
            }
        }
        for (tier, used) in &self.used_bytes {
            let capacity = self.capacity_bytes.get(tier).copied().unwrap_or(0);
            if *used > capacity {
                return Err(CoreError::invalid(
                    "residency",
                    format!("used bytes {used} exceed capacity {capacity} on {tier}"),
                ));
            }
        }
        Ok(())
    }
}

/// Monotonic transfer identifier.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct TransferId(pub u64);

/// A requested transfer between tiers.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TransferRequest {
    /// Transfer identifier.
    pub id: TransferId,
    /// Shard being moved.
    pub shard: ShardId,
    /// Source tier.
    pub from: StorageTier,
    /// Destination tier.
    pub to: StorageTier,
    /// Bytes requested.
    pub bytes: u64,
    /// Optional deadline in scheduler steps.
    pub deadline_step: Option<u64>,
    /// Whether the request is speculative.
    pub prefetch: bool,
}

impl TransferRequest {
    /// Validate the request.
    ///
    /// # Errors
    ///
    /// Rejects zero-byte transfers and same-tier transfers.
    pub fn validate(&self) -> Result<()> {
        if self.bytes == 0 {
            return Err(CoreError::invalid("transfer", "byte count is zero"));
        }
        if self.from == self.to {
            return Err(CoreError::invalid(
                "transfer",
                "source and destination tier are the same",
            ));
        }
        Ok(())
    }
}

/// Outcome of a transfer.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum TransferOutcome {
    /// Completed fully.
    Completed,
    /// Aborted by cancellation.
    Cancelled,
    /// Failed with a reason.
    Failed {
        /// Failure description.
        reason: String,
    },
}

/// A completed (or aborted) transfer with measured or estimated timing.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TransferCompletion {
    /// Transfer identifier.
    pub id: TransferId,
    /// Shard moved.
    pub shard: ShardId,
    /// Bytes actually transferred.
    pub bytes_transferred: u64,
    /// Bytes that were consumed by the step.
    pub bytes_useful: u64,
    /// Start timestamp in nanoseconds.
    pub started_ns: u64,
    /// Completion timestamp in nanoseconds.
    pub completed_ns: u64,
    /// Outcome.
    pub outcome: TransferOutcome,
}

impl TransferCompletion {
    /// Bytes moved but not used.
    ///
    /// # Errors
    ///
    /// Rejects a useful byte count above the transferred count.
    pub fn wasted_bytes(&self) -> Result<u64> {
        if self.bytes_useful > self.bytes_transferred {
            return Err(CoreError::invalid(
                "transfer",
                "useful bytes exceed transferred bytes",
            ));
        }
        Ok(self.bytes_transferred - self.bytes_useful)
    }

    /// Elapsed nanoseconds.
    ///
    /// # Errors
    ///
    /// Rejects a completion timestamp before the start timestamp.
    pub fn duration_ns(&self) -> Result<u64> {
        self.completed_ns
            .checked_sub(self.started_ns)
            .ok_or_else(|| CoreError::invalid("transfer", "completion precedes start"))
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::hash::Sha256Digest;

    #[test]
    fn residency_validation() {
        let snapshot = ResidencySnapshot {
            step: 1,
            capacity_bytes: [(StorageTier::Gpu, 100)].into(),
            used_bytes: [(StorageTier::Gpu, 50)].into(),
            resident: vec![ResidentShard {
                shard: ShardId::from_digest(Sha256Digest::of(b"a")),
                tier: StorageTier::Gpu,
                bytes: 50,
                pinned: true,
                last_used_step: Some(1),
            }],
        };
        assert!(snapshot.validate().is_ok());
        let mut broken = snapshot.clone();
        broken.used_bytes.insert(StorageTier::Gpu, 101);
        assert!(broken.validate().is_err());
    }

    #[test]
    fn transfer_math() {
        let completion = TransferCompletion {
            id: TransferId(7),
            shard: ShardId::from_digest(Sha256Digest::of(b"a")),
            bytes_transferred: 100,
            bytes_useful: 60,
            started_ns: 10,
            completed_ns: 40,
            outcome: TransferOutcome::Completed,
        };
        assert_eq!(completion.wasted_bytes().unwrap(), 40);
        assert_eq!(completion.duration_ns().unwrap(), 30);
    }
}
