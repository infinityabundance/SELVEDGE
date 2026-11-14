//! Tiers, residency and the cancellation-safe transfer state machine.
//!
//! A transfer is admitted by [`TransferQueue`], which enforces a bounded pending
//! count and a bounded in-flight byte budget. The ticket returned by `submit`
//! owns the admit; its `Drop` cancels the transfer and releases the budget. That
//! makes cancellation safe by construction: there is no path where a dropped
//! transfer keeps bytes reserved, and no path where a completed transfer is
//! counted twice, because completion consumes the ticket.
//!
//! All timestamps are monotonic nanoseconds since the queue epoch; the wall
//! clock is recorded separately by the metrics layer.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use selvedge_core::{Sha256Digest, ShardId, StorageTier, TransferId};
use serde::{Deserialize, Serialize};

use crate::error::{Result, StoreError};

/// The three tiers the engine models.
///
/// [`Tier::GpuStaging`] is a declared destination with no device implementation
/// in this build: transfers to it are recorded and counted but the copy is not
/// performed. No result in this repository claims GPU-tier measurement.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Tier {
    /// Bundle bytes on the block device.
    Nvme,
    /// Host DRAM: the cache and the caller's buffers.
    HostRam,
    /// Device-memory staging, declared but not implemented in this build.
    GpuStaging,
}

impl Tier {
    /// Whether this build can actually move bytes into the tier.
    #[must_use]
    pub const fn implemented(self) -> bool {
        matches!(self, Self::Nvme | Self::HostRam)
    }

    /// Map to the core residency vocabulary.
    #[must_use]
    pub const fn storage_tier(self) -> StorageTier {
        match self {
            Self::Nvme => StorageTier::Nvme,
            Self::HostRam => StorageTier::HostRam,
            Self::GpuStaging => StorageTier::Gpu,
        }
    }
}

/// Residency of one shard at a point in time.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Residency {
    /// Shard.
    pub shard: ShardId,
    /// Tier holding the authoritative copy for this step.
    pub tier: Tier,
    /// Resident byte count.
    pub bytes: u64,
    /// Pinned entries are never chosen as eviction victims.
    pub pinned: bool,
    /// Monotonic sequence number of the last access, when one was observed.
    pub last_used_seq: Option<u64>,
}

/// State of a transfer in the machine.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TransferState {
    /// Admitted, waiting for its turn.
    Queued,
    /// Owned by a worker that is moving the bytes.
    InFlight,
    /// Bytes arrived and were accounted.
    Completed,
    /// Abandoned by the owner; the reservation was released.
    Cancelled,
    /// Failed with a recorded reason; the reservation was released.
    Failed,
}

impl TransferState {
    /// Whether the transfer still holds its admitted reservation.
    #[must_use]
    pub const fn holds_reservation(self) -> bool {
        matches!(self, Self::Queued | Self::InFlight)
    }

    /// Whether the transfer is terminal.
    #[must_use]
    pub const fn is_terminal(self) -> bool {
        matches!(self, Self::Completed | Self::Cancelled | Self::Failed)
    }
}

/// One admitted transfer and its observed timing.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Transfer {
    /// Monotonic id.
    pub id: TransferId,
    /// Shard moved.
    pub shard: ShardId,
    /// Source tier.
    pub from: Tier,
    /// Destination tier.
    pub to: Tier,
    /// Bytes reserved at admission.
    pub bytes: u64,
    /// Deadline in scheduler steps, when the caller supplied one.
    pub deadline_step: Option<u64>,
    /// Whether the transfer is speculative.
    pub prefetch: bool,
    /// Current state.
    pub state: TransferState,
    /// Monotonic admission timestamp.
    pub queued_ns: u64,
    /// Monotonic start timestamp.
    pub started_ns: Option<u64>,
    /// Monotonic finish timestamp.
    pub finished_ns: Option<u64>,
    /// Failure or cancellation detail.
    pub detail: Option<String>,
}

impl Transfer {
    /// Queue wait in nanoseconds, if the transfer started.
    #[must_use]
    pub fn queue_wait_ns(&self) -> Option<u64> {
        self.started_ns
            .map(|started| started.saturating_sub(self.queued_ns))
    }

    /// Transfer duration in nanoseconds, if it finished.
    #[must_use]
    pub fn transfer_ns(&self) -> Option<u64> {
        match (self.started_ns, self.finished_ns) {
            (Some(started), Some(finished)) => Some(finished.saturating_sub(started)),
            _ => None,
        }
    }
}

/// Hard limits on the transfer queue.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QueueLimits {
    /// Maximum queued plus in-flight transfers.
    pub max_pending: usize,
    /// Maximum in-flight transfers.
    pub max_in_flight: usize,
    /// Maximum in-flight bytes.
    pub max_in_flight_bytes: u64,
}

impl QueueLimits {
    /// Validate the limits.
    ///
    /// # Errors
    ///
    /// Rejects a zero bound in any field.
    pub fn validate(&self) -> Result<()> {
        if self.max_pending == 0 || self.max_in_flight == 0 || self.max_in_flight_bytes == 0 {
            return Err(StoreError::config(
                "queue limits must all be non-zero; there is no unbounded mode",
            ));
        }
        if self.max_in_flight > self.max_pending {
            return Err(StoreError::config(
                "max_in_flight must not exceed max_pending",
            ));
        }
        Ok(())
    }
}

/// Queue counters, including every rejection.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QueueStats {
    /// Transfers admitted to the queue.
    pub admitted: u64,
    /// Transfers that reached the in-flight state.
    pub started: u64,
    /// Transfers that completed.
    pub completed: u64,
    /// Transfers cancelled by their owner.
    pub cancelled: u64,
    /// Transfers that failed.
    pub failed: u64,
    /// Admissions refused because the pending count was full.
    pub rejected_pending: u64,
    /// Admissions refused because the in-flight byte budget was exceeded.
    pub rejected_bytes: u64,
    /// Admissions refused because the in-flight transfer count was exceeded.
    pub rejected_in_flight: u64,
}

#[derive(Debug, Default)]
struct QueueState {
    next_id: u64,
    pending: BTreeMap<TransferId, Transfer>,
    in_flight: BTreeMap<TransferId, Transfer>,
    in_flight_bytes: u64,
    finished: BTreeMap<TransferId, Transfer>,
    stats: QueueStats,
}

/// Bounded, cancellation-safe transfer queue.
#[derive(Clone, Debug)]
pub struct TransferQueue {
    inner: Arc<Mutex<QueueState>>,
    limits: QueueLimits,
    epoch: Instant,
    /// Counted separately so a refusal is observable without touching the lock.
    refusals: Arc<AtomicU64>,
}

impl TransferQueue {
    /// Create a queue with validated limits.
    ///
    /// # Errors
    ///
    /// Rejects invalid limits.
    pub fn new(limits: QueueLimits) -> Result<Self> {
        limits.validate()?;
        Ok(Self {
            inner: Arc::new(Mutex::new(QueueState::default())),
            limits,
            epoch: Instant::now(),
            refusals: Arc::new(AtomicU64::new(0)),
        })
    }

    /// Configured limits.
    #[must_use]
    pub const fn limits(&self) -> QueueLimits {
        self.limits
    }

    /// Monotonic nanoseconds since this queue was created.
    #[must_use]
    pub fn now_ns(&self) -> u64 {
        elapsed_ns(self.epoch)
    }

    /// Admit a transfer.
    ///
    /// Required work is never refused for byte pressure: only speculative
    /// (prefetch) transfers can be rejected by the in-flight byte budget. Both
    /// kinds are refused when the pending count is full.
    ///
    /// # Errors
    ///
    /// Rejects zero-byte and same-tier transfers, a full queue, and an
    /// over-budget speculative transfer.
    pub fn submit(
        &self,
        shard: ShardId,
        bytes: u64,
        from: Tier,
        to: Tier,
        prefetch: bool,
        deadline_step: Option<u64>,
    ) -> Result<TransferTicket> {
        if bytes == 0 {
            return Err(StoreError::config("transfer byte count is zero"));
        }
        if from == to {
            return Err(StoreError::config(
                "transfer source and destination tier are the same",
            ));
        }
        if from == Tier::GpuStaging || to == Tier::GpuStaging {
            // Declared but not implemented: refuse instead of pretending to move
            // bytes. A later phase replaces this with a real device transfer.
            return Err(StoreError::config(
                "GPU staging transfers are declared but not implemented in this build",
            ));
        }
        let mut state = self.lock();
        let queued = state.pending.len() + state.in_flight.len();
        if queued >= self.limits.max_pending {
            state.stats.rejected_pending += 1;
            self.refusals.fetch_add(1, Ordering::Relaxed);
            return Err(StoreError::Backpressure {
                resource: "transfer_queue",
                reason: format!(
                    "{queued} transfers already queued or in flight (limit {})",
                    self.limits.max_pending
                ),
            });
        }
        if state.in_flight.len() >= self.limits.max_in_flight {
            state.stats.rejected_in_flight += 1;
            self.refusals.fetch_add(1, Ordering::Relaxed);
            return Err(StoreError::Backpressure {
                resource: "in_flight_transfers",
                reason: format!(
                    "{} transfers in flight (limit {})",
                    state.in_flight.len(),
                    self.limits.max_in_flight
                ),
            });
        }
        if prefetch {
            let requested = state.in_flight_bytes.saturating_add(bytes);
            if requested > self.limits.max_in_flight_bytes {
                state.stats.rejected_bytes += 1;
                self.refusals.fetch_add(1, Ordering::Relaxed);
                return Err(StoreError::Backpressure {
                    resource: "in_flight_bytes",
                    reason: format!(
                        "{requested} bytes would be in flight (limit {})",
                        self.limits.max_in_flight_bytes
                    ),
                });
            }
        }
        let id = TransferId(state.next_id);
        state.next_id += 1;
        let transfer = Transfer {
            id,
            shard,
            from,
            to,
            bytes,
            deadline_step,
            prefetch,
            state: TransferState::Queued,
            queued_ns: elapsed_ns(self.epoch),
            started_ns: None,
            finished_ns: None,
            detail: None,
        };
        state.pending.insert(id, transfer);
        state.stats.admitted += 1;
        Ok(TransferTicket {
            inner: Arc::clone(&self.inner),
            id,
            epoch: self.epoch,
            bytes,
            prefetch,
        })
    }

    /// Point-in-time copy of the queue contents and counters.
    #[must_use]
    pub fn snapshot(&self) -> QueueSnapshot {
        let state = self.lock();
        QueueSnapshot {
            pending: state.pending.values().cloned().collect(),
            in_flight: state.in_flight.values().cloned().collect(),
            in_flight_bytes: state.in_flight_bytes,
            stats: state.stats,
            refusals: self.refusals.load(Ordering::Relaxed),
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, QueueState> {
        match self.inner.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        }
    }
}

/// Queue contents and counters.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QueueSnapshot {
    /// Queued transfers.
    pub pending: Vec<Transfer>,
    /// In-flight transfers.
    pub in_flight: Vec<Transfer>,
    /// In-flight bytes.
    pub in_flight_bytes: u64,
    /// Counters.
    pub stats: QueueStats,
    /// Total refusals observed by the atomic counter.
    pub refusals: u64,
}

/// Ownership token for an admitted transfer.
///
/// Dropping a queued ticket cancels it; `start` transitions to in-flight and
/// returns an [`InFlightTransfer`] whose `Drop` also cancels. Completion must
/// consume the in-flight owner, so no transfer can be both completed and
/// cancelled.
#[derive(Debug)]
pub struct TransferTicket {
    inner: Arc<Mutex<QueueState>>,
    id: TransferId,
    epoch: Instant,
    bytes: u64,
    prefetch: bool,
}

impl TransferTicket {
    /// Id of the admitted transfer.
    #[must_use]
    pub const fn id(&self) -> TransferId {
        self.id
    }

    /// Reserved bytes.
    #[must_use]
    pub const fn bytes(&self) -> u64 {
        self.bytes
    }

    /// Whether the transfer is speculative.
    #[must_use]
    pub const fn is_prefetch(&self) -> bool {
        self.prefetch
    }

    /// Transition to in-flight and hand ownership to the worker.
    #[must_use]
    pub fn start(self) -> InFlightTransfer {
        let mut state = lock_shared(&self.inner);
        state.in_flight_bytes = state.in_flight_bytes.saturating_add(self.bytes);
        if let Some(mut transfer) = state.pending.remove(&self.id) {
            transfer.state = TransferState::InFlight;
            transfer.started_ns = Some(elapsed_ns(self.epoch));
            state.stats.started += 1;
            state.in_flight.insert(self.id, transfer);
        }
        drop(state);
        InFlightTransfer {
            inner: Arc::clone(&self.inner),
            id: self.id,
            epoch: self.epoch,
            bytes: self.bytes,
        }
    }
}

impl Drop for TransferTicket {
    fn drop(&mut self) {
        let mut state = lock_shared(&self.inner);
        if let Some(mut transfer) = state.pending.remove(&self.id) {
            transfer.state = TransferState::Cancelled;
            transfer.finished_ns = Some(elapsed_ns(self.epoch));
            transfer.detail = Some("ticket dropped before start".to_owned());
            state.stats.cancelled += 1;
            state.finished.insert(self.id, transfer);
        }
    }
}

/// In-flight owner of a transfer.
#[derive(Debug)]
pub struct InFlightTransfer {
    inner: Arc<Mutex<QueueState>>,
    id: TransferId,
    epoch: Instant,
    bytes: u64,
}

impl InFlightTransfer {
    /// Complete the transfer and release the byte reservation.
    #[must_use]
    pub fn complete(self) -> Transfer {
        let mut state = lock_shared(&self.inner);
        state.in_flight_bytes = state.in_flight_bytes.saturating_sub(self.bytes);
        let finished_ns = Some(elapsed_ns(self.epoch));
        let removed = state.in_flight.remove(&self.id);
        match removed {
            Some(mut transfer) => {
                transfer.state = TransferState::Completed;
                transfer.finished_ns = finished_ns;
                state.stats.completed += 1;
                let copy = transfer.clone();
                state.finished.insert(self.id, transfer);
                copy
            }
            None => Transfer {
                id: self.id,
                shard: ShardId::from_digest(Sha256Digest::of(b"missing")),
                from: Tier::Nvme,
                to: Tier::HostRam,
                bytes: 0,
                deadline_step: None,
                prefetch: false,
                state: TransferState::Failed,
                queued_ns: 0,
                started_ns: None,
                finished_ns,
                detail: Some("completed an unknown transfer".to_owned()),
            },
        }
    }

    /// Fail the transfer with a reason and release the byte reservation.
    #[must_use]
    pub fn fail(self, reason: &str) -> Transfer {
        let mut state = lock_shared(&self.inner);
        state.in_flight_bytes = state.in_flight_bytes.saturating_sub(self.bytes);
        let finished_ns = Some(elapsed_ns(self.epoch));
        let removed = state.in_flight.remove(&self.id);
        match removed {
            Some(mut transfer) => {
                transfer.state = TransferState::Failed;
                transfer.finished_ns = finished_ns;
                transfer.detail = Some(reason.to_owned());
                state.stats.failed += 1;
                let copy = transfer.clone();
                state.finished.insert(self.id, transfer);
                copy
            }
            None => Transfer {
                id: self.id,
                shard: ShardId::from_digest(Sha256Digest::of(b"missing")),
                from: Tier::Nvme,
                to: Tier::HostRam,
                bytes: 0,
                deadline_step: None,
                prefetch: false,
                state: TransferState::Failed,
                queued_ns: 0,
                started_ns: None,
                finished_ns,
                detail: Some(reason.to_owned()),
            },
        }
    }
}

impl Drop for InFlightTransfer {
    fn drop(&mut self) {
        let mut state = lock_shared(&self.inner);
        state.in_flight_bytes = state.in_flight_bytes.saturating_sub(self.bytes);
        if let Some(mut transfer) = state.in_flight.remove(&self.id) {
            transfer.state = TransferState::Cancelled;
            transfer.finished_ns = Some(elapsed_ns(self.epoch));
            transfer.detail = Some("in-flight owner dropped without completion".to_owned());
            state.stats.cancelled += 1;
            state.finished.insert(self.id, transfer);
        }
    }
}

fn lock_shared(inner: &Arc<Mutex<QueueState>>) -> std::sync::MutexGuard<'_, QueueState> {
    match inner.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    }
}

/// Monotonic nanoseconds since `epoch`, saturating instead of panicking.
#[must_use]
pub fn elapsed_ns(epoch: Instant) -> u64 {
    u64::try_from(epoch.elapsed().as_nanos()).unwrap_or(u64::MAX)
}
