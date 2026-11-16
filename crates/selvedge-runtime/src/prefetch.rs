//! Asynchronous prefetch: separate fetch and compute timelines.
//!
//! The synchronous prefetch in [`crate::bundle::GroupReader::prefetch`] reads
//! every shard of a layer through the store **before** the compute loop starts.
//! It exercises ordering and the cache, but it cannot hide storage latency,
//! because nothing runs while it blocks.
//!
//! This module runs the reads on a second timeline:
//!
//! ```text
//! fetch   |--s1--|--s2--|----s3----|--s4--|
//! compute       |matvec s1|matvec s2|  wait  |matvec s3|matvec s4|
//!                                ^exposed stall
//! ```
//!
//! * **Readiness dependency.** Compute asks for a specific shard id; it blocks
//!   only until that shard's staged bytes exist. It never waits for the whole
//!   layer, so a late shard at the end of the order costs nothing if compute is
//!   still working on the first ones.
//! * **Bounded staging.** One worker thread with one reusable buffer of at most
//!   `max_shard_bytes`, plus a bounded job queue of depth
//!   [`crate::config::StreamConfig::prefetch_depth`]. Submission is
//!   non-blocking: a full queue leaves jobs in a local pending list, so the
//!   compute thread never blocks on the queue itself.
//! * **Cancellation.** When the layer finishes, the fetcher is cancelled: the
//!   worker stops before its next shard, queued jobs are dropped, and everything
//!   fetched but not consumed is counted as wasted prefetch bytes.
//! * **Accounting.** Every measured nanosecond of the fetch timeline is either
//!   *hidden* (it elapsed while compute was not blocked) or *exposed* (compute
//!   waited for it). Every fetched byte is either *used* or *wasted*. Both are
//!   identities, not estimates; see [`StallAccounting::conserves`].
//!
//! The prefetcher never chooses what to execute: the shard list is exactly the
//! set the caller was already going to read, in the caller's order. It can only
//! change **when** bytes arrive.
//!
//! # What CPU measurements can and cannot show
//!
//! On this host the bundle is on a local NVMe device and the store serves warm
//! shards from RAM, so the real transfer time is small compared with a layer's
//! arithmetic. The overlap this module creates is therefore small on an
//! uninstrumented warm run, and that is stated in `docs/RUNTIME.md` rather than
//! hidden. With the injected delay (`--inject-delay-ns`) it is measurable and
//! attributable: the delay lands on the fetch timeline, and the receipt shows how
//! much of it compute hid and how much it waited for. No GPU overlap is claimed:
//! a CUDA backend would have to prove the same identities on the device.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::sync::mpsc::{Receiver, SyncSender, TrySendError, channel, sync_channel};
use std::sync::{Arc, atomic::AtomicBool};
use std::time::{Duration, Instant};

use selvedge_core::ShardId;
use selvedge_store::{ExecutionScope, ShardStore};
use serde::Serialize;

use crate::bundle::ShardFragment;
use crate::error::{Result, RuntimeError};

/// How one fetcher's timeline splits between hidden and exposed work.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
pub struct StallAccounting {
    /// Nanoseconds the fetch thread spent inside store reads.
    pub fetch_busy_ns: u64,
    /// Nanoseconds compute spent blocked waiting for a shard it needed now.
    pub exposed_stall_ns: u64,
    /// Fetch time that elapsed while compute was not blocked on it.
    pub hidden_ns: u64,
    /// Blocked time during which the fetch thread was not busy (queueing,
    /// scheduling, or a wake-up that arrived late).
    pub residual_wait_ns: u64,
    /// Waits that had to block at all.
    pub waits: u64,
    /// Shard dependencies that were already satisfied when compute asked.
    pub ready_without_wait: u64,
}

impl StallAccounting {
    /// The conservation identity this module guarantees.
    ///
    /// `fetch_busy + residual == hidden + exposed`: every nanosecond of the
    /// fetch timeline is either hidden behind compute, exposed as a stall, or
    /// accounted as residual wait outside the fetch itself. No term is dropped
    /// and no term is double counted.
    #[must_use]
    pub const fn conserves(&self) -> bool {
        self.fetch_busy_ns.saturating_add(self.residual_wait_ns)
            == self.hidden_ns.saturating_add(self.exposed_stall_ns)
    }
}

/// What the fetcher did for one layer.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
pub struct PrefetchAccounting {
    /// Jobs submitted to the worker.
    pub jobs: u64,
    /// Shards submitted.
    pub shards_submitted: u64,
    /// Shards whose bytes were delivered.
    pub shards_completed: u64,
    /// Shards dropped because the fetcher was cancelled first.
    pub shards_cancelled: u64,
    /// Bytes delivered by the fetcher (cache hits included).
    pub fetched_bytes: u64,
    /// Bytes of those that came from the backing tier.
    pub device_bytes: u64,
    /// Fetched bytes compute actually consumed.
    pub used_bytes: u64,
    /// Fetched bytes compute never consumed (cancelled or superseded).
    pub wasted_bytes: u64,
    /// Largest staging buffer the worker used.
    pub staging_high_water_bytes: u64,
    /// Configured queue depth.
    pub depth: u64,
    /// Hidden versus exposed time.
    pub stall: StallAccounting,
}

impl PrefetchAccounting {
    /// The byte identity this module guarantees: nothing fetched is unaccounted.
    #[must_use]
    pub const fn conserves_bytes(&self) -> bool {
        self.fetched_bytes == self.used_bytes.saturating_add(self.wasted_bytes)
    }
}

/// What one async prefetch pass produced.
#[derive(Clone, Debug, Default)]
pub struct PrefetchOutcome {
    /// Counters and timeline split.
    pub accounting: PrefetchAccounting,
    /// Every individual wait that blocked, in nanoseconds, for percentiles.
    pub stalls: Vec<u64>,
    /// For every shard that was staged, whether its bytes came from the device.
    ///
    /// The caller attributes tier bytes from this map exactly once, so a
    /// prefetched shard is never counted twice (once here, once by the execution
    /// read that follows it).
    pub sources: BTreeMap<ShardId, bool>,
}

enum Job {
    Fetch(ShardFragment),
}

struct Done {
    shard: ShardId,
    bytes: u64,
    device: bool,
    fetch_ns: u64,
    staging_bytes: u64,
    failure: Option<String>,
}

struct WorkerResult {
    cancelled: u64,
    staging_high_water_bytes: u64,
    failure: Option<String>,
}

/// A running fetch timeline.
///
/// Created by [`AsyncFetcher::start`], used through [`Self::wait_ready`] and
/// consumed by [`Self::finish`]. Dropping it cancels the worker, so cancellation
/// is the default, not an optional extra.
pub struct AsyncFetcher<'scope> {
    tx: Option<SyncSender<Job>>,
    rx: Receiver<Done>,
    handle: Option<std::thread::ScopedJoinHandle<'scope, WorkerResult>>,
    cancel: Arc<AtomicBool>,
    pending: VecDeque<ShardFragment>,
    reported: BTreeMap<ShardId, Done>,
    used: BTreeSet<ShardId>,
    sources: BTreeMap<ShardId, bool>,
    accounting: PrefetchAccounting,
    stalls: Vec<u64>,
    failure: Option<String>,
}

impl std::fmt::Debug for AsyncFetcher<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AsyncFetcher")
            .field("pending", &self.pending.len())
            .field("reported", &self.reported.len())
            .field("used", &self.used.len())
            .field("failure", &self.failure)
            .finish_non_exhaustive()
    }
}

impl<'scope> AsyncFetcher<'scope> {
    /// Start a fetch timeline for `fragments`, in the given order.
    ///
    /// # Errors
    ///
    /// Rejects a zero depth and a zero per-shard buffer bound.
    pub fn start(
        scope: &'scope std::thread::Scope<'scope, '_>,
        store: &'scope ShardStore,
        execution_scope: ExecutionScope,
        fragments: &[ShardFragment],
        depth: usize,
        max_shard_bytes: u64,
        injected_delay_ns: u64,
    ) -> Result<Self> {
        if depth == 0 {
            return Err(RuntimeError::config(
                "prefetch_depth",
                "the asynchronous fetcher needs at least one outstanding job",
            ));
        }
        if max_shard_bytes == 0 {
            return Err(RuntimeError::config(
                "max_shard_bytes",
                "the staging buffer needs a non-zero bound",
            ));
        }
        let (tx, rx_job) = sync_channel::<Job>(depth);
        let (tx_done, rx_done) = channel::<Done>();
        let cancel = Arc::new(AtomicBool::new(false));
        let cancel_worker = Arc::clone(&cancel);
        let largest = fragments
            .iter()
            .map(|fragment| fragment.bytes)
            .max()
            .unwrap_or(0);
        let staging_bound = largest.min(max_shard_bytes);
        let handle = scope.spawn(move || {
            worker(
                store,
                &execution_scope,
                rx_job,
                tx_done,
                &cancel_worker,
                staging_bound,
                injected_delay_ns,
            )
        });
        let mut accounting = PrefetchAccounting {
            depth: u64::try_from(depth).unwrap_or(u64::MAX),
            staging_high_water_bytes: staging_bound,
            ..PrefetchAccounting::default()
        };
        accounting.shards_submitted = u64::try_from(fragments.len()).unwrap_or(u64::MAX);
        accounting.jobs = 1;
        Ok(Self {
            tx: Some(tx),
            rx: rx_done,
            handle: Some(handle),
            cancel,
            pending: fragments.iter().copied().collect(),
            reported: BTreeMap::new(),
            used: BTreeSet::new(),
            sources: BTreeMap::new(),
            accounting,
            stalls: Vec::new(),
            failure: None,
        })
    }

    /// Submit as many queued jobs as the bounded queue accepts, without blocking.
    fn pump(&mut self) {
        let Some(tx) = self.tx.as_ref() else {
            return;
        };
        while let Some(fragment) = self.pending.front().copied() {
            match tx.try_send(Job::Fetch(fragment)) {
                Ok(()) => {
                    self.pending.pop_front();
                }
                Err(TrySendError::Full(_)) => break,
                Err(TrySendError::Disconnected(_)) => {
                    self.pending.clear();
                    break;
                }
            }
        }
    }

    /// Block until `shard`'s bytes are staged, then mark it consumed.
    ///
    /// # Errors
    ///
    /// Propagates a fetch failure: a prefetch this runtime performs is a read it
    /// accounted for, so a failure is a failed layer, not a silent fallback.
    pub fn wait_ready(&mut self, shard: &ShardId) -> Result<()> {
        self.pump();
        if let Some(failure) = &self.failure {
            return Err(RuntimeError::model(failure.clone()));
        }
        if !self.reported.contains_key(shard) {
            let started = Instant::now();
            loop {
                match self.rx.recv() {
                    Ok(done) => {
                        let failed = done.failure.clone();
                        let id = done.shard;
                        self.account(done);
                        if id == *shard {
                            break;
                        }
                        if failed.is_some() {
                            break;
                        }
                    }
                    Err(_) => {
                        let message = format!(
                            "the async fetcher stopped before staging shard {}",
                            shard.short()
                        );
                        self.failure = Some(message.clone());
                        return Err(RuntimeError::model(message));
                    }
                }
            }
            let stalled = u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX);
            self.accounting.stall.exposed_stall_ns = self
                .accounting
                .stall
                .exposed_stall_ns
                .saturating_add(stalled);
            self.accounting.stall.waits = self.accounting.stall.waits.saturating_add(1);
            self.stalls.push(stalled);
        } else {
            self.accounting.stall.ready_without_wait =
                self.accounting.stall.ready_without_wait.saturating_add(1);
        }
        if let Some(failure) = &self.failure {
            return Err(RuntimeError::model(failure.clone()));
        }
        if let Some(done) = self.reported.remove(shard) {
            if let Some(failure) = done.failure {
                self.failure = Some(failure.clone());
                return Err(RuntimeError::model(failure));
            }
            self.used.insert(*shard);
            self.sources.insert(*shard, done.device);
            self.accounting.used_bytes = self.accounting.used_bytes.saturating_add(done.bytes);
        }
        Ok(())
    }

    /// Whether a consumed shard's bytes came from the backing tier.
    ///
    /// `None` until the shard has been consumed, so a caller cannot attribute a
    /// tier for a shard it did not take from the fetcher.
    #[must_use]
    pub fn consumed_device(&self, shard: &ShardId) -> Option<bool> {
        self.sources.get(shard).copied()
    }

    fn account(&mut self, done: Done) {
        self.accounting.shards_completed = self.accounting.shards_completed.saturating_add(1);
        self.accounting.fetched_bytes = self.accounting.fetched_bytes.saturating_add(done.bytes);
        if done.device {
            self.accounting.device_bytes = self.accounting.device_bytes.saturating_add(done.bytes);
        }
        self.accounting.stall.fetch_busy_ns = self
            .accounting
            .stall
            .fetch_busy_ns
            .saturating_add(done.fetch_ns);
        self.accounting.staging_high_water_bytes = self
            .accounting
            .staging_high_water_bytes
            .max(done.staging_bytes);
        if self.failure.is_none()
            && let Some(failure) = &done.failure
        {
            self.failure = Some(failure.clone());
        }
        self.reported.insert(done.shard, done);
    }

    /// Cancel the fetcher and finish the accounting.
    ///
    /// # Errors
    ///
    /// Propagates a fetch failure observed by the worker.
    pub fn finish(mut self) -> Result<PrefetchOutcome> {
        self.cancel.store(true, std::sync::atomic::Ordering::SeqCst);
        self.tx = None;
        let unsent = u64::try_from(self.pending.len()).unwrap_or(u64::MAX);
        self.pending.clear();
        // Drain anything the worker delivered before it noticed the cancellation.
        while let Ok(done) = self.rx.try_recv() {
            self.account(done);
        }
        let mut worker = WorkerResult {
            cancelled: 0,
            staging_high_water_bytes: 0,
            failure: None,
        };
        if let Some(handle) = self.handle.take() {
            worker = handle.join().map_err(|_| {
                RuntimeError::model("the async fetcher thread panicked; the layer is incomplete")
            })?;
        }
        // After the join every completed delivery is in the unbounded channel, so
        // nothing the worker fetched can escape the accounting.
        while let Ok(done) = self.rx.try_recv() {
            self.account(done);
        }
        self.accounting.shards_cancelled = worker.cancelled.saturating_add(unsent);
        if let Some(failure) = worker.failure
            && self.failure.is_none()
        {
            self.failure = Some(failure);
        }
        // Anything fetched but not consumed is wasted prefetch bytes.
        self.accounting.wasted_bytes = self.accounting.fetched_bytes.saturating_sub(
            self.accounting
                .used_bytes
                .min(self.accounting.fetched_bytes),
        );
        let fetch_busy = self.accounting.stall.fetch_busy_ns;
        let exposed = self.accounting.stall.exposed_stall_ns;
        self.accounting.stall.hidden_ns = fetch_busy.saturating_sub(exposed);
        self.accounting.stall.residual_wait_ns = exposed.saturating_sub(fetch_busy);
        if let Some(failure) = self.failure.take() {
            return Err(RuntimeError::model(failure));
        }
        if !self.accounting.conserves_bytes() {
            return Err(RuntimeError::model(format!(
                "prefetch byte accounting does not conserve: fetched {} used {} wasted {}",
                self.accounting.fetched_bytes,
                self.accounting.used_bytes,
                self.accounting.wasted_bytes
            )));
        }
        if !self.accounting.stall.conserves() {
            return Err(RuntimeError::model(format!(
                "prefetch stall accounting does not conserve: busy {} residual {} hidden {} exposed {}",
                self.accounting.stall.fetch_busy_ns,
                self.accounting.stall.residual_wait_ns,
                self.accounting.stall.hidden_ns,
                self.accounting.stall.exposed_stall_ns
            )));
        }
        Ok(PrefetchOutcome {
            accounting: self.accounting,
            stalls: core::mem::take(&mut self.stalls),
            sources: core::mem::take(&mut self.sources),
        })
    }
}

impl Drop for AsyncFetcher<'_> {
    fn drop(&mut self) {
        self.cancel.store(true, std::sync::atomic::Ordering::SeqCst);
        self.tx = None;
        // The scoped handle joins at the end of the enclosing scope. Dropping the
        // sender above is what makes the worker exit; the flag stops it before
        // its next shard.
    }
}

#[allow(clippy::too_many_arguments)]
fn worker(
    store: &ShardStore,
    scope: &ExecutionScope,
    rx_job: Receiver<Job>,
    tx_done: std::sync::mpsc::Sender<Done>,
    cancel: &AtomicBool,
    staging_bound: u64,
    injected_delay_ns: u64,
) -> WorkerResult {
    let mut buffer: Vec<u8> = Vec::new();
    let mut result = WorkerResult {
        cancelled: 0,
        staging_high_water_bytes: staging_bound,
        failure: None,
    };
    while let Ok(job) = rx_job.recv() {
        if cancel.load(std::sync::atomic::Ordering::SeqCst) {
            let Job::Fetch(_fragment) = job;
            result.cancelled = result.cancelled.saturating_add(1);
            // Drain the rest of the queue as cancelled before exiting.
            while rx_job.try_recv().is_ok() {
                result.cancelled = result.cancelled.saturating_add(1);
            }
            break;
        }
        let Job::Fetch(fragment) = job;
        let length = match usize::try_from(fragment.bytes) {
            Ok(length) => length,
            Err(_) => {
                result.failure = Some("shard byte length does not fit usize".to_owned());
                break;
            }
        };
        if buffer.len() != length {
            buffer.clear();
            buffer.resize(length, 0);
        }
        let staging_bytes = u64::try_from(buffer.len()).unwrap_or(u64::MAX);
        result.staging_high_water_bytes = result.staging_high_water_bytes.max(staging_bytes);
        let started = Instant::now();
        let outcome = store.read_required(scope, &fragment.id, &mut buffer);
        let transfer_ns = u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX);
        let done = match outcome {
            Ok(report) => {
                let mut fetch_ns = transfer_ns;
                if injected_delay_ns > 0 && !report.hit {
                    // The injected delay models device latency, so it belongs to
                    // the fetch timeline, not to the compute timeline.
                    std::thread::sleep(Duration::from_nanos(injected_delay_ns));
                    fetch_ns = fetch_ns.saturating_add(injected_delay_ns);
                }
                Done {
                    shard: fragment.id,
                    bytes: report.bytes_used,
                    device: !report.hit,
                    fetch_ns,
                    staging_bytes,
                    failure: None,
                }
            }
            Err(error) => Done {
                shard: fragment.id,
                bytes: 0,
                device: false,
                fetch_ns: transfer_ns,
                staging_bytes,
                failure: Some(format!(
                    "async prefetch of shard {} failed: {error}",
                    fragment.id.short()
                )),
            },
        };
        let failed = done.failure.clone();
        if tx_done.send(done).is_err() {
            break;
        }
        if let Some(failure) = failed {
            result.failure = Some(failure);
            break;
        }
    }
    result
}
