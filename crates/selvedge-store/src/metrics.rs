//! Request-level observability: JSONL records and aggregate counters.
//!
//! Every read emits one record with the schema string
//! [`METRICS_SCHEMA`]. The documented fields are the accounting contract:
//! bytes requested, completed, used and wasted; queue wait; transfer time;
//! verification time; cache-insertion and eviction cost; exposed stall; and the
//! read amplification measured for that request. Wall-clock and monotonic
//! timestamps are both present so a record can be correlated across processes
//! without trusting a single clock.

use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use selvedge_core::ShardId;
use serde::Serialize;

use crate::error::{Result, StoreError};

/// Schema string for one JSONL record.
pub const METRICS_SCHEMA: &str = "selvedge/store-metrics/v1";

/// One read, as observed by the store.
///
/// Field meanings:
///
/// - `bytes_requested` — bytes the shard needs (its manifest length);
/// - `bytes_completed` — bytes that arrived into the caller's buffer;
/// - `bytes_used` — bytes the caller asked the store to deliver; in exact mode
///   this equals `bytes_completed`, because a whole shard either executes or
///   does not;
/// - `bytes_wasted` — `bytes_completed - bytes_used`; non-zero only for
///   advisory prefetch, which never delivers to the caller;
/// - `device_bytes` — bytes read from the backing tier for this request (zero
///   on a cache hit);
/// - `read_amplification` — `device_bytes / bytes_used`, or `0.0` when nothing
///   was used;
/// - `stall_ns` — time the consumer blocked inside the call (queue wait plus
///   transfer plus verification plus cache insertion).
#[derive(Clone, Debug, Serialize)]
pub struct ReadMetrics {
    /// Always [`METRICS_SCHEMA`].
    pub schema: &'static str,
    /// `read`, `prefetch` or `prefetch_batch`.
    pub event: &'static str,
    /// Wall clock, Unix nanoseconds.
    pub wall_unix_ns: u64,
    /// Monotonic nanoseconds since the store epoch.
    pub mono_ns: u64,
    /// Shard.
    pub shard: ShardId,
    /// `nvme` or `host_ram`.
    pub source: &'static str,
    /// Bytes the shard needs.
    pub bytes_requested: u64,
    /// Bytes that arrived.
    pub bytes_completed: u64,
    /// Bytes delivered for use.
    pub bytes_used: u64,
    /// Bytes moved but not used.
    pub bytes_wasted: u64,
    /// Bytes read from the backing tier.
    pub device_bytes: u64,
    /// Time waiting for the cache lock or queue admission.
    pub queue_wait_ns: u64,
    /// Time moving bytes.
    pub transfer_ns: u64,
    /// Time hashing and comparing.
    pub verify_ns: u64,
    /// Time copying into the cache.
    pub cache_insert_ns: u64,
    /// Time spent selecting and evicting victims.
    pub eviction_ns: u64,
    /// Exposed consumer stall.
    pub stall_ns: u64,
    /// Measured `device_bytes / bytes_used`.
    pub read_amplification: f64,
    /// Whether the bytes came from the cache.
    pub hit: bool,
    /// Whether the bytes were verified before use.
    pub verified: bool,
    /// `sha256_read`, `publication_proof` or `cache_hit`.
    pub verification: &'static str,
    /// Entries evicted to admit this shard.
    pub evicted_entries: u64,
}

/// Aggregate counters, updated with relaxed atomics.
#[derive(Debug, Default)]
pub struct StoreCounters {
    reads: AtomicU64,
    hits: AtomicU64,
    misses: AtomicU64,
    bytes_requested: AtomicU64,
    bytes_completed: AtomicU64,
    bytes_used: AtomicU64,
    bytes_wasted: AtomicU64,
    device_bytes: AtomicU64,
    queue_wait_ns: AtomicU64,
    transfer_ns: AtomicU64,
    verify_ns: AtomicU64,
    stall_ns: AtomicU64,
    prefetch_requests: AtomicU64,
    prefetch_admitted: AtomicU64,
    prefetch_rejected: AtomicU64,
    prefetch_bytes_read: AtomicU64,
    prefetch_bytes_used: AtomicU64,
    checksum_failures: AtomicU64,
    backpressure_rejections: AtomicU64,
    cache_refusals: AtomicU64,
}

impl StoreCounters {
    pub(crate) fn bump(&self, counter: &AtomicU64, by: u64) {
        counter.fetch_add(by, Ordering::Relaxed);
    }

    pub(crate) fn read(&self) -> &AtomicU64 {
        &self.reads
    }
    pub(crate) fn hit(&self) -> &AtomicU64 {
        &self.hits
    }
    pub(crate) fn miss(&self) -> &AtomicU64 {
        &self.misses
    }
    pub(crate) fn bytes_requested(&self) -> &AtomicU64 {
        &self.bytes_requested
    }
    pub(crate) fn bytes_completed(&self) -> &AtomicU64 {
        &self.bytes_completed
    }
    pub(crate) fn bytes_used(&self) -> &AtomicU64 {
        &self.bytes_used
    }
    pub(crate) fn bytes_wasted(&self) -> &AtomicU64 {
        &self.bytes_wasted
    }
    pub(crate) fn device_bytes(&self) -> &AtomicU64 {
        &self.device_bytes
    }
    pub(crate) fn queue_wait_ns(&self) -> &AtomicU64 {
        &self.queue_wait_ns
    }
    pub(crate) fn transfer_ns(&self) -> &AtomicU64 {
        &self.transfer_ns
    }
    pub(crate) fn verify_ns(&self) -> &AtomicU64 {
        &self.verify_ns
    }
    pub(crate) fn stall_ns(&self) -> &AtomicU64 {
        &self.stall_ns
    }
    pub(crate) fn prefetch_requests(&self) -> &AtomicU64 {
        &self.prefetch_requests
    }
    pub(crate) fn prefetch_admitted(&self) -> &AtomicU64 {
        &self.prefetch_admitted
    }
    pub(crate) fn prefetch_rejected(&self) -> &AtomicU64 {
        &self.prefetch_rejected
    }
    pub(crate) fn prefetch_bytes_read(&self) -> &AtomicU64 {
        &self.prefetch_bytes_read
    }
    pub(crate) fn prefetch_bytes_used(&self) -> &AtomicU64 {
        &self.prefetch_bytes_used
    }
    pub(crate) fn checksum_failures(&self) -> &AtomicU64 {
        &self.checksum_failures
    }
    pub(crate) fn backpressure_rejections(&self) -> &AtomicU64 {
        &self.backpressure_rejections
    }
    pub(crate) fn cache_refusals(&self) -> &AtomicU64 {
        &self.cache_refusals
    }

    /// Snapshot as plain numbers.
    #[must_use]
    pub fn snapshot(&self) -> StoreStats {
        let get = |counter: &AtomicU64| counter.load(Ordering::Relaxed);
        let used = get(&self.bytes_used);
        let device = get(&self.device_bytes);
        let prefetch_read = get(&self.prefetch_bytes_read);
        let prefetch_used = get(&self.prefetch_bytes_used);
        StoreStats {
            reads: get(&self.reads),
            hits: get(&self.hits),
            misses: get(&self.misses),
            bytes_requested: get(&self.bytes_requested),
            bytes_completed: get(&self.bytes_completed),
            bytes_used: used,
            bytes_wasted: get(&self.bytes_wasted),
            device_bytes: device,
            read_amplification: if used == 0 {
                0.0
            } else {
                device as f64 / used as f64
            },
            queue_wait_ns: get(&self.queue_wait_ns),
            transfer_ns: get(&self.transfer_ns),
            verify_ns: get(&self.verify_ns),
            stall_ns: get(&self.stall_ns),
            prefetch_requests: get(&self.prefetch_requests),
            prefetch_admitted: get(&self.prefetch_admitted),
            prefetch_rejected: get(&self.prefetch_rejected),
            prefetch_bytes_read: prefetch_read,
            prefetch_bytes_used: prefetch_used,
            prefetch_bytes_wasted: prefetch_read.saturating_sub(prefetch_used),
            cancellations: 0,
            checksum_failures: get(&self.checksum_failures),
            backpressure_rejections: get(&self.backpressure_rejections),
            cache_refusals: get(&self.cache_refusals),
        }
    }
}

/// Plain-number aggregate of [`StoreCounters`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
pub struct StoreStats {
    /// Read calls.
    pub reads: u64,
    /// Cache hits.
    pub hits: u64,
    /// Cache misses.
    pub misses: u64,
    /// Bytes the shards needed.
    pub bytes_requested: u64,
    /// Bytes that arrived.
    pub bytes_completed: u64,
    /// Bytes delivered for use.
    pub bytes_used: u64,
    /// Bytes moved but not used.
    pub bytes_wasted: u64,
    /// Bytes read from the backing tier.
    pub device_bytes: u64,
    /// Aggregate `device_bytes / bytes_used`.
    pub read_amplification: f64,
    /// Total queue wait.
    pub queue_wait_ns: u64,
    /// Total transfer time.
    pub transfer_ns: u64,
    /// Total verification time.
    pub verify_ns: u64,
    /// Total exposed stall.
    pub stall_ns: u64,
    /// Advisory prefetch requests submitted.
    pub prefetch_requests: u64,
    /// Advisory prefetch requests admitted.
    pub prefetch_admitted: u64,
    /// Advisory prefetch requests rejected.
    pub prefetch_rejected: u64,
    /// Bytes read by prefetch.
    pub prefetch_bytes_read: u64,
    /// Prefetched bytes later delivered.
    pub prefetch_bytes_used: u64,
    /// Prefetched bytes never delivered: the measured prefetch waste.
    pub prefetch_bytes_wasted: u64,
    /// Transfers cancelled (a ticket or in-flight owner was dropped).
    pub cancellations: u64,
    /// Checksum failures observed.
    pub checksum_failures: u64,
    /// Backpressure refusals observed.
    pub backpressure_rejections: u64,
    /// Cache budget refusals observed.
    pub cache_refusals: u64,
}

/// JSONL sink. Disabled sinks are a no-op so tests do not write files.
#[derive(Debug)]
pub struct MetricsSink {
    path: Option<PathBuf>,
    writer: Mutex<Option<std::fs::File>>,
    epoch: Instant,
    records: AtomicU64,
}

impl MetricsSink {
    /// A sink that writes nothing.
    #[must_use]
    pub fn disabled() -> Self {
        Self {
            path: None,
            writer: Mutex::new(None),
            epoch: Instant::now(),
            records: AtomicU64::new(0),
        }
    }

    /// A sink that truncates `path` and appends one JSON object per line.
    ///
    /// # Errors
    ///
    /// Rejects an empty path or any file-creation failure.
    pub fn create(path: &Path) -> Result<Self> {
        if path.as_os_str().is_empty() {
            return Err(StoreError::config("metrics path is empty"));
        }
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent)?;
        }
        let file = std::fs::File::create(path)?;
        Ok(Self {
            path: Some(path.to_path_buf()),
            writer: Mutex::new(Some(file)),
            epoch: Instant::now(),
            records: AtomicU64::new(0),
        })
    }

    /// Path being written, if any.
    #[must_use]
    pub fn path(&self) -> Option<&Path> {
        self.path.as_deref()
    }

    /// Monotonic nanoseconds since this sink was created.
    #[must_use]
    pub fn mono_ns(&self) -> u64 {
        u64::try_from(self.epoch.elapsed().as_nanos()).unwrap_or(u64::MAX)
    }

    /// The epoch both clocks in this sink are relative to.
    #[must_use]
    pub const fn epoch(&self) -> Instant {
        self.epoch
    }

    /// Wall clock, Unix nanoseconds.
    #[must_use]
    pub fn wall_unix_ns(&self) -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|duration| u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX))
            .unwrap_or(0)
    }

    /// Records written.
    #[must_use]
    pub fn records(&self) -> u64 {
        self.records.load(Ordering::Relaxed)
    }

    /// Append one record.
    ///
    /// # Errors
    ///
    /// Rejects serialization and write failures; a metrics failure is visible,
    /// never swallowed.
    pub fn record(&self, record: &ReadMetrics) -> Result<()> {
        let mut guard = match self.writer.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };
        let Some(file) = guard.as_mut() else {
            return Ok(());
        };
        let line = serde_json::to_string(record).map_err(|err| StoreError::Metrics {
            reason: err.to_string(),
        })?;
        file.write_all(line.as_bytes())?;
        file.write_all(b"\n")?;
        self.records.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    /// Flush and sync the sink.
    ///
    /// # Errors
    ///
    /// Propagates flush and sync failures.
    pub fn flush(&self) -> Result<()> {
        let mut guard = match self.writer.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };
        if let Some(file) = guard.as_mut() {
            file.flush()?;
            file.sync_data()?;
        }
        Ok(())
    }
}

/// Group per-request latencies into percentiles for a report.
///
/// Returns `(count, p50, p95, p99, max)` in the input unit. The percentile is
/// the nearest-rank definition on the sorted sample.
#[must_use]
pub fn percentiles(samples: &mut [u64]) -> (usize, u64, u64, u64, u64) {
    if samples.is_empty() {
        return (0, 0, 0, 0, 0);
    }
    samples.sort_unstable();
    let count = samples.len();
    let at = |fraction: f64| -> u64 {
        let index = ((fraction * count as f64).ceil() as usize).clamp(1, count) - 1;
        samples[index]
    };
    (count, at(0.50), at(0.95), at(0.99), samples[count - 1])
}

/// Group counters by a string key for compact reports.
pub type CounterMap = BTreeMap<String, u64>;
