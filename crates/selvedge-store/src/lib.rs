//! Tiered shard storage and measured physical I/O for SELVEDGE.
//!
//! This crate owns the path between a validated bundle on NVMe and a caller's
//! buffer: positional reads, SHA-256 verification, a byte-bounded cache with
//! pluggable replacement policies, advisory prefetch, a cancellation-safe
//! transfer state machine and per-request observability. It also owns the
//! physical I/O measurement used as the M3 baseline.
//!
//! Invariants:
//!
//! - **Exact mode is typed.** [`ExecutionScope`](store::ExecutionScope) is the
//!   immutable set of shards an exact plan will execute. Only
//!   [`ShardStore::read_required`](store::ShardStore::read_required) returns
//!   bytes, and it refuses anything outside that scope. Advisory prefetch cannot
//!   widen the executed set; it returns no bytes.
//! - **Integrity fails closed.** Every extent is checked against the manifest
//!   digest before use. A cache hit is only possible on an entry that carries a
//!   [`VerificationRecord`](cache::VerificationRecord) produced when the bytes
//!   were verified. A proof is usable only if it names the open bundle's id and
//!   manifest digest.
//! - **Memory is bounded.** The cache never exceeds its configured byte budget;
//!   an admission that cannot fit is refused with
//!   [`StoreError::CacheBudgetExceeded`]. Prefetch buffers are bounded by
//!   `max_run_bytes`, caller reads by the caller's own buffer.
//! - **Backpressure is visible.** The transfer queue has a bounded pending count,
//!   in-flight count and in-flight byte budget. A refusal is an error the caller
//!   can see and count, never a silent drop.
//! - **Numbers come from measurement.** [`io_scan`] reports device counters and
//!   page-cache control state next to every timing.

#![forbid(unsafe_code)]
#![deny(missing_docs)]

pub mod cache;
pub mod error;
pub mod io_scan;
pub mod layout;
pub mod metrics;
pub mod pagecache;
pub mod store;
pub mod store_bench;
pub mod sysfs;
pub mod tier;

pub use cache::{
    CacheOrigin, CachePolicy, CacheStats, CachedShard, EvictedEntry, InsertOutcome,
    LeastStaleCache, LfuCache, LruCache, NoCache, PinnedCache, VerificationMethod,
    VerificationRecord,
};
pub use error::{Result, StoreError};
pub use layout::{Extent, Layout, LayoutReport, Run, RunPart};
pub use metrics::{METRICS_SCHEMA, MetricsSink, ReadMetrics, StoreStats, percentiles};
pub use pagecache::{AdviceReport, PageCache};
pub use store::{
    ExecutionScope, PolicyKind, PrefetchHint, PrefetchOutcome, PrefetchRejection, ProofRegistry,
    ReadOrder, ReadReport, ReadSource, ScopeReadReport, ShardStore, StoreConfig, VerificationMode,
    VerificationProof,
};
pub use store_bench::{BenchConfig, BenchReport, STORE_BENCH_SCHEMA};
pub use sysfs::{DeviceCounters, Environment, MemoryInfo, MountInfo};
pub use tier::{
    InFlightTransfer, QueueLimits, QueueSnapshot, QueueStats, Residency, Tier, Transfer,
    TransferQueue, TransferState, TransferTicket,
};
