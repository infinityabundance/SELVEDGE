//! The tiered shard store: validated bundle reads, bounded cache, advisory
//! prefetch and exact-mode execution scopes.
//!
//! # Exact-mode invariant, encoded in types
//!
//! [`ExecutionScope`] is the set of shards the exact plan will execute. It is
//! immutable and can only be built from an explicit shard list. The only call
//! that returns bytes for execution is [`ShardStore::read_required`], and it
//! refuses a shard outside its scope ([`StoreError::NotInScope`]). Advisory
//! prefetch returns [`PrefetchOutcome`], a record of admissions and rejections,
//! and never returns bytes. There is therefore no API by which prefetch can add
//! a shard to the executed set: it can only change where a later required read
//! is served from. `tests/court_d_extension.rs` asserts the executed set equals
//! the scope after a prefetch that names both in-scope and out-of-scope shards.
//!
//! # Bounded memory
//!
//! The cache never exceeds `cache_bytes`. A required read whose verified bytes
//! cannot be admitted because the budget would be exceeded returns
//! [`StoreError::CacheBudgetExceeded`]; the caller must not use the buffer in
//! that case. Prefetch allocation is bounded by `max_run_bytes`.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use selvedge_core::{Sha256Builder, Sha256Digest, ShardId, TransferId};
use selvedge_format::{BundleFile, Manifest};
use serde::{Deserialize, Serialize};

use crate::cache::{
    CacheOrigin, CachePolicy, CacheStats, CachedShard, LeastStaleCache, LfuCache, LruCache,
    NoCache, PinnedCache, VerificationMethod, VerificationRecord,
};
use crate::error::{Result, StoreError};
use crate::layout::{Layout, LayoutReport};
use crate::metrics::{
    METRICS_SCHEMA, MetricsSink, ReadMetrics, StoreCounters, StoreStats, percentiles,
};
use crate::tier::{QueueLimits, QueueSnapshot, Tier, TransferQueue, elapsed_ns};

/// Cache policy selection.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PolicyKind {
    /// No cache: every read goes to the backing tier.
    None,
    /// Least recently used.
    Lru,
    /// Least frequently used.
    Lfu,
    /// SELVEDGE's own least-stale policy.
    LeastStale,
    /// Pinned residency; admissions beyond the budget are refused.
    Pinned,
}

impl PolicyKind {
    /// Stable name.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Lru => "lru",
            Self::Lfu => "lfu",
            Self::LeastStale => "least_stale",
            Self::Pinned => "pinned",
        }
    }

    fn build(self, capacity_bytes: u64) -> Box<dyn CachePolicy> {
        match self {
            Self::None => Box::new(NoCache::default()),
            Self::Lru => Box::new(LruCache::new(capacity_bytes)),
            Self::Lfu => Box::new(LfuCache::new(capacity_bytes)),
            Self::LeastStale => Box::new(LeastStaleCache::new(capacity_bytes)),
            Self::Pinned => Box::new(PinnedCache::new(capacity_bytes)),
        }
    }
}

/// A recorded verification proof for one shard.
///
/// A proof is only usable when it names the manifest that is actually open: the
/// bundle id, the manifest digest and the shard digest must all match. This is
/// what allows a read to skip hashing, so a mismatch is a hard error.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VerificationProof {
    /// Shard the proof covers.
    pub shard: ShardId,
    /// Digest the shard bytes were verified against.
    pub sha256: Sha256Digest,
    /// Bundle the proof was produced for.
    pub bundle_id: String,
    /// SHA-256 of that bundle's manifest bytes.
    pub manifest_sha256: Sha256Digest,
    /// Wall-clock verification time, Unix nanoseconds.
    pub verified_unix_ns: u64,
    /// Producing tool and command, verbatim.
    pub producer: String,
}

/// A set of publication proofs for one bundle.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProofRegistry {
    /// Bundle id the proofs belong to.
    pub bundle_id: String,
    /// Manifest digest the proofs belong to.
    pub manifest_sha256: Sha256Digest,
    /// Proofs by shard.
    pub proofs: BTreeMap<ShardId, VerificationProof>,
    /// Schema string.
    pub schema: String,
}

impl ProofRegistry {
    /// Schema identifier for a proof registry document.
    pub const SCHEMA: &'static str = "selvedge/verification-proofs/v1";

    /// An empty registry for a bundle.
    #[must_use]
    pub fn new(bundle_id: impl Into<String>, manifest_sha256: Sha256Digest) -> Self {
        Self {
            bundle_id: bundle_id.into(),
            manifest_sha256,
            proofs: BTreeMap::new(),
            schema: Self::SCHEMA.to_owned(),
        }
    }

    /// Load a registry from JSON.
    ///
    /// # Errors
    ///
    /// Rejects unreadable files, malformed JSON and an unknown schema.
    pub fn load(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)?;
        let registry: Self = serde_json::from_str(&text).map_err(|err| {
            StoreError::precondition(format!("cannot parse proof registry {path:?}: {err}"))
        })?;
        if registry.schema != Self::SCHEMA {
            return Err(StoreError::precondition(format!(
                "proof registry schema is {:?}, expected {:?}",
                registry.schema,
                Self::SCHEMA
            )));
        }
        Ok(registry)
    }

    /// Insert a proof, checking it names this registry's bundle.
    ///
    /// # Errors
    ///
    /// Rejects a proof whose bundle or manifest digest differs.
    pub fn insert(&mut self, proof: VerificationProof) -> Result<()> {
        if proof.bundle_id != self.bundle_id {
            return Err(StoreError::ProofRejected {
                reason: format!(
                    "proof names bundle {:?}, registry is {:?}",
                    proof.bundle_id, self.bundle_id
                ),
            });
        }
        if proof.manifest_sha256 != self.manifest_sha256 {
            return Err(StoreError::ProofRejected {
                reason: "proof names a different manifest digest".to_owned(),
            });
        }
        self.proofs.insert(proof.shard, proof);
        Ok(())
    }

    /// Proof for a shard.
    #[must_use]
    pub fn get(&self, shard: &ShardId) -> Option<&VerificationProof> {
        self.proofs.get(shard)
    }
}

/// How reads are verified.
#[derive(Clone, Debug)]
pub enum VerificationMode {
    /// Hash every extent on every read from the backing tier. The default and
    /// the only mode used for any measured exact-mode result in this
    /// repository.
    Always,
    /// Accept a shard without hashing when a proof registry names this bundle
    /// and the manifest digest recorded in it, and the proof's shard digest
    /// equals the manifest's. Shards with no proof are hashed and the fallback
    /// is recorded.
    PublicationProof(Arc<ProofRegistry>),
}

impl VerificationMode {
    /// Stable name used in receipts.
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            Self::Always => "always",
            Self::PublicationProof(_) => "publication_proof",
        }
    }
}

/// Store configuration.
#[derive(Clone, Debug)]
pub struct StoreConfig {
    /// Cache byte budget.
    pub cache_bytes: u64,
    /// Cache policy.
    pub policy: PolicyKind,
    /// Largest extent the caller will read.
    pub max_shard_bytes: u64,
    /// Verification mode.
    pub verification: VerificationMode,
    /// Transfer queue limits.
    pub queue: QueueLimits,
    /// Largest coalesced run a prefetch may allocate.
    pub max_run_bytes: u64,
    /// Gaps up to this size are bridged when coalescing.
    pub coalesce_gap_bytes: u64,
    /// Optional JSONL metrics path.
    pub metrics: Option<PathBuf>,
}

impl Default for StoreConfig {
    fn default() -> Self {
        Self {
            cache_bytes: 0,
            policy: PolicyKind::None,
            max_shard_bytes: 64 * 1024 * 1024,
            verification: VerificationMode::Always,
            queue: QueueLimits {
                max_pending: 64,
                max_in_flight: 16,
                max_in_flight_bytes: 512 * 1024 * 1024,
            },
            max_run_bytes: 32 * 1024 * 1024,
            coalesce_gap_bytes: 0,
            metrics: None,
        }
    }
}

impl StoreConfig {
    /// Validate the configuration.
    ///
    /// # Errors
    ///
    /// Rejects a zero shard cap or run cap, a run cap below the shard cap when
    /// coalescing is enabled, an invalid queue, and a cache budget without a
    /// policy (or a policy without a budget).
    pub fn validate(&self) -> Result<()> {
        if self.max_shard_bytes == 0 {
            return Err(StoreError::config("max_shard_bytes is zero"));
        }
        if self.max_run_bytes == 0 {
            return Err(StoreError::config("max_run_bytes is zero"));
        }
        if self.max_run_bytes < self.max_shard_bytes {
            return Err(StoreError::config(
                "max_run_bytes must be at least max_shard_bytes so a run never splits a shard",
            ));
        }
        if self.policy == PolicyKind::None {
            if self.cache_bytes != 0 {
                return Err(StoreError::config(
                    "cache_bytes must be zero when the policy is 'none'",
                ));
            }
        } else if self.cache_bytes == 0 {
            return Err(StoreError::config(
                "cache_bytes is zero: use policy 'none' for a no-cache baseline",
            ));
        }
        self.queue.validate()?;
        Ok(())
    }
}

/// The immutable set of shards an exact plan will execute.
///
/// Exact mode is a type-level claim here: this set is the only source of
/// execution scope and nothing can extend it after construction.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ExecutionScope {
    ids: BTreeSet<ShardId>,
}

impl ExecutionScope {
    /// Build an exact scope.
    ///
    /// # Errors
    ///
    /// Rejects an empty set and duplicate ids.
    pub fn exact(ids: impl IntoIterator<Item = ShardId>) -> Result<Self> {
        let mut set = BTreeSet::new();
        for id in ids {
            if !set.insert(id) {
                return Err(StoreError::config(format!(
                    "duplicate shard {id:?} in execution scope"
                )));
            }
        }
        if set.is_empty() {
            return Err(StoreError::config("execution scope is empty"));
        }
        Ok(Self { ids: set })
    }

    /// Whether a shard is in the scope.
    #[must_use]
    pub fn contains(&self, id: &ShardId) -> bool {
        self.ids.contains(id)
    }

    /// Scope size.
    #[must_use]
    pub fn len(&self) -> usize {
        self.ids.len()
    }

    /// Whether the scope is empty (always false after construction).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    /// Sorted shard ids.
    #[must_use]
    pub fn ids(&self) -> &BTreeSet<ShardId> {
        &self.ids
    }
}

/// Where a read was served from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReadSource {
    /// The backing bundle.
    Nvme,
    /// The host-RAM cache.
    HostRam,
}

impl ReadSource {
    const fn name(self) -> &'static str {
        match self {
            Self::Nvme => "nvme",
            Self::HostRam => "host_ram",
        }
    }
}

/// Result of one required read.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ReadReport {
    /// Shard delivered.
    pub shard: ShardId,
    /// Where the bytes came from.
    pub source: ReadSource,
    /// Whether the cache served it.
    pub hit: bool,
    /// Bytes the shard needs.
    pub bytes_requested: u64,
    /// Bytes received.
    pub bytes_completed: u64,
    /// Bytes delivered for use.
    pub bytes_used: u64,
    /// Bytes moved but not used.
    pub bytes_wasted: u64,
    /// Bytes read from the backing tier.
    pub device_bytes: u64,
    /// Time waiting for cache access and queue admission.
    pub queue_wait_ns: u64,
    /// Time moving bytes.
    pub transfer_ns: u64,
    /// Time verifying bytes (zero on a cache hit).
    pub verify_ns: u64,
    /// Time copying into the cache.
    pub cache_insert_ns: u64,
    /// Time spent choosing and removing eviction victims.
    pub eviction_ns: u64,
    /// Total time the consumer blocked.
    pub stall_ns: u64,
    /// `device_bytes / bytes_used`.
    pub read_amplification: f64,
    /// Whether the bytes were verified.
    pub verified: bool,
    /// How.
    pub verification: VerificationMethod,
    /// Entries evicted to admit this shard.
    pub evicted_entries: u64,
    /// Transfer id when the read went through the queue.
    pub transfer: Option<TransferId>,
}

/// One advisory prefetch request.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PrefetchHint {
    /// Shard to prefetch.
    pub shard: ShardId,
    /// Deadline in scheduler steps, recorded with the transfer.
    pub deadline_step: Option<u64>,
}

impl PrefetchHint {
    /// Hint for a shard with no deadline.
    #[must_use]
    pub const fn new(shard: ShardId) -> Self {
        Self {
            shard,
            deadline_step: None,
        }
    }
}

/// Why a hint was refused.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PrefetchRejection {
    /// The hint names a shard outside the execution scope. Prefetch cannot
    /// widen execution.
    NotInScope,
    /// The hint names a shard absent from the bundle.
    UnknownShard,
    /// The queue refused the speculative transfer.
    Backpressure,
    /// The cache refused the bytes because the budget would be exceeded.
    CacheBudget,
    /// The hint was a duplicate of another hint in the same call.
    Duplicate,
}

/// What advisory prefetch did. This type carries no bytes.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PrefetchOutcome {
    /// Hints admitted.
    pub admitted: Vec<ShardId>,
    /// Shards already resident and not re-read.
    pub already_resident: Vec<ShardId>,
    /// Refused hints with the reason.
    pub rejected: Vec<(ShardId, PrefetchRejection)>,
    /// Coalesced reads performed.
    pub runs: u64,
    /// Bytes read by run.
    pub bytes_read: u64,
    /// Read calls avoided because adjacent extents were merged.
    pub reads_avoided_by_coalescing: u64,
    /// Bytes retained in the cache by this prefetch.
    pub bytes_admitted: u64,
    /// Wall time spent on the prefetch call.
    pub elapsed_ns: u64,
}

/// Read order for a whole-scope pass.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReadOrder {
    /// Manifest order (lexicographic by shard id).
    Plan,
    /// Ascending physical offset.
    Physical,
    /// Reverse manifest order. Same set, different visit order: used to show
    /// what a partially filled cache is worth without the cyclic-scan
    /// pathology of a forward scan.
    Reverse,
}

/// Result of reading a whole scope.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ScopeReadReport {
    /// Shards read.
    pub shards: u64,
    /// Bytes delivered.
    pub bytes_used: u64,
    /// Wall time for the pass.
    pub elapsed_ns: u64,
    /// Sum of exposed stalls.
    pub stall_ns: u64,
    /// Per-shard stall percentiles in nanoseconds.
    pub stall_p50_ns: u64,
    /// p95.
    pub stall_p95_ns: u64,
    /// p99.
    pub stall_p99_ns: u64,
    /// Slowest shard.
    pub stall_max_ns: u64,
    /// Errors observed; a non-zero value fails the pass.
    pub errors: u64,
}

/// Tiered shard store over one validated bundle.
pub struct ShardStore {
    path: PathBuf,
    file: File,
    manifest: Arc<Manifest>,
    manifest_sha256: Sha256Digest,
    payload_offset: u64,
    layout: Layout,
    cache: Mutex<Box<dyn CachePolicy>>,
    queue: TransferQueue,
    counters: StoreCounters,
    metrics: MetricsSink,
    config: StoreConfig,
    proofs: Option<Arc<ProofRegistry>>,
    epoch: Instant,
}

impl std::fmt::Debug for ShardStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ShardStore")
            .field("path", &self.path)
            .field("bundle_id", &self.manifest.bundle_id)
            .field("shards", &self.manifest.shards.len())
            .field("policy", &self.config.policy.name())
            .field("cache_bytes", &self.config.cache_bytes)
            .field("verification", &self.config.verification.name())
            .finish()
    }
}

impl ShardStore {
    /// Open a validated bundle.
    ///
    /// Structural validation runs through `selvedge_format::BundleFile`; the
    /// store then opens its own read-only handle for positional reads. The
    /// bundle is immutable and publication is atomic, so both handles observe
    /// the same inode.
    ///
    /// # Errors
    ///
    /// Propagates validator failures, an invalid configuration, an extent
    /// larger than `max_shard_bytes`, and metrics-file creation failures.
    pub fn open(path: &Path, config: StoreConfig) -> Result<Self> {
        config.validate()?;
        let bundle = BundleFile::open(path)?;
        let manifest = Arc::new(bundle.manifest().clone());
        let manifest_sha256 = bundle.manifest_sha256();
        let payload_offset = bundle.payload_offset();
        let layout = Layout::from_manifest(&manifest)?;
        let largest = layout.max_extent_bytes();
        if largest > config.max_shard_bytes {
            return Err(StoreError::config(format!(
                "largest extent is {largest} bytes, above max_shard_bytes {}",
                config.max_shard_bytes
            )));
        }
        let file = File::open(path)?;
        let proofs = match &config.verification {
            VerificationMode::PublicationProof(registry) => {
                if registry.bundle_id != manifest.bundle_id {
                    return Err(StoreError::ProofRejected {
                        reason: format!(
                            "proof registry is for bundle {:?}, opened {:?}",
                            registry.bundle_id, manifest.bundle_id
                        ),
                    });
                }
                if registry.manifest_sha256 != manifest_sha256 {
                    return Err(StoreError::ProofRejected {
                        reason: "proof registry names a different manifest digest".to_owned(),
                    });
                }
                Some(Arc::clone(registry))
            }
            VerificationMode::Always => None,
        };
        let metrics = match &config.metrics {
            Some(path) => MetricsSink::create(path)?,
            None => MetricsSink::disabled(),
        };
        let epoch = metrics.epoch();
        let cache = config.policy.build(config.cache_bytes);
        Ok(Self {
            path: path.to_path_buf(),
            file,
            manifest,
            manifest_sha256,
            payload_offset,
            layout,
            cache: Mutex::new(cache),
            queue: TransferQueue::new(config.queue)?,
            counters: StoreCounters::default(),
            metrics,
            config,
            proofs,
            epoch,
        })
    }

    /// Bundle path.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Bundle id.
    #[must_use]
    pub fn bundle_id(&self) -> &str {
        &self.manifest.bundle_id
    }

    /// Manifest digest.
    #[must_use]
    pub const fn manifest_sha256(&self) -> Sha256Digest {
        self.manifest_sha256
    }

    /// Parsed manifest.
    #[must_use]
    pub fn manifest(&self) -> &Manifest {
        &self.manifest
    }

    /// Physical layout summary.
    #[must_use]
    pub fn layout_report(&self) -> LayoutReport {
        self.layout.report()
    }

    /// All shard ids in manifest order.
    #[must_use]
    pub fn shard_ids(&self) -> Vec<ShardId> {
        self.manifest
            .shards
            .iter()
            .map(|shard| shard.shard_id)
            .collect()
    }

    /// All shard ids in ascending physical order.
    #[must_use]
    pub fn physical_shard_ids(&self) -> Vec<ShardId> {
        self.layout.physical_order(None)
    }

    /// Aggregate read counters.
    #[must_use]
    pub fn stats(&self) -> StoreStats {
        let mut stats = self.counters.snapshot();
        stats.cancellations = self.queue.snapshot().stats.cancelled;
        stats
    }

    /// Cache counters.
    #[must_use]
    pub fn cache_stats(&self) -> CacheStats {
        self.lock_cache().stats()
    }

    /// Transfer queue snapshot.
    #[must_use]
    pub fn queue_snapshot(&self) -> QueueSnapshot {
        self.queue.snapshot()
    }

    /// Metrics sink record count.
    #[must_use]
    pub fn metrics_records(&self) -> u64 {
        self.metrics.records()
    }

    /// Monotonic nanoseconds since the store opened.
    #[must_use]
    pub fn mono_ns(&self) -> u64 {
        elapsed_ns(self.epoch)
    }

    /// Pin a resident shard. Returns false when it is not resident.
    pub fn pin(&self, id: &ShardId) -> bool {
        self.lock_cache().pin(id)
    }

    /// Number of pinned entries.
    #[must_use]
    pub fn pin_all_resident(&self) -> u64 {
        let ids: Vec<ShardId> = self.lock_cache().resident_ids();
        let mut cache = self.lock_cache();
        let mut pinned = 0;
        for id in ids {
            if cache.pin(&id) {
                pinned += 1;
            }
        }
        pinned
    }

    /// Read one shard required by the exact plan.
    ///
    /// # Errors
    ///
    /// Rejects a shard outside `scope`, a wrong buffer length, an unknown
    /// shard, a queue refusal, a read failure, a checksum mismatch and a cache
    /// budget refusal. On a cache budget refusal the buffer contents are
    /// unspecified and must not be used.
    pub fn read_required(
        &self,
        scope: &ExecutionScope,
        id: &ShardId,
        buffer: &mut [u8],
    ) -> Result<ReadReport> {
        let entry_start = Instant::now();
        let extent = *self
            .layout
            .extent(id)
            .ok_or(StoreError::UnknownShard(*id))?;
        if !scope.contains(id) {
            return Err(StoreError::NotInScope { shard: *id });
        }
        if buffer.len() as u64 != extent.len {
            return Err(StoreError::BufferSize {
                shard: *id,
                expected: extent.len,
                actual: buffer.len() as u64,
            });
        }
        // Cache lookup.
        let lookup_start = Instant::now();
        let hit = self.lock_cache().get(id);
        let queue_wait_ns = u64::try_from(lookup_start.elapsed().as_nanos()).unwrap_or(u64::MAX);
        if let Some(cached) = hit {
            if cached.bytes.len() != buffer.len() {
                return Err(StoreError::BufferSize {
                    shard: *id,
                    expected: cached.bytes.len() as u64,
                    actual: buffer.len() as u64,
                });
            }
            buffer.copy_from_slice(&cached.bytes);
            if cached.origin == CacheOrigin::Prefetch {
                self.counters
                    .bump(self.counters.prefetch_bytes_used(), extent.len);
            }
            self.counters.bump(self.counters.read(), 1);
            self.counters.bump(self.counters.hit(), 1);
            self.counters
                .bump(self.counters.bytes_requested(), extent.len);
            self.counters
                .bump(self.counters.bytes_completed(), extent.len);
            self.counters.bump(self.counters.bytes_used(), extent.len);
            let stall_ns = u64::try_from(entry_start.elapsed().as_nanos()).unwrap_or(u64::MAX);
            self.counters.bump(self.counters.stall_ns(), stall_ns);
            self.counters
                .bump(self.counters.queue_wait_ns(), queue_wait_ns);
            let report = ReadReport {
                shard: *id,
                source: ReadSource::HostRam,
                hit: true,
                bytes_requested: extent.len,
                bytes_completed: extent.len,
                bytes_used: extent.len,
                bytes_wasted: 0,
                device_bytes: 0,
                queue_wait_ns,
                transfer_ns: 0,
                verify_ns: 0,
                cache_insert_ns: 0,
                eviction_ns: 0,
                stall_ns,
                read_amplification: 0.0,
                verified: true,
                verification: cached.proof.method,
                evicted_entries: 0,
                transfer: None,
            };
            self.record(&report)?;
            return Ok(report);
        }

        // Miss: required read through the bounded queue.
        let ticket =
            match self
                .queue
                .submit(*id, extent.len, Tier::Nvme, Tier::HostRam, false, None)
            {
                Ok(ticket) => ticket,
                Err(error) => {
                    self.counters
                        .bump(self.counters.backpressure_rejections(), 1);
                    return Err(error);
                }
            };
        let transfer_id = ticket.id();
        let in_flight = ticket.start();
        let transfer_start = Instant::now();
        let offset =
            self.payload_offset
                .checked_add(extent.offset)
                .ok_or(StoreError::Overflow {
                    context: "shard file offset",
                })?;
        if let Err(error) = self.file.read_exact_at(buffer, offset) {
            let _ = in_flight.fail("positional read failed");
            return Err(StoreError::Io(error));
        }
        let transfer_ns = u64::try_from(transfer_start.elapsed().as_nanos()).unwrap_or(u64::MAX);
        let verify_start = Instant::now();
        let record = match self.verify(*id, buffer) {
            Ok(record) => record,
            Err(error) => {
                if matches!(error, StoreError::Checksum { .. }) {
                    self.counters.bump(self.counters.checksum_failures(), 1);
                }
                let _ = in_flight.fail("verification failed");
                return Err(error);
            }
        };
        let verify_ns = u64::try_from(verify_start.elapsed().as_nanos()).unwrap_or(u64::MAX);
        let _ = in_flight.complete();

        // Admit verified bytes to the cache.
        let insert_start = Instant::now();
        let cached = CachedShard {
            id: *id,
            bytes: Arc::from(buffer),
            proof: record.clone(),
            origin: CacheOrigin::Read,
        };
        let outcome = match self.lock_cache().insert(cached) {
            Ok(outcome) => outcome,
            Err(error) => {
                self.counters.bump(self.counters.cache_refusals(), 1);
                return Err(error);
            }
        };
        let cache_insert_ns = u64::try_from(insert_start.elapsed().as_nanos()).unwrap_or(u64::MAX);

        self.counters.bump(self.counters.read(), 1);
        self.counters.bump(self.counters.miss(), 1);
        self.counters
            .bump(self.counters.bytes_requested(), extent.len);
        self.counters
            .bump(self.counters.bytes_completed(), extent.len);
        self.counters.bump(self.counters.bytes_used(), extent.len);
        self.counters.bump(self.counters.device_bytes(), extent.len);
        self.counters.bump(self.counters.transfer_ns(), transfer_ns);
        self.counters.bump(self.counters.verify_ns(), verify_ns);
        self.counters
            .bump(self.counters.queue_wait_ns(), queue_wait_ns);
        let stall_ns = u64::try_from(entry_start.elapsed().as_nanos()).unwrap_or(u64::MAX);
        self.counters.bump(self.counters.stall_ns(), stall_ns);
        let report = ReadReport {
            shard: *id,
            source: ReadSource::Nvme,
            hit: false,
            bytes_requested: extent.len,
            bytes_completed: extent.len,
            bytes_used: extent.len,
            bytes_wasted: 0,
            device_bytes: extent.len,
            queue_wait_ns,
            transfer_ns,
            verify_ns,
            cache_insert_ns,
            eviction_ns: outcome.eviction_ns,
            stall_ns,
            read_amplification: 1.0,
            verified: true,
            verification: record.method,
            evicted_entries: outcome.evicted.len() as u64,
            transfer: Some(transfer_id),
        };
        self.record(&report)?;
        Ok(report)
    }

    /// Read every shard in the scope, in `order`, into one reusable buffer.
    ///
    /// The executed set is exactly the scope. Order changes only the sequence of
    /// positional reads, never membership.
    ///
    /// # Errors
    ///
    /// Rejects a buffer smaller than the largest extent and propagates read
    /// failures.
    pub fn read_scope(
        &self,
        scope: &ExecutionScope,
        buffer: &mut [u8],
        order: ReadOrder,
    ) -> Result<ScopeReadReport> {
        if (buffer.len() as u64) < self.layout.max_extent_bytes() {
            return Err(StoreError::BufferSize {
                shard: *scope
                    .ids()
                    .iter()
                    .next()
                    .ok_or(StoreError::config("execution scope is empty"))?,
                expected: self.layout.max_extent_bytes(),
                actual: buffer.len() as u64,
            });
        }
        let order_ids: Vec<ShardId> = match order {
            ReadOrder::Plan => scope.ids().iter().copied().collect(),
            ReadOrder::Physical => self.layout.physical_order(Some(scope.ids())),
            ReadOrder::Reverse => scope.ids().iter().rev().copied().collect(),
        };
        let start = Instant::now();
        let mut samples = Vec::with_capacity(order_ids.len());
        let mut bytes_used = 0_u64;
        let mut stall_ns = 0_u64;
        let mut errors = 0_u64;
        for id in order_ids {
            let extent = *self
                .layout
                .extent(&id)
                .ok_or(StoreError::UnknownShard(id))?;
            let slice = &mut buffer[..extent.len as usize];
            match self.read_required(scope, &id, slice) {
                Ok(report) => {
                    bytes_used = bytes_used.saturating_add(report.bytes_used);
                    stall_ns = stall_ns.saturating_add(report.stall_ns);
                    samples.push(report.stall_ns);
                }
                Err(_) => errors += 1,
            }
        }
        let (_, p50, p95, p99, max) = percentiles(&mut samples);
        Ok(ScopeReadReport {
            shards: scope.len() as u64,
            bytes_used,
            elapsed_ns: u64::try_from(start.elapsed().as_nanos()).unwrap_or(u64::MAX),
            stall_ns,
            stall_p50_ns: p50,
            stall_p95_ns: p95,
            stall_p99_ns: p99,
            stall_max_ns: max,
            errors,
        })
    }

    /// Read the scope using coalesced adjacent runs, verifying every shard.
    ///
    /// Reads are merged where the layout allows; the hash check still covers
    /// each shard's own bytes. This is the physical-layout variant of
    /// [`ShardStore::read_scope`], not a different execution set.
    ///
    /// # Errors
    ///
    /// Rejects runs larger than `max_run_bytes`, allocation failures, queue
    /// refusals and checksum mismatches.
    pub fn read_scope_coalesced(&self, scope: &ExecutionScope) -> Result<ScopeReadReport> {
        let runs = self.layout.runs(
            Some(scope.ids()),
            self.config.max_run_bytes,
            self.config.coalesce_gap_bytes,
        )?;
        let start = Instant::now();
        let mut samples = Vec::with_capacity(scope.len());
        let mut bytes_used = 0_u64;
        let mut stall_ns = 0_u64;
        for run in &runs {
            let mut buffer = Vec::new();
            buffer
                .try_reserve_exact(run.length as usize)
                .map_err(|_| StoreError::Overflow {
                    context: "run buffer",
                })?;
            buffer.resize(run.length as usize, 0);
            let run_start = Instant::now();
            let ticket = self.queue.submit(
                run.parts[0].shard,
                run.length,
                Tier::Nvme,
                Tier::HostRam,
                true,
                None,
            )?;
            let in_flight = ticket.start();
            let offset =
                self.payload_offset
                    .checked_add(run.offset)
                    .ok_or(StoreError::Overflow {
                        context: "run file offset",
                    })?;
            if let Err(error) = self.file.read_exact_at(&mut buffer, offset) {
                let _ = in_flight.fail("run read failed");
                return Err(StoreError::Io(error));
            }
            let _ = in_flight.complete();
            for part in &run.parts {
                let begin = part.offset as usize;
                let end = begin + part.len as usize;
                let record = self.verify(part.shard, &buffer[begin..end])?;
                bytes_used = bytes_used.saturating_add(part.len);
                self.counters.bump(self.counters.read(), 1);
                self.counters.bump(self.counters.miss(), 1);
                self.counters
                    .bump(self.counters.bytes_requested(), part.len);
                self.counters
                    .bump(self.counters.bytes_completed(), part.len);
                self.counters.bump(self.counters.bytes_used(), part.len);
                self.counters.bump(self.counters.device_bytes(), part.len);
                let cached = CachedShard {
                    id: part.shard,
                    bytes: Arc::from(&buffer[begin..end]),
                    proof: record,
                    origin: CacheOrigin::Read,
                };
                let _ = self.lock_cache().insert(cached);
            }
            let stall = u64::try_from(run_start.elapsed().as_nanos()).unwrap_or(u64::MAX);
            stall_ns = stall_ns.saturating_add(stall);
            for _part in &run.parts {
                samples.push(stall / run.parts.len() as u64);
            }
        }
        self.counters.bump(self.counters.stall_ns(), stall_ns);
        let (_, p50, p95, p99, max) = percentiles(&mut samples);
        Ok(ScopeReadReport {
            shards: scope.len() as u64,
            bytes_used,
            elapsed_ns: u64::try_from(start.elapsed().as_nanos()).unwrap_or(u64::MAX),
            stall_ns,
            stall_p50_ns: p50,
            stall_p95_ns: p95,
            stall_p99_ns: p99,
            stall_max_ns: max,
            errors: 0,
        })
    }

    /// Submit advisory prefetch hints for shards in the plan.
    ///
    /// Hints outside the scope are refused. The call returns no bytes; it can
    /// only populate the cache. Adjacent extents among the admitted hints are
    /// coalesced into one positional read bounded by `max_run_bytes`.
    ///
    /// # Errors
    ///
    /// Propagates allocation, read and metrics failures. Per-hint refusals are
    /// recorded in the outcome rather than returned, because advisory work is
    /// refusable by contract.
    pub fn prefetch_advisory(
        &self,
        scope: &ExecutionScope,
        hints: &[PrefetchHint],
    ) -> Result<PrefetchOutcome> {
        let start = Instant::now();
        let mut admitted: Vec<ShardId> = Vec::new();
        let mut already_resident: Vec<ShardId> = Vec::new();
        let mut rejected: Vec<(ShardId, PrefetchRejection)> = Vec::new();
        let mut seen: BTreeSet<ShardId> = BTreeSet::new();
        for hint in hints {
            self.counters.bump(self.counters.prefetch_requests(), 1);
            if !scope.contains(&hint.shard) {
                self.counters.bump(self.counters.prefetch_rejected(), 1);
                rejected.push((hint.shard, PrefetchRejection::NotInScope));
                continue;
            }
            if self.layout.extent(&hint.shard).is_none() {
                self.counters.bump(self.counters.prefetch_rejected(), 1);
                rejected.push((hint.shard, PrefetchRejection::UnknownShard));
                continue;
            }
            if !seen.insert(hint.shard) {
                self.counters.bump(self.counters.prefetch_rejected(), 1);
                rejected.push((hint.shard, PrefetchRejection::Duplicate));
                continue;
            }
            if self.lock_cache().contains(&hint.shard) {
                already_resident.push(hint.shard);
                continue;
            }
            admitted.push(hint.shard);
        }
        admitted.sort_by_key(|id| self.layout.extent(id).map(|extent| extent.offset));
        let selected: BTreeSet<ShardId> = admitted.iter().copied().collect();
        let runs = self.layout.runs(
            Some(&selected),
            self.config.max_run_bytes,
            self.config.coalesce_gap_bytes,
        )?;
        let mut bytes_read = 0_u64;
        let mut separate_reads = 0_u64;
        let mut bytes_admitted = 0_u64;
        let mut runs_done = 0_u64;
        let mut admitted_final: Vec<ShardId> = Vec::new();
        for run in &runs {
            let ticket = match self.queue.submit(
                run.parts[0].shard,
                run.length,
                Tier::Nvme,
                Tier::HostRam,
                true,
                None,
            ) {
                Ok(ticket) => ticket,
                Err(_) => {
                    self.counters.bump(self.counters.prefetch_rejected(), 1);
                    for part in &run.parts {
                        rejected.push((part.shard, PrefetchRejection::Backpressure));
                    }
                    continue;
                }
            };
            let in_flight = ticket.start();
            let mut buffer = Vec::new();
            buffer
                .try_reserve_exact(run.length as usize)
                .map_err(|_| StoreError::Overflow {
                    context: "prefetch run buffer",
                })?;
            buffer.resize(run.length as usize, 0);
            let offset =
                self.payload_offset
                    .checked_add(run.offset)
                    .ok_or(StoreError::Overflow {
                        context: "prefetch file offset",
                    })?;
            if let Err(error) = self.file.read_exact_at(&mut buffer, offset) {
                let _ = in_flight.fail("prefetch read failed");
                return Err(StoreError::Io(error));
            }
            let _ = in_flight.complete();
            runs_done += 1;
            bytes_read = bytes_read.saturating_add(run.length);
            separate_reads = separate_reads.saturating_add(run.parts.len() as u64);
            let mut run_admitted = true;
            for part in &run.parts {
                let begin = part.offset as usize;
                let end = begin + part.len as usize;
                let record = match self.verify(part.shard, &buffer[begin..end]) {
                    Ok(record) => record,
                    Err(error) => {
                        if matches!(error, StoreError::Checksum { .. }) {
                            self.counters.bump(self.counters.checksum_failures(), 1);
                        }
                        self.counters.bump(self.counters.prefetch_rejected(), 1);
                        rejected.push((part.shard, PrefetchRejection::CacheBudget));
                        run_admitted = false;
                        continue;
                    }
                };
                let cached = CachedShard {
                    id: part.shard,
                    bytes: Arc::from(&buffer[begin..end]),
                    proof: record,
                    origin: CacheOrigin::Prefetch,
                };
                match self.lock_cache().insert(cached) {
                    Ok(_) => {
                        self.counters.bump(self.counters.prefetch_admitted(), 1);
                        self.counters
                            .bump(self.counters.prefetch_bytes_read(), part.len);
                        bytes_admitted = bytes_admitted.saturating_add(part.len);
                        admitted_final.push(part.shard);
                    }
                    Err(_) => {
                        self.counters.bump(self.counters.cache_refusals(), 1);
                        self.counters.bump(self.counters.prefetch_rejected(), 1);
                        rejected.push((part.shard, PrefetchRejection::CacheBudget));
                        run_admitted = false;
                    }
                }
            }
            if !run_admitted {
                // Keep the outcome honest: the bytes were read but not retained.
                self.counters.bump(self.counters.bytes_wasted(), run.length);
            }
        }
        let outcome = PrefetchOutcome {
            admitted: admitted_final,
            already_resident,
            rejected,
            runs: runs_done,
            bytes_read,
            reads_avoided_by_coalescing: separate_reads.saturating_sub(runs_done),
            bytes_admitted,
            elapsed_ns: u64::try_from(start.elapsed().as_nanos()).unwrap_or(u64::MAX),
        };
        let record = ReadMetrics {
            schema: METRICS_SCHEMA,
            event: "prefetch_batch",
            wall_unix_ns: self.metrics.wall_unix_ns(),
            mono_ns: self.metrics.mono_ns(),
            shard: outcome.admitted.first().copied().unwrap_or_else(|| {
                ShardId::from_digest(Sha256Digest::of(b"selvedge-prefetch-batch"))
            }),
            source: "nvme",
            bytes_requested: outcome.bytes_read,
            bytes_completed: outcome.bytes_read,
            bytes_used: 0,
            bytes_wasted: outcome.bytes_read.saturating_sub(outcome.bytes_admitted),
            device_bytes: outcome.bytes_read,
            queue_wait_ns: 0,
            transfer_ns: outcome.elapsed_ns,
            verify_ns: 0,
            cache_insert_ns: 0,
            eviction_ns: 0,
            stall_ns: 0,
            read_amplification: 0.0,
            hit: false,
            verified: true,
            verification: match self.config.verification {
                VerificationMode::Always => "sha256_read",
                VerificationMode::PublicationProof(_) => "publication_proof",
            },
            evicted_entries: 0,
        };
        self.metrics.record(&record)?;
        Ok(outcome)
    }

    /// Flush the metrics sink.
    ///
    /// # Errors
    ///
    /// Propagates flush and sync failures.
    pub fn flush_metrics(&self) -> Result<()> {
        self.metrics.flush()
    }

    fn verify(&self, id: ShardId, bytes: &[u8]) -> Result<VerificationRecord> {
        let shard = self
            .manifest
            .shard(&id)
            .ok_or(StoreError::UnknownShard(id))?;
        if let Some(registry) = &self.proofs
            && let Some(proof) = registry.get(&id)
        {
            if proof.sha256 != shard.sha256 {
                return Err(StoreError::ProofRejected {
                    reason: format!("proof digest for {id:?} differs from the manifest digest"),
                });
            }
            return Ok(VerificationRecord {
                shard: id,
                sha256: shard.sha256,
                method: VerificationMethod::PublicationProof,
                verified_unix_ns: proof.verified_unix_ns,
            });
        }
        let mut hasher = Sha256Builder::new();
        hasher.update(bytes);
        let actual = hasher.finish();
        if actual != shard.sha256 {
            return Err(StoreError::Checksum {
                shard: id,
                expected: shard.sha256,
                actual,
            });
        }
        Ok(VerificationRecord {
            shard: id,
            sha256: actual,
            method: VerificationMethod::Sha256Read,
            verified_unix_ns: wall_unix_ns(),
        })
    }

    fn record(&self, report: &ReadReport) -> Result<()> {
        if self.metrics.path().is_none() {
            return Ok(());
        }
        let record = ReadMetrics {
            schema: METRICS_SCHEMA,
            event: "read",
            wall_unix_ns: self.metrics.wall_unix_ns(),
            mono_ns: self.metrics.mono_ns(),
            shard: report.shard,
            source: report.source.name(),
            bytes_requested: report.bytes_requested,
            bytes_completed: report.bytes_completed,
            bytes_used: report.bytes_used,
            bytes_wasted: report.bytes_wasted,
            device_bytes: report.device_bytes,
            queue_wait_ns: report.queue_wait_ns,
            transfer_ns: report.transfer_ns,
            verify_ns: report.verify_ns,
            cache_insert_ns: report.cache_insert_ns,
            eviction_ns: report.eviction_ns,
            stall_ns: report.stall_ns,
            read_amplification: report.read_amplification,
            hit: report.hit,
            verified: report.verified,
            verification: match report.verification {
                VerificationMethod::Sha256Read => "sha256_read",
                VerificationMethod::PublicationProof => "publication_proof",
            },
            evicted_entries: report.evicted_entries,
        };
        self.metrics.record(&record)
    }

    fn lock_cache(&self) -> std::sync::MutexGuard<'_, Box<dyn CachePolicy>> {
        match self.cache.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        }
    }
}

fn wall_unix_ns() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX))
        .unwrap_or(0)
}
