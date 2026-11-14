//! Bounded cache policies behind one trait.
//!
//! The cache never exceeds its byte capacity. `insert` evicts according to the
//! policy; when the policy can find no victim (for example because every entry
//! is pinned) the insertion is **refused** with
//! [`StoreError::CacheBudgetExceeded`] and the caller's read fails closed. That
//! is deliberate: an over-budget cache is a configuration bug, not something to
//! absorb silently.
//!
//! Policies implemented here:
//!
//! - `lru` — least recently used, deterministic tie-break by shard id;
//! - `lfu` — least frequently used, then least recently used, then id;
//! - `least_stale` — **our own** policy (`selvedge-least-stale-v1`), not a
//!   reproduction of a published replacement rule: evict the entry with the
//!   largest staleness, protecting entries used at the current sequence number,
//!   then prefer entries with fewer hits, then id. It is labelled our own in
//!   every report and is not claimed to reproduce any named prior policy;
//! - `pinned` — nothing is ever evicted; admissions beyond the budget are
//!   refused;
//! - `none` — a no-cache baseline that counts every lookup as a miss.

use std::collections::BTreeMap;
use std::sync::Arc;

use selvedge_core::{Sha256Digest, ShardId};
use serde::{Deserialize, Serialize};

use crate::error::{Result, StoreError};

/// How the bytes in a cache entry were verified.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VerificationMethod {
    /// The bytes were hashed on this read, before insertion.
    Sha256Read,
    /// The bytes were accepted under a publication proof registry entry.
    PublicationProof,
}

/// Proof that an entry's bytes matched the manifest digest.
///
/// A cache hit is served without re-hashing **because** this record exists. No
/// cache entry may exist without one.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VerificationRecord {
    /// Shard.
    pub shard: ShardId,
    /// Digest the bytes were checked against.
    pub sha256: Sha256Digest,
    /// How the check was performed.
    pub method: VerificationMethod,
    /// Wall-clock time of verification, Unix nanoseconds.
    pub verified_unix_ns: u64,
}

/// Where an entry came from, so a prefetch hit is counted as prefetch value.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CacheOrigin {
    /// Fetched by a required read.
    Read,
    /// Fetched by advisory prefetch.
    Prefetch,
}

/// A verified, immutable cache entry.
#[derive(Clone, Debug)]
pub struct CachedShard {
    /// Shard id.
    pub id: ShardId,
    /// Verified bytes.
    pub bytes: Arc<[u8]>,
    /// Verification record for those bytes.
    pub proof: VerificationRecord,
    /// Origin of the bytes.
    pub origin: CacheOrigin,
}

/// One evicted entry.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvictedEntry {
    /// Shard removed.
    pub shard: ShardId,
    /// Bytes reclaimed.
    pub bytes: u64,
    /// Hits observed while resident; zero means the bytes were never reused.
    pub hits: u64,
}

/// Result of an insertion attempt.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InsertOutcome {
    /// Whether the entry is now resident.
    pub admitted: bool,
    /// Entries evicted to make room.
    pub evicted: Vec<EvictedEntry>,
    /// Bytes reclaimed by eviction.
    pub bytes_evicted: u64,
    /// Time spent selecting and removing victims.
    pub eviction_ns: u64,
}

/// Counters every policy exposes.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CacheStats {
    /// Policy name.
    pub policy: String,
    /// Configured capacity in bytes.
    pub capacity_bytes: u64,
    /// Resident bytes.
    pub used_bytes: u64,
    /// Resident entries.
    pub entries: u64,
    /// Pinned entries.
    pub pinned_entries: u64,
    /// Lookups served from the cache.
    pub hits: u64,
    /// Lookups that required the backing tier.
    pub misses: u64,
    /// Insertions attempted.
    pub inserts: u64,
    /// Successful insertions.
    pub admissions: u64,
    /// Evictions performed.
    pub evictions: u64,
    /// Bytes reclaimed by eviction.
    pub evicted_bytes: u64,
    /// Reclaimed bytes whose entries were never hit again (prefetch waste at
    /// the cache level).
    pub evicted_unused_bytes: u64,
    /// Insertions refused by the budget.
    pub refusals: u64,
    /// Bytes that never entered a policy because it has no storage.
    pub bypassed_bytes: u64,
}

impl CacheStats {
    /// Hit ratio as a fraction of lookups, or zero when there were none.
    #[must_use]
    pub fn hit_ratio(&self) -> f64 {
        let lookups = self.hits.saturating_add(self.misses);
        if lookups == 0 {
            return 0.0;
        }
        self.hits as f64 / lookups as f64
    }
}

/// Cache policy contract.
///
/// Implementations are used behind a `Mutex` inside [`crate::ShardStore`] and
/// must be `Send`. They never perform I/O.
pub trait CachePolicy: std::fmt::Debug + Send {
    /// Stable policy name used in receipts.
    fn name(&self) -> &'static str;

    /// Look up and count a hit or miss.
    fn get(&mut self, id: &ShardId) -> Option<CachedShard>;

    /// Insert verified bytes, evicting as the policy allows.
    ///
    /// # Errors
    ///
    /// Returns [`StoreError::CacheBudgetExceeded`] when the entry cannot be
    /// admitted without exceeding the capacity.
    fn insert(&mut self, shard: CachedShard) -> Result<InsertOutcome>;

    /// Pin a resident entry. Returns false when the entry is absent.
    fn pin(&mut self, id: &ShardId) -> bool;

    /// Unpin a resident entry. Returns false when it was not pinned.
    fn unpin(&mut self, id: &ShardId) -> bool;

    /// Membership test that does not change counters.
    fn contains(&self, id: &ShardId) -> bool;

    /// Resident shard ids, for pinning sweeps and reports.
    fn resident_ids(&self) -> Vec<ShardId>;

    /// Counters.
    fn stats(&self) -> CacheStats;
}

pub(crate) struct Entry {
    shard: CachedShard,
    pinned: bool,
    last_used_seq: u64,
    hits: u64,
}

impl std::fmt::Debug for Entry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Entry")
            .field("shard", &self.shard.id)
            .field("bytes", &self.shard.bytes.len())
            .field("pinned", &self.pinned)
            .field("hits", &self.hits)
            .finish()
    }
}

pub(crate) trait VictimSelector {
    fn victim(
        &self,
        entries: &BTreeMap<ShardId, Entry>,
        seq: u64,
        exclude: &std::collections::BTreeSet<ShardId>,
    ) -> Option<ShardId>;
}

#[derive(Debug)]
pub(crate) struct CacheCore {
    capacity: u64,
    used: u64,
    entries: BTreeMap<ShardId, Entry>,
    seq: u64,
    hits: u64,
    misses: u64,
    inserts: u64,
    admissions: u64,
    evictions: u64,
    evicted_bytes: u64,
    evicted_unused_bytes: u64,
    refusals: u64,
}

impl CacheCore {
    fn new(capacity: u64) -> Self {
        Self {
            capacity,
            used: 0,
            entries: BTreeMap::new(),
            seq: 0,
            hits: 0,
            misses: 0,
            inserts: 0,
            admissions: 0,
            evictions: 0,
            evicted_bytes: 0,
            evicted_unused_bytes: 0,
            refusals: 0,
        }
    }

    fn next_seq(&mut self) -> u64 {
        let value = self.seq;
        self.seq = self.seq.saturating_add(1);
        value
    }

    fn get(&mut self, id: &ShardId) -> Option<CachedShard> {
        let seq = self.next_seq();
        match self.entries.get_mut(id) {
            Some(entry) => {
                entry.last_used_seq = seq;
                entry.hits = entry.hits.saturating_add(1);
                let shard = entry.shard.clone();
                self.hits = self.hits.saturating_add(1);
                Some(shard)
            }
            None => {
                self.misses = self.misses.saturating_add(1);
                None
            }
        }
    }

    fn insert(
        &mut self,
        shard: CachedShard,
        selector: &dyn VictimSelector,
    ) -> Result<InsertOutcome> {
        self.inserts = self.inserts.saturating_add(1);
        let bytes = u64::try_from(shard.bytes.len()).map_err(|_| StoreError::Overflow {
            context: "cache entry bytes",
        })?;
        if bytes > self.capacity {
            self.refusals = self.refusals.saturating_add(1);
            return Err(StoreError::CacheBudgetExceeded {
                requested: bytes,
                used: self.used,
                capacity: self.capacity,
            });
        }
        // A replacement of the same key does not need eviction room.
        if let Some(existing) = self.entries.remove(&shard.id) {
            self.used = self.used.saturating_sub(byte_len(&existing));
        }
        // Plan the eviction set first so a refusal leaves the cache untouched.
        let mut victims: Vec<ShardId> = Vec::new();
        let mut excluded: std::collections::BTreeSet<ShardId> = std::collections::BTreeSet::new();
        let mut projected = self.used;
        let start = std::time::Instant::now();
        while projected.saturating_add(bytes) > self.capacity {
            let seq = self.seq;
            let Some(victim) = selector.victim(&self.entries, seq, &excluded) else {
                self.refusals = self.refusals.saturating_add(1);
                return Err(StoreError::CacheBudgetExceeded {
                    requested: bytes,
                    used: self.used,
                    capacity: self.capacity,
                });
            };
            match self.entries.get(&victim) {
                Some(entry) => {
                    projected = projected.saturating_sub(byte_len(entry));
                    excluded.insert(victim);
                    victims.push(victim);
                }
                None => break,
            }
        }
        let mut evicted = Vec::new();
        let mut bytes_evicted = 0_u64;
        for victim in victims {
            if let Some(entry) = self.entries.remove(&victim) {
                let victim_bytes = byte_len(&entry);
                self.used = self.used.saturating_sub(victim_bytes);
                bytes_evicted = bytes_evicted.saturating_add(victim_bytes);
                self.evictions = self.evictions.saturating_add(1);
                self.evicted_bytes = self.evicted_bytes.saturating_add(victim_bytes);
                if entry.hits == 0 {
                    self.evicted_unused_bytes =
                        self.evicted_unused_bytes.saturating_add(victim_bytes);
                }
                evicted.push(EvictedEntry {
                    shard: victim,
                    bytes: victim_bytes,
                    hits: entry.hits,
                });
            }
        }
        let eviction_ns = u64::try_from(start.elapsed().as_nanos()).unwrap_or(u64::MAX);
        let seq = self.next_seq();
        self.used = self.used.saturating_add(bytes);
        self.entries.insert(
            shard.id,
            Entry {
                shard,
                pinned: false,
                last_used_seq: seq,
                hits: 0,
            },
        );
        self.admissions = self.admissions.saturating_add(1);
        Ok(InsertOutcome {
            admitted: true,
            evicted,
            bytes_evicted,
            eviction_ns,
        })
    }

    fn pin(&mut self, id: &ShardId) -> bool {
        match self.entries.get_mut(id) {
            Some(entry) => {
                entry.pinned = true;
                true
            }
            None => false,
        }
    }

    fn unpin(&mut self, id: &ShardId) -> bool {
        match self.entries.get_mut(id) {
            Some(entry) if entry.pinned => {
                entry.pinned = false;
                true
            }
            _ => false,
        }
    }

    fn stats(&self, policy: &'static str) -> CacheStats {
        CacheStats {
            policy: policy.to_owned(),
            capacity_bytes: self.capacity,
            used_bytes: self.used,
            entries: self.entries.len() as u64,
            pinned_entries: self.entries.values().filter(|entry| entry.pinned).count() as u64,
            hits: self.hits,
            misses: self.misses,
            inserts: self.inserts,
            admissions: self.admissions,
            evictions: self.evictions,
            evicted_bytes: self.evicted_bytes,
            evicted_unused_bytes: self.evicted_unused_bytes,
            refusals: self.refusals,
            bypassed_bytes: 0,
        }
    }
}

fn byte_len(entry: &Entry) -> u64 {
    u64::try_from(entry.shard.bytes.len()).unwrap_or(u64::MAX)
}

/// Least recently used.
#[derive(Debug)]
pub struct LruCache {
    core: CacheCore,
}

impl LruCache {
    /// Create an LRU cache with a byte capacity.
    #[must_use]
    pub fn new(capacity_bytes: u64) -> Self {
        Self {
            core: CacheCore::new(capacity_bytes),
        }
    }
}

#[derive(Debug)]
struct LruVictim;

impl VictimSelector for LruVictim {
    fn victim(
        &self,
        entries: &BTreeMap<ShardId, Entry>,
        _seq: u64,
        exclude: &std::collections::BTreeSet<ShardId>,
    ) -> Option<ShardId> {
        entries
            .iter()
            .filter(|(id, entry)| !entry.pinned && !exclude.contains(id))
            .min_by_key(|(id, entry)| (entry.last_used_seq, **id))
            .map(|(id, _)| *id)
    }
}

impl CachePolicy for LruCache {
    fn name(&self) -> &'static str {
        "lru"
    }

    fn get(&mut self, id: &ShardId) -> Option<CachedShard> {
        self.core.get(id)
    }

    fn insert(&mut self, shard: CachedShard) -> Result<InsertOutcome> {
        self.core.insert(shard, &LruVictim)
    }

    fn pin(&mut self, id: &ShardId) -> bool {
        self.core.pin(id)
    }

    fn unpin(&mut self, id: &ShardId) -> bool {
        self.core.unpin(id)
    }

    fn contains(&self, id: &ShardId) -> bool {
        self.core.entries.contains_key(id)
    }

    fn resident_ids(&self) -> Vec<ShardId> {
        self.core.entries.keys().copied().collect()
    }

    fn stats(&self) -> CacheStats {
        self.core.stats("lru")
    }
}

/// Least frequently used, then least recently used.
#[derive(Debug)]
pub struct LfuCache {
    core: CacheCore,
}

impl LfuCache {
    /// Create an LFU cache with a byte capacity.
    #[must_use]
    pub fn new(capacity_bytes: u64) -> Self {
        Self {
            core: CacheCore::new(capacity_bytes),
        }
    }
}

#[derive(Debug)]
struct LfuVictim;

impl VictimSelector for LfuVictim {
    fn victim(
        &self,
        entries: &BTreeMap<ShardId, Entry>,
        _seq: u64,
        exclude: &std::collections::BTreeSet<ShardId>,
    ) -> Option<ShardId> {
        entries
            .iter()
            .filter(|(id, entry)| !entry.pinned && !exclude.contains(id))
            .min_by_key(|(id, entry)| (entry.hits, entry.last_used_seq, **id))
            .map(|(id, _)| *id)
    }
}

impl CachePolicy for LfuCache {
    fn name(&self) -> &'static str {
        "lfu"
    }

    fn get(&mut self, id: &ShardId) -> Option<CachedShard> {
        self.core.get(id)
    }

    fn insert(&mut self, shard: CachedShard) -> Result<InsertOutcome> {
        self.core.insert(shard, &LfuVictim)
    }

    fn pin(&mut self, id: &ShardId) -> bool {
        self.core.pin(id)
    }

    fn unpin(&mut self, id: &ShardId) -> bool {
        self.core.unpin(id)
    }

    fn contains(&self, id: &ShardId) -> bool {
        self.core.entries.contains_key(id)
    }

    fn resident_ids(&self) -> Vec<ShardId> {
        self.core.entries.keys().copied().collect()
    }

    fn stats(&self) -> CacheStats {
        self.core.stats("lfu")
    }
}

/// SELVEDGE's own least-stale policy.
///
/// The eviction key is `(staleness, hits, id)`, maximised, where staleness is
/// `seq - last_used_seq`. Entries used at the current sequence number are
/// protected while any other candidate exists. This is our own rule, labelled
/// `selvedge-least-stale-v1`; no claim is made that it reproduces a published
/// policy, and its measured behaviour is reported in `docs/STORAGE.md`.
#[derive(Debug)]
pub struct LeastStaleCache {
    core: CacheCore,
}

impl LeastStaleCache {
    /// Create a least-stale cache with a byte capacity.
    #[must_use]
    pub fn new(capacity_bytes: u64) -> Self {
        Self {
            core: CacheCore::new(capacity_bytes),
        }
    }
}

#[derive(Debug)]
struct LeastStaleVictim;

impl VictimSelector for LeastStaleVictim {
    fn victim(
        &self,
        entries: &BTreeMap<ShardId, Entry>,
        seq: u64,
        exclude: &std::collections::BTreeSet<ShardId>,
    ) -> Option<ShardId> {
        let mut best: Option<(u64, u64, ShardId)> = None;
        for (id, entry) in entries {
            if entry.pinned || exclude.contains(id) {
                continue;
            }
            let staleness = seq.saturating_sub(entry.last_used_seq);
            // Skip the just-used entries on the first pass; they are only
            // evicted if nothing else is available.
            let key = (staleness, u64::MAX.saturating_sub(entry.hits), *id);
            match &best {
                Some(current) if *current >= key => {}
                _ => best = Some(key),
            }
        }
        best.map(|(_, _, id)| id)
    }
}

impl CachePolicy for LeastStaleCache {
    fn name(&self) -> &'static str {
        "least_stale"
    }

    fn get(&mut self, id: &ShardId) -> Option<CachedShard> {
        self.core.get(id)
    }

    fn insert(&mut self, shard: CachedShard) -> Result<InsertOutcome> {
        self.core.insert(shard, &LeastStaleVictim)
    }

    fn pin(&mut self, id: &ShardId) -> bool {
        self.core.pin(id)
    }

    fn unpin(&mut self, id: &ShardId) -> bool {
        self.core.unpin(id)
    }

    fn contains(&self, id: &ShardId) -> bool {
        self.core.entries.contains_key(id)
    }

    fn resident_ids(&self) -> Vec<ShardId> {
        self.core.entries.keys().copied().collect()
    }

    fn stats(&self) -> CacheStats {
        self.core.stats("least_stale")
    }
}

/// Pinned residency: no eviction, admissions beyond the budget are refused.
#[derive(Debug)]
pub struct PinnedCache {
    core: CacheCore,
}

impl PinnedCache {
    /// Create a pinned cache with a byte capacity.
    #[must_use]
    pub fn new(capacity_bytes: u64) -> Self {
        Self {
            core: CacheCore::new(capacity_bytes),
        }
    }
}

#[derive(Debug)]
struct NoVictim;

impl VictimSelector for NoVictim {
    fn victim(
        &self,
        _entries: &BTreeMap<ShardId, Entry>,
        _seq: u64,
        _exclude: &std::collections::BTreeSet<ShardId>,
    ) -> Option<ShardId> {
        None
    }
}

impl CachePolicy for PinnedCache {
    fn name(&self) -> &'static str {
        "pinned"
    }

    fn get(&mut self, id: &ShardId) -> Option<CachedShard> {
        self.core.get(id)
    }

    fn insert(&mut self, shard: CachedShard) -> Result<InsertOutcome> {
        let outcome = self.core.insert(shard, &NoVictim)?;
        // Everything admitted is pinned by construction.
        let keys: Vec<ShardId> = self.core.entries.keys().copied().collect();
        for key in keys {
            self.core.pin(&key);
        }
        Ok(outcome)
    }

    fn pin(&mut self, id: &ShardId) -> bool {
        self.core.pin(id)
    }

    fn unpin(&mut self, id: &ShardId) -> bool {
        self.core.unpin(id)
    }

    fn contains(&self, id: &ShardId) -> bool {
        self.core.entries.contains_key(id)
    }

    fn resident_ids(&self) -> Vec<ShardId> {
        self.core.entries.keys().copied().collect()
    }

    fn stats(&self) -> CacheStats {
        self.core.stats("pinned")
    }
}

/// No-cache baseline: every lookup misses and every insertion is discarded.
#[derive(Debug, Default)]
pub struct NoCache {
    misses: u64,
    hits: u64,
    bypassed_bytes: u64,
    inserts: u64,
}

impl CachePolicy for NoCache {
    fn name(&self) -> &'static str {
        "none"
    }

    fn get(&mut self, _id: &ShardId) -> Option<CachedShard> {
        self.misses = self.misses.saturating_add(1);
        None
    }

    fn insert(&mut self, shard: CachedShard) -> Result<InsertOutcome> {
        self.inserts = self.inserts.saturating_add(1);
        self.bypassed_bytes = self
            .bypassed_bytes
            .saturating_add(u64::try_from(shard.bytes.len()).unwrap_or(u64::MAX));
        Ok(InsertOutcome {
            admitted: false,
            evicted: Vec::new(),
            bytes_evicted: 0,
            eviction_ns: 0,
        })
    }

    fn pin(&mut self, _id: &ShardId) -> bool {
        false
    }

    fn unpin(&mut self, _id: &ShardId) -> bool {
        false
    }

    fn contains(&self, _id: &ShardId) -> bool {
        false
    }

    fn resident_ids(&self) -> Vec<ShardId> {
        Vec::new()
    }

    fn stats(&self) -> CacheStats {
        CacheStats {
            policy: "none".to_owned(),
            capacity_bytes: 0,
            used_bytes: 0,
            entries: 0,
            pinned_entries: 0,
            hits: self.hits,
            misses: self.misses,
            inserts: self.inserts,
            admissions: 0,
            evictions: 0,
            evicted_bytes: 0,
            evicted_unused_bytes: 0,
            refusals: 0,
            bypassed_bytes: self.bypassed_bytes,
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn shard(fill: u8, bytes: usize) -> CachedShard {
        let id = ShardId::from_digest(Sha256Digest::of(&[fill]));
        CachedShard {
            id,
            bytes: Arc::from(vec![fill; bytes]),
            proof: VerificationRecord {
                shard: id,
                sha256: Sha256Digest::of(&vec![fill; bytes]),
                method: VerificationMethod::Sha256Read,
                verified_unix_ns: 1,
            },
            origin: CacheOrigin::Read,
        }
    }

    #[test]
    fn lru_evicts_least_recently_used_deterministically() {
        let mut cache = LruCache::new(3);
        let a = shard(1, 1);
        let b = shard(2, 1);
        let c = shard(3, 1);
        let (a_id, b_id, c_id) = (a.id, b.id, c.id);
        assert!(cache.insert(a).unwrap().admitted);
        assert!(cache.insert(b).unwrap().admitted);
        assert!(cache.insert(c).unwrap().admitted);
        assert!(cache.get(&a_id).is_some());
        assert!(cache.get(&b_id).is_some());
        let outcome = cache.insert(shard(4, 1)).unwrap();
        assert_eq!(outcome.evicted.len(), 1);
        assert_eq!(outcome.evicted[0].shard, c_id);
        assert_eq!(cache.stats().evictions, 1);
        assert_eq!(cache.stats().hits, 2);
    }

    #[test]
    fn lfu_prefers_the_least_frequently_used_entry() {
        let mut cache = LfuCache::new(2);
        let a = shard(1, 1);
        let b = shard(2, 1);
        let (a_id, b_id) = (a.id, b.id);
        cache.insert(a).unwrap();
        cache.insert(b).unwrap();
        for _ in 0..3 {
            assert!(cache.get(&a_id).is_some());
        }
        let outcome = cache.insert(shard(3, 1)).unwrap();
        assert_eq!(outcome.evicted[0].shard, b_id);
        assert!(cache.contains(&a_id));
    }

    #[test]
    fn least_stale_protects_the_current_sequence_entry() {
        let mut cache = LeastStaleCache::new(2);
        let a = shard(1, 1);
        let b = shard(2, 1);
        let (a_id, b_id) = (a.id, b.id);
        cache.insert(a).unwrap();
        cache.insert(b).unwrap();
        assert!(cache.get(&b_id).is_some());
        let outcome = cache.insert(shard(3, 1)).unwrap();
        assert_eq!(outcome.evicted[0].shard, a_id);
        assert!(cache.contains(&b_id));
    }

    #[test]
    fn pinned_cache_refuses_rather_than_evicting() {
        let mut cache = PinnedCache::new(2);
        cache.insert(shard(1, 1)).unwrap();
        cache.insert(shard(2, 1)).unwrap();
        let error = cache.insert(shard(3, 1)).unwrap_err();
        assert!(matches!(error, StoreError::CacheBudgetExceeded { .. }));
        assert_eq!(cache.stats().refusals, 1);
        assert_eq!(cache.stats().pinned_entries, 2);
    }

    #[test]
    fn no_cache_counts_every_lookup_as_a_miss() {
        let mut cache = NoCache::default();
        let value = shard(1, 4);
        let id = value.id;
        assert!(!cache.insert(value).unwrap().admitted);
        assert!(cache.get(&id).is_none());
        let stats = cache.stats();
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.bypassed_bytes, 4);
    }

    #[test]
    fn waste_is_counted_for_evicted_unused_bytes() {
        let mut cache = LruCache::new(2);
        cache.insert(shard(1, 1)).unwrap();
        cache.insert(shard(2, 1)).unwrap();
        let outcome = cache.insert(shard(3, 1)).unwrap();
        assert_eq!(outcome.evicted[0].hits, 0);
        assert_eq!(cache.stats().evicted_unused_bytes, 1);
    }
}
