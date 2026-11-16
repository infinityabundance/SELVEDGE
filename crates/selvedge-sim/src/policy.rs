//! Cache policies and advisory prefetch predictors.
//!
//! The cache policies reproduce the semantics of the measured engine in
//! `selvedge-store` exactly, including tie-breaks and the sequence-number
//! protocol, so that a simulated hit/miss sequence can be compared with a
//! recorded one. They are re-implemented rather than reused because the store's
//! `CachedShard` owns verified bytes and an `Arc<[u8]>`, while the simulator
//! tracks residency only; sharing would force the simulator to allocate and
//! hash bytes it never has.
//!
//! A prefetch predictor is **advisory**. It returns candidates; the engine
//! decides when to move them, and a wrong prediction becomes a measured stall
//! or wasted bytes. It cannot add a shard to a record's selected set, and the
//! engine never reads the selection from a predictor.

use std::collections::{BTreeMap, BTreeSet};

use selvedge_core::ShardId;
use selvedge_store::PolicyKind;

use crate::envelope::Envelope;
use crate::error::{Result, SimError};
use crate::trace::{TraceRecord, TraceShard};

/// One resident entry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Entry {
    bytes: u64,
    pinned: bool,
    last_used_seq: u64,
    hits: u64,
}

/// Result of an insertion attempt.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InsertOutcome {
    /// Whether the bytes are now resident.
    pub admitted: bool,
    /// Bytes reclaimed by eviction.
    pub evicted_bytes: u64,
    /// Entries evicted.
    pub evictions: u64,
    /// Evicted bytes that had never been reused.
    pub evicted_unused_bytes: u64,
}

/// Counter snapshot for one simulation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CacheCounters {
    /// Lookups served from host RAM.
    pub hits: u64,
    /// Lookups that missed.
    pub misses: u64,
    /// Insertion attempts.
    pub inserts: u64,
    /// Admissions.
    pub admissions: u64,
    /// Evictions.
    pub evictions: u64,
    /// Evicted bytes.
    pub evicted_bytes: u64,
    /// Evicted bytes that were never reused.
    pub evicted_unused_bytes: u64,
    /// Admissions refused because the policy could not free room.
    pub refusals: u64,
}

/// Byte-bounded cache with the measured engine's replacement policies.
#[derive(Clone, Debug)]
pub struct SimCache {
    policy: PolicyKind,
    capacity: u64,
    used: u64,
    seq: u64,
    entries: BTreeMap<ShardId, Entry>,
    counters: CacheCounters,
}

impl SimCache {
    /// Create a cache. A zero capacity with `policy != none` is rejected, as in
    /// the measured engine.
    ///
    /// # Errors
    ///
    /// Rejects a zero-capacity policy-backed cache.
    pub fn new(policy: PolicyKind, capacity: u64) -> Result<Self> {
        if policy != PolicyKind::None && capacity == 0 {
            return Err(SimError::invalid(
                "cache_bytes",
                "a policy-backed cache needs a non-zero capacity",
            ));
        }
        Ok(Self {
            policy,
            capacity,
            used: 0,
            seq: 0,
            entries: BTreeMap::new(),
            counters: CacheCounters::default(),
        })
    }

    /// Policy name.
    #[must_use]
    pub const fn policy(&self) -> PolicyKind {
        self.policy
    }

    /// Configured capacity.
    #[must_use]
    pub const fn capacity(&self) -> u64 {
        self.capacity
    }

    /// Resident bytes.
    #[must_use]
    pub const fn used(&self) -> u64 {
        self.used
    }

    /// Resident entry count.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether nothing is resident.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Counters.
    #[must_use]
    pub const fn counters(&self) -> CacheCounters {
        self.counters
    }

    /// Resident shard ids, ascending.
    #[must_use]
    pub fn resident_ids(&self) -> Vec<ShardId> {
        self.entries.keys().copied().collect()
    }

    /// Whether a shard is resident, without changing counters.
    #[must_use]
    pub fn contains(&self, id: &ShardId) -> bool {
        self.entries.contains_key(id)
    }

    /// Look up a shard, updating recency and frequency exactly as the measured
    /// engine does.
    pub fn get(&mut self, id: &ShardId) -> bool {
        let seq = self.next_seq();
        match self.entries.get_mut(id) {
            Some(entry) => {
                entry.last_used_seq = seq;
                entry.hits = entry.hits.saturating_add(1);
                self.counters.hits = self.counters.hits.saturating_add(1);
                true
            }
            None => {
                self.counters.misses = self.counters.misses.saturating_add(1);
                false
            }
        }
    }

    /// Pin a resident shard.
    pub fn pin(&mut self, id: &ShardId) -> bool {
        match self.entries.get_mut(id) {
            Some(entry) => {
                entry.pinned = true;
                true
            }
            None => false,
        }
    }

    /// Unpin a resident shard.
    pub fn unpin(&mut self, id: &ShardId) -> bool {
        match self.entries.get_mut(id) {
            Some(entry) if entry.pinned => {
                entry.pinned = false;
                true
            }
            _ => false,
        }
    }

    /// Insert a shard of `bytes`.
    ///
    /// # Errors
    ///
    /// Rejects a zero-byte shard.
    pub fn insert(&mut self, id: ShardId, bytes: u64) -> Result<InsertOutcome> {
        if bytes == 0 {
            return Err(SimError::invalid("insert", "shard has zero bytes"));
        }
        self.counters.inserts = self.counters.inserts.saturating_add(1);
        if self.policy == PolicyKind::None {
            self.counters.refusals = self.counters.refusals.saturating_add(1);
            return Ok(InsertOutcome {
                admitted: false,
                evicted_bytes: 0,
                evictions: 0,
                evicted_unused_bytes: 0,
            });
        }
        if bytes > self.capacity {
            self.counters.refusals = self.counters.refusals.saturating_add(1);
            return Ok(InsertOutcome {
                admitted: false,
                evicted_bytes: 0,
                evictions: 0,
                evicted_unused_bytes: 0,
            });
        }
        if let Some(existing) = self.entries.remove(&id) {
            self.used = self.used.saturating_sub(existing.bytes);
        }
        let mut victims: Vec<ShardId> = Vec::new();
        let mut excluded: BTreeSet<ShardId> = BTreeSet::new();
        let mut projected = self.used;
        while projected.saturating_add(bytes) > self.capacity {
            let Some(victim) = self.victim(&excluded) else {
                // Restore the replaced entry, then refuse.
                if let Some(existing) = victims.first().and_then(|id| self.entries.get(id)) {
                    self.used = self.used.saturating_add(existing.bytes);
                }
                self.counters.refusals = self.counters.refusals.saturating_add(1);
                return Ok(InsertOutcome {
                    admitted: false,
                    evicted_bytes: 0,
                    evictions: 0,
                    evicted_unused_bytes: 0,
                });
            };
            if let Some(entry) = self.entries.get(&victim) {
                projected = projected.saturating_sub(entry.bytes);
                excluded.insert(victim);
                victims.push(victim);
            } else {
                break;
            }
        }
        let mut evicted_bytes = 0_u64;
        let mut evicted_unused = 0_u64;
        for victim in &victims {
            if let Some(entry) = self.entries.remove(victim) {
                self.used = self.used.saturating_sub(entry.bytes);
                evicted_bytes = evicted_bytes.saturating_add(entry.bytes);
                if entry.hits == 0 {
                    evicted_unused = evicted_unused.saturating_add(entry.bytes);
                }
            }
        }
        self.counters.evictions = self.counters.evictions.saturating_add(victims.len() as u64);
        self.counters.evicted_bytes = self.counters.evicted_bytes.saturating_add(evicted_bytes);
        self.counters.evicted_unused_bytes = self
            .counters
            .evicted_unused_bytes
            .saturating_add(evicted_unused);
        let seq = self.next_seq();
        self.used = self.used.saturating_add(bytes);
        self.entries.insert(
            id,
            Entry {
                bytes,
                pinned: self.policy == PolicyKind::Pinned,
                last_used_seq: seq,
                hits: 0,
            },
        );
        self.counters.admissions = self.counters.admissions.saturating_add(1);
        Ok(InsertOutcome {
            admitted: true,
            evicted_bytes,
            evictions: victims.len() as u64,
            evicted_unused_bytes: evicted_unused,
        })
    }

    fn next_seq(&mut self) -> u64 {
        let value = self.seq;
        self.seq = self.seq.saturating_add(1);
        value
    }

    fn victim(&self, exclude: &BTreeSet<ShardId>) -> Option<ShardId> {
        let candidates = || {
            self.entries
                .iter()
                .filter(move |(id, entry)| !entry.pinned && !exclude.contains(*id))
        };
        match self.policy {
            PolicyKind::None | PolicyKind::Pinned => None,
            PolicyKind::Lru => candidates()
                .min_by_key(|(id, entry)| (entry.last_used_seq, **id))
                .map(|(id, _)| *id),
            PolicyKind::Lfu => candidates()
                .min_by_key(|(id, entry)| (entry.hits, entry.last_used_seq, **id))
                .map(|(id, _)| *id),
            PolicyKind::LeastStale => candidates()
                .max_by_key(|(id, entry)| {
                    let staleness = self.seq.saturating_sub(entry.last_used_seq);
                    (staleness, u64::MAX.saturating_sub(entry.hits), **id)
                })
                .map(|(id, _)| *id),
        }
    }
}

/// Context handed to a prefetch predictor.
#[derive(Debug)]
pub struct PredictContext<'a> {
    /// Record index being predicted for.
    pub record_index: usize,
    /// Current record.
    pub current: &'a TraceRecord,
    /// Previous record in the trace, if any.
    pub previous: Option<&'a TraceRecord>,
    /// Next record in the trace, if any. Only a labelled oracle predictor may
    /// read this; it is future information.
    pub next: Option<&'a TraceRecord>,
    /// Active envelope, if the policy has one.
    pub envelope: Option<&'a Envelope>,
    /// Cache capacity, so a predictor can size its working set.
    pub capacity_bytes: u64,
    /// Shard table.
    pub table: &'a BTreeMap<ShardId, TraceShard>,
}

/// An advisory prefetch predictor.
pub trait Predictor: std::fmt::Debug {
    /// Stable name used in receipts.
    fn name(&self) -> &'static str;

    /// Candidates to prefetch after the current record.
    fn predict(&mut self, context: &PredictContext<'_>) -> Vec<ShardId>;

    /// Whether the predictor uses future information.
    ///
    /// A `true` value must be reported as non-deployable in any result that
    /// uses it.
    fn uses_future(&self) -> bool {
        false
    }
}

/// No advisory prefetch.
#[derive(Clone, Copy, Debug, Default)]
pub struct NoPrefetch;

impl Predictor for NoPrefetch {
    fn name(&self) -> &'static str {
        "none"
    }

    fn predict(&mut self, _context: &PredictContext<'_>) -> Vec<ShardId> {
        Vec::new()
    }
}

/// Causal last-value predictor: assume the next step resembles this one.
#[derive(Clone, Copy, Debug, Default)]
pub struct LastValuePrefetch;

impl Predictor for LastValuePrefetch {
    fn name(&self) -> &'static str {
        "last-value"
    }

    fn predict(&mut self, context: &PredictContext<'_>) -> Vec<ShardId> {
        context.current.selected.clone()
    }
}

/// Envelope-driven advisory prefetch. This is the deployable predictive policy:
/// it reads only the envelope, which was built from a disjoint window.
#[derive(Clone, Copy, Debug, Default)]
pub struct EnvelopePrefetch;

impl Predictor for EnvelopePrefetch {
    fn name(&self) -> &'static str {
        "envelope"
    }

    fn predict(&mut self, context: &PredictContext<'_>) -> Vec<ShardId> {
        context
            .envelope
            .map(|envelope| envelope.candidate.clone())
            .unwrap_or_default()
    }
}

/// Oracle next-record prefetch. **Non-deployable**: it reads the next record's
/// selection, which a real system does not have. It is an upper bound on what
/// any predictor could save in exact mode, never a result.
#[derive(Clone, Copy, Debug, Default)]
pub struct OracleNextRecordPrefetch;

impl Predictor for OracleNextRecordPrefetch {
    fn name(&self) -> &'static str {
        "oracle-next-record"
    }

    fn predict(&mut self, context: &PredictContext<'_>) -> Vec<ShardId> {
        context
            .next
            .map(|record| record.selected.clone())
            .unwrap_or_default()
    }

    fn uses_future(&self) -> bool {
        true
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use selvedge_core::Sha256Digest;

    fn id(tag: u8) -> ShardId {
        ShardId::from_digest(Sha256Digest::of(&[tag]))
    }

    #[test]
    fn no_cache_never_hits_and_never_admits() {
        let mut cache = SimCache::new(PolicyKind::None, 0).unwrap();
        assert!(!cache.get(&id(1)));
        let outcome = cache.insert(id(1), 100).unwrap();
        assert!(!outcome.admitted);
        assert!(!cache.contains(&id(1)));
        assert_eq!(cache.used(), 0);
        assert_eq!(cache.counters().misses, 1);
        assert_eq!(cache.counters().refusals, 1);
    }

    #[test]
    fn capacity_boundary_exact_fit_is_admitted() {
        let mut cache = SimCache::new(PolicyKind::Lru, 200).unwrap();
        assert!(cache.insert(id(1), 100).unwrap().admitted);
        assert!(cache.insert(id(2), 100).unwrap().admitted);
        assert_eq!(cache.used(), 200);
        // One byte more needs an eviction.
        let outcome = cache.insert(id(3), 1).unwrap();
        assert!(outcome.admitted);
        assert_eq!(outcome.evictions, 1);
        assert_eq!(cache.used(), 101);
        assert!(!cache.contains(&id(1)));
    }

    #[test]
    fn lru_evicts_oldest_with_id_tiebreak() {
        let mut cache = SimCache::new(PolicyKind::Lru, 200).unwrap();
        cache.insert(id(1), 100).unwrap();
        cache.insert(id(2), 100).unwrap();
        assert!(cache.get(&id(1)));
        // id(2) is now the least recently used.
        let outcome = cache.insert(id(3), 100).unwrap();
        assert_eq!(outcome.evictions, 1);
        assert!(!cache.contains(&id(2)));
        assert!(cache.contains(&id(1)));
        // Recency dominates: id(9) was inserted first and is the victim.
        let mut cache = SimCache::new(PolicyKind::Lru, 200).unwrap();
        cache.insert(id(9), 100).unwrap();
        cache.insert(id(4), 100).unwrap();
        cache.insert(id(7), 100).unwrap();
        assert!(!cache.contains(&id(9)));
        assert!(cache.contains(&id(4)));
        assert!(cache.contains(&id(7)));
    }

    #[test]
    fn lfu_prefers_frequency_then_recency() {
        let mut cache = SimCache::new(PolicyKind::Lfu, 200).unwrap();
        cache.insert(id(1), 100).unwrap();
        cache.insert(id(2), 100).unwrap();
        cache.get(&id(2));
        cache.get(&id(2));
        let outcome = cache.insert(id(3), 100).unwrap();
        assert_eq!(outcome.evictions, 1);
        assert!(!cache.contains(&id(1)));
        assert!(cache.contains(&id(2)));
    }

    #[test]
    fn least_stale_protects_current_sequence() {
        let mut cache = SimCache::new(PolicyKind::LeastStale, 200).unwrap();
        cache.insert(id(1), 100).unwrap();
        cache.insert(id(2), 100).unwrap();
        cache.get(&id(1));
        let outcome = cache.insert(id(3), 100).unwrap();
        assert_eq!(outcome.evictions, 1);
        assert!(!cache.contains(&id(2)));
        assert!(cache.contains(&id(1)));
    }

    #[test]
    fn pinned_refuses_beyond_capacity() {
        let mut cache = SimCache::new(PolicyKind::Pinned, 100).unwrap();
        assert!(cache.insert(id(1), 100).unwrap().admitted);
        let outcome = cache.insert(id(2), 100).unwrap();
        assert!(!outcome.admitted);
        assert_eq!(cache.counters().refusals, 1);
        assert!(cache.contains(&id(1)));
    }

    #[test]
    fn pinned_entries_are_never_evicted() {
        let mut cache = SimCache::new(PolicyKind::Lru, 200).unwrap();
        cache.insert(id(1), 100).unwrap();
        cache.insert(id(2), 100).unwrap();
        assert!(cache.pin(&id(1)));
        assert!(cache.get(&id(2)));
        // id(1) is pinned, so id(2) is the only victim.
        let outcome = cache.insert(id(3), 100).unwrap();
        assert_eq!(outcome.evictions, 1);
        assert!(cache.contains(&id(1)));
        assert!(cache.contains(&id(3)));
    }

    #[test]
    fn oversized_shard_is_refused() {
        let mut cache = SimCache::new(PolicyKind::Lru, 100).unwrap();
        let outcome = cache.insert(id(1), 101).unwrap();
        assert!(!outcome.admitted);
        assert_eq!(cache.counters().refusals, 1);
    }

    #[test]
    fn zero_capacity_policy_cache_is_rejected() {
        assert!(SimCache::new(PolicyKind::Lru, 0).is_err());
        assert!(SimCache::new(PolicyKind::None, 0).is_ok());
    }
}
