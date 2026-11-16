//! Execution residency: decoded f32 weights, distinct from the storage cache.
//!
//! The store's cache holds **verified bytes** exactly as they appear in the
//! bundle. The kernels cannot consume those bytes: they need `f32` in execution
//! layout, and converting little-endian bytes into `[f32]` is work that a
//! byte-cache hit does not avoid. Before this module existed every shard read —
//! hit or miss — decoded its bytes into a temporary `Vec<f32>` that was thrown
//! away at the end of the group, so a "warm" run still paid 1,344 decodes per
//! token on the pinned 0.6B control.
//!
//! This module is the second residency tier: **decoded, execution-ready
//! weights**, kept per `(version, dtype, shard id)` under its own byte budget
//! and its own eviction policy. The two tiers answer different questions:
//!
//! | tier | question | unit |
//! | --- | --- | --- |
//! | [`selvedge_store::ShardStore`] cache | are this shard's verified bytes in RAM? | bundle bytes |
//! | [`DecodeResidency`] | are this shard's values already `f32` in execution order? | decoded bytes |
//!
//! A hit in this tier skips the store entirely (no read, no verification copy,
//! no decode). A miss still goes through the store, so the storage measurements
//! are unchanged when the residency is empty.
//!
//! # Exactness
//!
//! The residency stores exactly what [`crate::bundle`]'s decoder would produce:
//! `f32::from_le_bytes` per four-byte chunk, in the same order. A residency hit
//! and a residency miss therefore hand the kernels **the same bits**, which
//! `tests/p14_parity.rs::decoded_residency_is_bit_identical` asserts on the
//! committed tiny fixture and the Court receipt asserts on the real checkpoint.
//!
//! # Eviction
//!
//! Least recently used, ties broken by shard id, so eviction is deterministic
//! and does not depend on hash iteration order or thread scheduling. An entry
//! larger than the whole budget is not admitted (and is counted as `bypassed`)
//! rather than being allowed to blow the cap: every allocation stays inside the
//! configured byte budget.

use std::collections::BTreeMap;
use std::sync::Arc;

use selvedge_core::{Sha256Digest, ShardId};
use serde::Serialize;

/// The dtype of a decoded shard.
///
/// Only `f32` exists today; the field exists so the residency key names the
/// representation it holds, and so a quantized path cannot silently alias an
/// f32 entry. It is part of the key, not a comment.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DecodeDtype {
    /// Little-endian f32, the only dtype the bundle format publishes today.
    F32,
}

impl DecodeDtype {
    /// Bytes per element.
    #[must_use]
    pub const fn element_bytes(self) -> u64 {
        match self {
            Self::F32 => 4,
        }
    }

    /// Stable label for receipts.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::F32 => "f32",
        }
    }
}

/// Identity of one decoded entry: `(bundle version, dtype, shard id)`.
///
/// `version` is the validated manifest digest, so a different bundle (or a
/// re-publication of the same id) can never serve a stale decoded entry, and
/// `dtype` means a future quantized representation cannot alias this one.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct DecodeKey {
    /// Validated manifest digest of the bundle the shard came from.
    pub version: Sha256Digest,
    /// Execution representation.
    pub dtype: DecodeDtype,
    /// Content-addressed shard identity.
    pub shard: ShardId,
}

impl DecodeKey {
    /// Build a key.
    #[must_use]
    pub const fn new(version: Sha256Digest, dtype: DecodeDtype, shard: ShardId) -> Self {
        Self {
            version,
            dtype,
            shard,
        }
    }
}

/// Counters for one residency, cumulative over the run.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
pub struct ResidencyCounters {
    /// Lookups served from the residency (no decode, no store read).
    pub hits: u64,
    /// Lookups that had to decode.
    pub misses: u64,
    /// Decode passes performed (equals `misses` for an admitted entry, and also
    /// counts entries too large to admit).
    pub decodes: u64,
    /// Decode passes avoided by a hit.
    pub decode_avoided: u64,
    /// Entries evicted to stay inside the budget.
    pub evictions: u64,
    /// Bytes reclaimed by eviction.
    pub evicted_bytes: u64,
    /// Entries refused because they alone exceeded the budget.
    pub bypassed: u64,
    /// Bytes decoded into the residency over the run.
    pub decoded_bytes: u64,
}

impl ResidencyCounters {
    /// Lookups performed.
    #[must_use]
    pub const fn lookups(&self) -> u64 {
        self.hits.saturating_add(self.misses)
    }

    /// Hit ratio in `[0, 1]`, or zero when nothing was looked up.
    #[must_use]
    pub fn hit_ratio(&self) -> f64 {
        let lookups = self.lookups();
        if lookups == 0 {
            return 0.0;
        }
        self.hits as f64 / lookups as f64
    }
}

/// A residency snapshot for a receipt.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
pub struct ResidencyStats {
    /// Configured byte budget.
    pub capacity_bytes: u64,
    /// Bytes resident now.
    pub used_bytes: u64,
    /// Largest `used_bytes` ever observed.
    pub peak_bytes: u64,
    /// Resident entries.
    pub entries: u64,
    /// Cumulative counters.
    #[serde(flatten)]
    pub counters: ResidencyCounters,
}

#[derive(Debug)]
struct Entry {
    values: Arc<[f32]>,
    bytes: u64,
    last_used_seq: u64,
}

/// Bounded, deterministic decoded-weight residency.
#[derive(Debug)]
pub struct DecodeResidency {
    capacity_bytes: u64,
    used_bytes: u64,
    peak_bytes: u64,
    entries: BTreeMap<DecodeKey, Entry>,
    seq: u64,
    counters: ResidencyCounters,
}

impl DecodeResidency {
    /// Create a residency with a byte budget. Zero disables it entirely: every
    /// lookup misses, nothing is stored, and the caller behaves exactly as the
    /// pre-P14.1 path did.
    #[must_use]
    pub fn new(capacity_bytes: u64) -> Self {
        Self {
            capacity_bytes,
            used_bytes: 0,
            peak_bytes: 0,
            entries: BTreeMap::new(),
            seq: 0,
            counters: ResidencyCounters::default(),
        }
    }

    /// Whether the residency stores anything.
    #[must_use]
    pub const fn is_enabled(&self) -> bool {
        self.capacity_bytes > 0
    }

    /// Configured byte budget.
    #[must_use]
    pub const fn capacity_bytes(&self) -> u64 {
        self.capacity_bytes
    }

    /// Bytes resident now.
    #[must_use]
    pub const fn used_bytes(&self) -> u64 {
        self.used_bytes
    }

    /// Largest residency observed.
    #[must_use]
    pub const fn peak_bytes(&self) -> u64 {
        self.peak_bytes
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

    /// Cumulative counters.
    #[must_use]
    pub const fn counters(&self) -> ResidencyCounters {
        self.counters
    }

    /// A receipt-shaped snapshot.
    #[must_use]
    pub fn stats(&self) -> ResidencyStats {
        ResidencyStats {
            capacity_bytes: self.capacity_bytes,
            used_bytes: self.used_bytes,
            peak_bytes: self.peak_bytes,
            entries: u64::try_from(self.entries.len()).unwrap_or(u64::MAX),
            counters: self.counters,
        }
    }

    /// Look up a decoded entry, counting the lookup.
    ///
    /// Returns a cheap `Arc` clone; the values are never copied, so a hit costs
    /// no decode and no byte copy.
    pub fn get(&mut self, key: &DecodeKey) -> Option<Arc<[f32]>> {
        let seq = {
            self.seq = self.seq.saturating_add(1);
            self.seq
        };
        match self.entries.get_mut(key) {
            Some(entry) => {
                entry.last_used_seq = seq;
                self.counters.hits = self.counters.hits.saturating_add(1);
                self.counters.decode_avoided = self.counters.decode_avoided.saturating_add(1);
                Some(Arc::clone(&entry.values))
            }
            None => {
                self.counters.misses = self.counters.misses.saturating_add(1);
                None
            }
        }
    }

    /// Record that a decode happened and, when it fits, admit the result.
    ///
    /// Returns the values to hand to the kernels: the admitted `Arc` on success,
    /// or the caller's own `Arc` when the entry is larger than the budget. The
    /// decode is counted either way, so `decodes` is the number of conversion
    /// passes actually performed.
    pub fn insert(&mut self, key: DecodeKey, values: Arc<[f32]>) -> Arc<[f32]> {
        let bytes = (values.len() as u64).saturating_mul(DecodeDtype::F32.element_bytes());
        self.counters.decodes = self.counters.decodes.saturating_add(1);
        self.counters.decoded_bytes = self.counters.decoded_bytes.saturating_add(bytes);
        if !self.is_enabled() || bytes > self.capacity_bytes {
            self.counters.bypassed = self.counters.bypassed.saturating_add(1);
            return values;
        }
        // Replacing an existing entry subtracts its old footprint first, so an
        // identical re-insert can never evict a live entry.
        if let Some(previous) = self.entries.remove(&key) {
            self.used_bytes = self.used_bytes.saturating_sub(previous.bytes);
        }
        while self.used_bytes.saturating_add(bytes) > self.capacity_bytes {
            let Some(victim) = self
                .entries
                .iter()
                .min_by_key(|(key, entry)| (entry.last_used_seq, key.shard))
                .map(|(key, _)| *key)
            else {
                break;
            };
            if let Some(evicted) = self.entries.remove(&victim) {
                self.used_bytes = self.used_bytes.saturating_sub(evicted.bytes);
                self.counters.evictions = self.counters.evictions.saturating_add(1);
                self.counters.evicted_bytes =
                    self.counters.evicted_bytes.saturating_add(evicted.bytes);
            }
        }
        let seq = {
            self.seq = self.seq.saturating_add(1);
            self.seq
        };
        self.used_bytes = self.used_bytes.saturating_add(bytes);
        self.peak_bytes = self.peak_bytes.max(self.used_bytes);
        self.entries.insert(
            key,
            Entry {
                values,
                bytes,
                last_used_seq: seq,
            },
        );
        Arc::clone(&self.entries[&key].values)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use selvedge_core::Sha256Digest;

    fn version() -> Sha256Digest {
        Sha256Digest::of(b"p14.1-test-version")
    }

    fn key(index: u8) -> DecodeKey {
        let mut bytes = [0_u8; 32];
        bytes[0] = index;
        DecodeKey::new(
            version(),
            DecodeDtype::F32,
            ShardId::from_digest(Sha256Digest::from_bytes(bytes)),
        )
    }

    fn values(count: usize, seed: f32) -> Arc<[f32]> {
        Arc::from(
            (0..count)
                .map(|index| seed + index as f32)
                .collect::<Vec<_>>(),
        )
    }

    #[test]
    fn zero_budget_reproduces_the_undecoded_path() {
        let mut residency = DecodeResidency::new(0);
        assert!(!residency.is_enabled());
        let key = key(0);
        let inserted = residency.insert(key, values(4, 1.0));
        assert_eq!(inserted.len(), 4);
        assert!(residency.get(&key).is_none());
        assert_eq!(residency.counters().decodes, 1);
        assert_eq!(residency.counters().bypassed, 1);
        assert_eq!(residency.used_bytes(), 0);
        assert!(residency.is_empty());
    }

    #[test]
    fn a_hit_never_decodes_again_and_returns_the_same_bits() {
        let mut residency = DecodeResidency::new(1 << 20);
        let key = key(1);
        let first = residency.insert(key, values(8, 0.5));
        let second = residency.get(&key).unwrap();
        assert_eq!(
            first.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            second.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
        assert_eq!(residency.counters().decodes, 1);
        assert_eq!(residency.counters().hits, 1);
        assert_eq!(residency.counters().decode_avoided, 1);
        // An insert is not a lookup; one hit out of one lookup.
        assert_eq!(residency.counters().hit_ratio(), 1.0);
        assert!(residency.get(&super::tests::key(7)).is_none());
        assert_eq!(residency.counters().hit_ratio(), 0.5);
    }

    #[test]
    fn eviction_is_lru_with_a_stable_tie_break_and_respects_the_budget() {
        // Room for two 4-element (16 byte) entries.
        let mut residency = DecodeResidency::new(32);
        residency.insert(key(1), values(4, 0.0));
        residency.insert(key(2), values(4, 0.0));
        // Touch 1 so 2 becomes least recently used.
        assert!(residency.get(&key(1)).is_some());
        residency.insert(key(3), values(4, 0.0));
        assert!(
            residency.get(&key(2)).is_none(),
            "the LRU entry was evicted"
        );
        assert!(residency.get(&key(1)).is_some());
        assert!(residency.get(&key(3)).is_some());
        assert!(residency.used_bytes() <= residency.capacity_bytes());
        assert_eq!(residency.peak_bytes(), 32);
        assert_eq!(residency.counters().evictions, 1);
    }

    #[test]
    fn an_entry_larger_than_the_budget_is_bypassed_not_admitted() {
        let mut residency = DecodeResidency::new(16);
        let large = residency.insert(key(9), values(64, 0.0));
        assert_eq!(large.len(), 64);
        assert!(residency.is_empty());
        assert_eq!(residency.counters().bypassed, 1);
        assert_eq!(residency.used_bytes(), 0);
    }
}
