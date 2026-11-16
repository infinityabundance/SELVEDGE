//! Synthetic trace generation for controlled studies.
//!
//! The generator exists because no trained-router trace exists yet. It makes the
//! three effects the storage and envelope work needs to isolate:
//!
//! - **hot/cold skew**: each phase has a hot set drawn from with probability
//!   `hot_share`, and the cold tail is Zipf-weighted;
//! - **phase switches**: consecutive phases use different hot sets, so a policy
//!   that assumed stationarity is measurably wrong;
//! - **chunk unions**: a selected group can drag in its adjacent groups, which
//!   is what makes coalescing worth anything in a real layout.
//!
//! Every synthetic trace is marked `source: synthetic` in its manifest and
//! carries the generator parameters, so no reader can mistake it for a
//! measurement.

use selvedge_core::{SelectionMode, Sha256Digest, ShardId};
use serde::{Deserialize, Serialize};

use crate::error::{Result, SimError};
use crate::prng::SplitMix64;
use crate::trace::{
    TRACE_MANIFEST_SCHEMA, TRACE_SCHEMA, Trace, TraceBundle, TraceManifest, TraceProvenance,
    TraceRecord, TraceShard, TraceSource,
};

/// Cap on generated records.
pub const MAX_SYNTHETIC_RECORDS: u64 = 1_000_000;

/// Parameters of a synthetic trace.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SyntheticSpec {
    /// Decoder layers.
    pub layers: u32,
    /// Shards per layer (one per group).
    pub groups_per_layer: u32,
    /// Bytes per shard.
    pub shard_bytes: u64,
    /// Requests to generate.
    pub requests: u64,
    /// Tokens per request.
    pub tokens_per_request: u64,
    /// Groups selected per (token, layer).
    pub groups_per_token: u32,
    /// Number of distinct phases.
    pub phases: u32,
    /// Switch phase every request instead of in contiguous blocks.
    pub alternate_phases: bool,
    /// Probability that a selection comes from the phase hot set.
    pub hot_share: f64,
    /// Hot-set size per layer.
    pub hot_size: u32,
    /// Zipf exponent for the cold tail.
    pub skew: f64,
    /// Merge each selected group with this many adjacent groups.
    pub chunk_union: u32,
    /// Task labels, assigned round-robin per request.
    pub task_labels: Vec<String>,
    /// Generator seed.
    pub seed: u64,
}

impl Default for SyntheticSpec {
    fn default() -> Self {
        Self {
            layers: 28,
            groups_per_layer: 16,
            shard_bytes: 786_432,
            requests: 8,
            tokens_per_request: 16,
            groups_per_token: 5,
            phases: 2,
            alternate_phases: false,
            hot_share: 0.8,
            hot_size: 4,
            skew: 1.0,
            chunk_union: 1,
            task_labels: vec!["code".to_owned(), "math".to_owned()],
            seed: 0x5E1D_E6E5_11ED_5EED,
        }
    }
}

impl SyntheticSpec {
    /// A stationary hot-set workload.
    #[must_use]
    pub fn hot() -> Self {
        Self {
            phases: 1,
            hot_share: 0.95,
            ..Self::default()
        }
    }

    /// A cold, uniform workload.
    #[must_use]
    pub fn cold() -> Self {
        Self {
            phases: 1,
            hot_share: 0.0,
            skew: 0.0,
            ..Self::default()
        }
    }

    /// A workload that alternates two phases every request.
    #[must_use]
    pub fn alternating() -> Self {
        Self {
            phases: 2,
            alternate_phases: true,
            requests: 16,
            ..Self::default()
        }
    }

    /// Validate the specification.
    ///
    /// # Errors
    ///
    /// Rejects zero dimensions, out-of-range fractions, a hot set larger than a
    /// layer, a selected set larger than a layer and an over-limit record count.
    pub fn validate(&self) -> Result<()> {
        for (name, value) in [
            ("layers", u64::from(self.layers)),
            ("groups_per_layer", u64::from(self.groups_per_layer)),
            ("requests", self.requests),
            ("tokens_per_request", self.tokens_per_request),
            ("groups_per_token", u64::from(self.groups_per_token)),
            ("phases", u64::from(self.phases)),
            ("shard_bytes", self.shard_bytes),
        ] {
            if value == 0 {
                return Err(SimError::invalid("spec", format!("{name} is zero")));
            }
        }
        if self.groups_per_token > self.groups_per_layer {
            return Err(SimError::invalid(
                "groups_per_token",
                "exceeds the groups in a layer",
            ));
        }
        if self.hot_size > self.groups_per_layer {
            return Err(SimError::invalid(
                "hot_size",
                "exceeds the groups in a layer",
            ));
        }
        if !(0.0..=1.0).contains(&self.hot_share) {
            return Err(SimError::invalid("hot_share", "must be in [0, 1]"));
        }
        if self.skew < 0.0 {
            return Err(SimError::invalid("skew", "must be non-negative"));
        }
        let records = self
            .requests
            .checked_mul(self.tokens_per_request)
            .and_then(|value| value.checked_mul(u64::from(self.layers)))
            .ok_or_else(|| SimError::invalid("spec", "record count overflows"))?;
        if records > MAX_SYNTHETIC_RECORDS {
            return Err(SimError::limit(
                "synthetic_records",
                records,
                MAX_SYNTHETIC_RECORDS,
            ));
        }
        Ok(())
    }

    /// Shards implied by the specification.
    #[must_use]
    pub fn shard_count(&self) -> u64 {
        u64::from(self.layers) * u64::from(self.groups_per_layer)
    }
}

/// A generated trace plus the spec that produced it.
#[derive(Clone, Debug)]
pub struct SyntheticTrace {
    /// The trace.
    pub trace: Trace,
    /// The spec, duplicated for convenience.
    pub spec: SyntheticSpec,
}

/// Generate a synthetic trace.
///
/// # Errors
///
/// Rejects an invalid specification and propagates trace-validation failures.
pub fn generate(spec: SyntheticSpec, revision: &str, stamp: &str) -> Result<SyntheticTrace> {
    spec.validate()?;
    let mut shards = Vec::with_capacity(spec.shard_count() as usize);
    let mut ids: Vec<Vec<ShardId>> = Vec::with_capacity(spec.layers as usize);
    for layer in 0..spec.layers {
        let mut layer_ids = Vec::with_capacity(spec.groups_per_layer as usize);
        for group in 0..spec.groups_per_layer {
            let index = u64::from(layer) * u64::from(spec.groups_per_layer) + u64::from(group);
            let mut tag = Vec::with_capacity(24);
            tag.extend_from_slice(b"selvedge-synthetic");
            tag.extend_from_slice(&spec.seed.to_le_bytes());
            tag.extend_from_slice(&index.to_le_bytes());
            let id = ShardId::from_digest(Sha256Digest::of(&tag));
            shards.push(TraceShard {
                id,
                bytes: spec.shard_bytes,
                offset: index.saturating_mul(spec.shard_bytes),
            });
            layer_ids.push(id);
        }
        ids.push(layer_ids);
    }
    let mut rng = SplitMix64::new(spec.seed);
    // One hot set per phase, per layer.
    let mut hot: Vec<Vec<Vec<ShardId>>> = Vec::with_capacity(spec.phases as usize);
    for _ in 0..spec.phases {
        let mut per_layer = Vec::with_capacity(spec.layers as usize);
        for layer_ids in &ids {
            let mut pool: Vec<ShardId> = layer_ids.clone();
            let mut chosen = Vec::with_capacity(spec.hot_size as usize);
            let take = (spec.hot_size as usize).min(pool.len());
            for _ in 0..take {
                let index = rng.below(pool.len() as u64)? as usize;
                chosen.push(pool.remove(index));
            }
            per_layer.push(chosen);
        }
        hot.push(per_layer);
    }
    let mut records = Vec::new();
    for request in 0..spec.requests {
        let phase = if spec.phases == 1 || spec.alternate_phases {
            (request % u64::from(spec.phases)) as usize
        } else {
            ((request * u64::from(spec.phases)) / spec.requests).min(u64::from(spec.phases) - 1)
                as usize
        };
        let task = if spec.task_labels.is_empty() {
            None
        } else {
            Some(spec.task_labels[(request % spec.task_labels.len() as u64) as usize].clone())
        };
        for token in 0..spec.tokens_per_request {
            for layer in 0..spec.layers {
                let layer_ids = &ids[layer as usize];
                let hot_layer = &hot[phase][layer as usize];
                let mut selected: Vec<ShardId> = Vec::new();
                for _ in 0..spec.groups_per_token {
                    let from_hot = !hot_layer.is_empty() && rng.next_f64() < spec.hot_share;
                    let pick = if from_hot {
                        let index = rng.below(hot_layer.len() as u64)? as usize;
                        hot_layer[index]
                    } else {
                        let mut weights = Vec::with_capacity(layer_ids.len());
                        for rank in 0..layer_ids.len() {
                            weights.push(1.0 / ((rank + 1) as f64).powf(spec.skew));
                        }
                        weighted_pick(&mut rng, layer_ids, &weights)?
                    };
                    let position = layer_ids.iter().position(|id| *id == pick).unwrap_or(0);
                    let span = spec.chunk_union.max(1) as usize;
                    let start = position.saturating_sub(span / 2);
                    for offset in 0..span {
                        if let Some(id) = layer_ids.get(start + offset) {
                            selected.push(*id);
                        }
                    }
                }
                selected.sort_unstable();
                selected.dedup();
                records.push(TraceRecord {
                    schema: TRACE_SCHEMA.to_owned(),
                    request_id: format!("synth-r{request:04}"),
                    phase_id: format!("synth-p{phase}"),
                    token_index: token,
                    layer,
                    selected,
                    selector_id: "synthetic-skew".to_owned(),
                    mode: SelectionMode::Approximate,
                    task: task.clone(),
                    timestamps: None,
                    observed: None,
                });
            }
        }
    }
    let generator = serde_json::to_value(&spec)?;
    let manifest = TraceManifest {
        schema: TRACE_MANIFEST_SCHEMA.to_owned(),
        trace_id: format!("synthetic-{:016x}", spec.seed),
        source: TraceSource::Synthetic,
        provenance: TraceProvenance {
            origin: "selvedge trace export --generator".to_owned(),
            receipt: None,
            receipt_sha256: None,
            revision: revision.to_owned(),
            stamp: stamp.to_owned(),
            notes: vec![
                "synthetic workload: not a measurement and not a trained-router trace".to_owned(),
                "hot/cold skew, phase switching and chunk unions are controlled by the spec"
                    .to_owned(),
            ],
        },
        bundle: TraceBundle {
            bundle_id: "synthetic".to_owned(),
            manifest_sha256: Sha256Digest::of(b"selvedge-synthetic-bundle"),
        },
        mode: SelectionMode::Approximate,
        exact_basis: None,
        shards,
        records: 0,
        records_sha256: Sha256Digest::of(b""),
        generator: Some(generator),
        created_utc: stamp.to_owned(),
    };
    let trace = Trace::new(manifest, records)?;
    Ok(SyntheticTrace { trace, spec })
}

fn weighted_pick(rng: &mut SplitMix64, ids: &[ShardId], weights: &[f64]) -> Result<ShardId> {
    let total: f64 = weights.iter().sum();
    if total <= 0.0 {
        return Ok(ids[0]);
    }
    let mut point = rng.next_f64() * total;
    for (id, weight) in ids.iter().zip(weights) {
        point -= weight;
        if point <= 0.0 {
            return Ok(*id);
        }
    }
    ids.last()
        .copied()
        .ok_or_else(|| SimError::invalid("weights", "no candidates"))
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn generator_is_deterministic_and_hash_stable() {
        let spec = SyntheticSpec::default();
        let a = generate(spec.clone(), "rev", "stamp").unwrap();
        let b = generate(spec, "rev", "stamp").unwrap();
        assert_eq!(
            a.trace.manifest.records_sha256,
            b.trace.manifest.records_sha256
        );
        assert_eq!(a.trace.records, b.trace.records);
        assert_eq!(a.trace.manifest.source, TraceSource::Synthetic);
    }

    #[test]
    fn different_seeds_differ() {
        let mut spec = SyntheticSpec::default();
        let a = generate(spec.clone(), "rev", "stamp").unwrap();
        spec.seed ^= 0xFFFF;
        let b = generate(spec, "rev", "stamp").unwrap();
        assert_ne!(
            a.trace.manifest.records_sha256,
            b.trace.manifest.records_sha256
        );
    }

    #[test]
    fn phase_switch_changes_the_hot_set() {
        let mut spec = SyntheticSpec::alternating();
        spec.requests = 4;
        spec.tokens_per_request = 1;
        spec.layers = 2;
        spec.chunk_union = 1;
        let generated = generate(spec, "rev", "stamp").unwrap();
        let mut phase0: Vec<ShardId> = Vec::new();
        let mut phase1: Vec<ShardId> = Vec::new();
        for record in &generated.trace.records {
            if record.phase_id == "synth-p0" {
                phase0.extend(record.selected.iter().copied());
            } else {
                phase1.extend(record.selected.iter().copied());
            }
        }
        phase0.sort_unstable();
        phase0.dedup();
        phase1.sort_unstable();
        phase1.dedup();
        assert!(!phase0.is_empty() && !phase1.is_empty());
        assert_ne!(phase0, phase1);
    }

    #[test]
    fn chunk_union_creates_physical_adjacency() {
        let spec = SyntheticSpec {
            requests: 1,
            tokens_per_request: 1,
            layers: 1,
            groups_per_layer: 16,
            groups_per_token: 1,
            hot_share: 0.0,
            chunk_union: 3,
            ..SyntheticSpec::default()
        };
        let generated = generate(spec.clone(), "rev", "stamp").unwrap();
        let record = &generated.trace.records[0];
        assert!(record.selected.len() >= 2);
        let offsets: Vec<u64> = record
            .selected
            .iter()
            .map(|id| {
                generated
                    .trace
                    .shards()
                    .iter()
                    .find(|shard| shard.id == *id)
                    .map(|shard| shard.offset)
                    .unwrap_or(0)
            })
            .collect();
        let adjacent = offsets
            .windows(2)
            .filter(|pair| pair[1] - pair[0] == spec.shard_bytes)
            .count();
        assert!(adjacent >= 1);
    }

    #[test]
    fn invalid_specs_are_rejected() {
        let spec = SyntheticSpec {
            groups_per_token: 100,
            ..SyntheticSpec::default()
        };
        assert!(spec.validate().is_err());
        let spec = SyntheticSpec {
            hot_share: 2.0,
            ..SyntheticSpec::default()
        };
        assert!(spec.validate().is_err());
        let spec = SyntheticSpec {
            requests: MAX_SYNTHETIC_RECORDS,
            ..SyntheticSpec::default()
        };
        assert!(spec.validate().is_err());
    }
}
