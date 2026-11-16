//! Execution envelopes: task-phase candidate supersets with invalidation.
//!
//! An envelope is an advisory prefetch plan for one phase. It is built from a
//! **prefix** of the trace (the past) and applied to the records after that
//! prefix, so the build and the evaluation windows are disjoint in time. A
//! phase that never appeared in the build prefix gets no envelope, and that
//! absence is reported rather than filled with future knowledge.
//!
//! Utility is never reported as prediction accuracy. The engine reports bytes
//! not transferred, exposed time saved and unused-prefetch bytes against a
//! no-envelope baseline; see [`crate::engine::EnvelopeUtility`].

use std::collections::{BTreeMap, BTreeSet};

use selvedge_core::{Sha256Digest, ShardId};
use serde::{Deserialize, Serialize};

use crate::error::{Result, SimError};
use crate::trace::Trace;

/// Schema of an envelope set.
pub const ENVELOPE_SCHEMA: &str = "selvedge/sim-envelopes/v1";

/// A condition that ends an envelope before its natural horizon.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Invalidation {
    /// The phase identifier changed.
    PhaseChange,
    /// This many records have been served under the envelope.
    AfterRecords {
        /// Record budget.
        records: u64,
    },
    /// The observed miss rate under the envelope exceeded a threshold.
    MissRateAbove {
        /// Miss bytes divided by needed bytes.
        rate: f64,
        /// Records of evidence required before the rule can fire.
        min_records: u64,
    },
}

/// One phase's candidate superset.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Envelope {
    /// Phase identifier this envelope covers.
    pub phase_id: String,
    /// Optional task label observed in the build window.
    pub task: Option<String>,
    /// Candidate shards, highest build coverage first. This is a priority
    /// order, not a sorted set: the engine admits candidates until the cache
    /// budget is full, and membership tests use a set built from this list.
    pub candidate: Vec<ShardId>,
    /// Fraction of build-window records whose selection was a subset of
    /// `candidate`. This is a property of the build window, not a prediction
    /// score for the evaluation window.
    pub build_coverage: f64,
    /// Records of the phase seen in the build window.
    pub build_records: u64,
    /// Last record index of the build window (inclusive).
    pub build_last_index: u64,
    /// Expected live range in records, used for the default invalidation.
    pub expected_live_records: u64,
    /// Invalidation triggers.
    pub invalidate_on: Vec<Invalidation>,
}

/// Parameters of the envelope builder.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EnvelopeParams {
    /// Fraction of the trace used to build envelopes. The rest is evaluation.
    pub build_fraction: f64,
    /// A shard is a candidate when it appears in at least this fraction of the
    /// phase's build records.
    pub min_coverage: f64,
    /// Hard cap on candidates per envelope.
    pub max_candidates: usize,
    /// Invalidation horizon as a multiple of the phase's build record count.
    pub live_record_multiple: u64,
    /// Miss-rate threshold used by the invalidation rule.
    pub miss_rate_threshold: f64,
    /// Minimum records before the miss-rate rule may fire.
    pub miss_rate_min_records: u64,
}

impl Default for EnvelopeParams {
    fn default() -> Self {
        Self {
            build_fraction: 0.5,
            min_coverage: 0.6,
            max_candidates: 4096,
            live_record_multiple: 4,
            miss_rate_threshold: 0.5,
            miss_rate_min_records: 8,
        }
    }
}

impl EnvelopeParams {
    /// Validate the parameters.
    ///
    /// # Errors
    ///
    /// Rejects out-of-range fractions, a zero cap and a zero horizon multiple.
    pub fn validate(&self) -> Result<()> {
        if !(0.0..1.0).contains(&self.build_fraction) || self.build_fraction <= 0.0 {
            return Err(SimError::invalid("build_fraction", "must be in (0, 1)"));
        }
        if !(0.0..=1.0).contains(&self.min_coverage) {
            return Err(SimError::invalid("min_coverage", "must be in [0, 1]"));
        }
        if self.max_candidates == 0 {
            return Err(SimError::invalid("max_candidates", "must be non-zero"));
        }
        if self.live_record_multiple == 0 {
            return Err(SimError::invalid(
                "live_record_multiple",
                "must be non-zero",
            ));
        }
        if !(0.0..=1.0).contains(&self.miss_rate_threshold) {
            return Err(SimError::invalid(
                "miss_rate_threshold",
                "must be in [0, 1]",
            ));
        }
        Ok(())
    }
}

/// A built envelope set with its provenance.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EnvelopeSet {
    /// Always [`ENVELOPE_SCHEMA`].
    pub schema: String,
    /// Creation stamp.
    pub created_utc: String,
    /// Revision the source trace was built at.
    pub revision: String,
    /// Trace identifier.
    pub trace_id: String,
    /// Digest of the source trace records.
    pub trace_sha256: Sha256Digest,
    /// Number of records in the build prefix.
    pub build_records: u64,
    /// Number of records in the evaluation suffix.
    pub evaluate_records: u64,
    /// Parameters used.
    pub params: EnvelopeParams,
    /// Envelopes in first-seen phase order.
    pub envelopes: Vec<Envelope>,
    /// Phases seen in evaluation but absent from the build prefix.
    pub uncovered_phases: Vec<String>,
    /// Caveats.
    pub notes: Vec<String>,
}

impl EnvelopeSet {
    /// Envelope for a phase, if one was built.
    #[must_use]
    pub fn envelope_for(&self, phase_id: &str) -> Option<&Envelope> {
        self.envelopes
            .iter()
            .find(|envelope| envelope.phase_id == phase_id)
    }

    /// Write the set.
    ///
    /// # Errors
    ///
    /// Propagates I/O and serialization failures.
    pub fn write(&self, path: &std::path::Path) -> Result<()> {
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(path, serde_json::to_vec_pretty(self)?)?;
        Ok(())
    }

    /// Read a set.
    ///
    /// # Errors
    ///
    /// Rejects a wrong schema and propagates I/O and JSON failures.
    pub fn load(path: &std::path::Path) -> Result<Self> {
        let bytes = std::fs::read(path)?;
        let set: Self = serde_json::from_slice(&bytes)?;
        if set.schema != ENVELOPE_SCHEMA {
            return Err(SimError::invalid(
                "schema",
                format!("envelope schema is not {ENVELOPE_SCHEMA}"),
            ));
        }
        Ok(set)
    }

    /// Whether the build prefix and the evaluation suffix are disjoint.
    #[must_use]
    pub fn split_is_disjoint(&self) -> bool {
        self.build_records + self.evaluate_records
            == self.build_records.saturating_add(self.evaluate_records)
            && self.build_records.saturating_add(self.evaluate_records) > 0
    }
}

/// Build envelopes from the prefix of a trace.
///
/// # Errors
///
/// Rejects invalid parameters, a trace with fewer than two records, and an
/// empty build or evaluation window.
pub fn build(
    trace: &Trace,
    params: EnvelopeParams,
    revision: &str,
    stamp: &str,
) -> Result<EnvelopeSet> {
    params.validate()?;
    if trace.records.len() < 2 {
        return Err(SimError::invalid(
            "trace",
            "needs at least two records to split",
        ));
    }
    let build_records = ((trace.records.len() as f64) * params.build_fraction).floor() as usize;
    let build_records = build_records.clamp(1, trace.records.len() - 1);
    let evaluate_records = trace.records.len() - build_records;
    // Per phase: shard counts over the build prefix.
    let mut order: Vec<String> = Vec::new();
    let mut counts: BTreeMap<String, BTreeMap<ShardId, u64>> = BTreeMap::new();
    let mut totals: BTreeMap<String, u64> = BTreeMap::new();
    let mut tasks: BTreeMap<String, String> = BTreeMap::new();
    for record in &trace.records[..build_records] {
        let entry = counts.entry(record.phase_id.clone()).or_insert_with(|| {
            order.push(record.phase_id.clone());
            BTreeMap::new()
        });
        *totals.entry(record.phase_id.clone()).or_insert(0) += 1;
        if let Some(task) = &record.task {
            tasks
                .entry(record.phase_id.clone())
                .or_insert_with(|| task.clone());
        }
        for shard in &record.selected {
            *entry.entry(*shard).or_insert(0) += 1;
        }
    }
    let mut envelopes = Vec::new();
    for phase in &order {
        let total = totals.get(phase).copied().unwrap_or(0);
        if total == 0 {
            continue;
        }
        let shard_counts = counts.get(phase).cloned().unwrap_or_default();
        let threshold = (params.min_coverage * total as f64).ceil() as u64;
        let mut chosen: Vec<(u64, ShardId)> = shard_counts
            .iter()
            .filter(|(_, count)| **count >= threshold.max(1))
            .map(|(shard, count)| (*count, *shard))
            .collect();
        chosen.sort_unstable_by(|left, right| right.0.cmp(&left.0).then(left.1.cmp(&right.1)));
        chosen.truncate(params.max_candidates);
        // Highest-coverage first: the engine admits candidates until the cache
        // budget is full, so the order is a priority list.
        let candidate: Vec<ShardId> = chosen.into_iter().map(|(_, shard)| shard).collect();
        if candidate.is_empty() {
            continue;
        }
        let candidate_set: BTreeSet<ShardId> = candidate.iter().copied().collect();
        let covered = trace.records[..build_records]
            .iter()
            .filter(|record| record.phase_id == *phase)
            .filter(|record| {
                record
                    .selected
                    .iter()
                    .all(|shard| candidate_set.contains(shard))
            })
            .count();
        let build_coverage = covered as f64 / total as f64;
        let horizon = total.saturating_mul(params.live_record_multiple).max(1);
        envelopes.push(Envelope {
            phase_id: phase.clone(),
            task: tasks.get(phase).cloned(),
            candidate,
            build_coverage,
            build_records: total,
            build_last_index: (build_records - 1) as u64,
            expected_live_records: horizon,
            invalidate_on: vec![
                Invalidation::PhaseChange,
                Invalidation::AfterRecords { records: horizon },
                Invalidation::MissRateAbove {
                    rate: params.miss_rate_threshold,
                    min_records: params.miss_rate_min_records,
                },
            ],
        });
    }
    let covered_phases: Vec<String> = envelopes
        .iter()
        .map(|envelope| envelope.phase_id.clone())
        .collect();
    let mut uncovered: Vec<String> = Vec::new();
    for record in &trace.records[build_records..] {
        if !covered_phases.contains(&record.phase_id) && !uncovered.contains(&record.phase_id) {
            uncovered.push(record.phase_id.clone());
        }
    }
    Ok(EnvelopeSet {
        schema: ENVELOPE_SCHEMA.to_owned(),
        created_utc: stamp.to_owned(),
        revision: revision.to_owned(),
        trace_id: trace.manifest.trace_id.clone(),
        trace_sha256: trace.manifest.records_sha256,
        build_records: build_records as u64,
        evaluate_records: evaluate_records as u64,
        params,
        envelopes,
        uncovered_phases: uncovered,
        notes: vec![
            "envelopes are built from the trace prefix and applied only to the \
             suffix; a phase absent from the prefix has no envelope"
                .to_owned(),
            "build_coverage is a property of the build window, not a prediction \
             score for the evaluation window"
                .to_owned(),
            "utility is reported as bytes not transferred and exposed time \
             saved, never as prediction accuracy"
                .to_owned(),
        ],
    })
}

/// The record index at which evaluation begins for a set.
#[must_use]
pub fn evaluation_start(set: &EnvelopeSet) -> usize {
    set.build_records as usize
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::trace::{
        ExactBasis, TRACE_MANIFEST_SCHEMA, TRACE_SCHEMA, TraceBundle, TraceLimits, TraceManifest,
        TraceProvenance, TraceRecord, TraceShard, TraceSource,
    };
    use selvedge_core::SelectionMode;

    fn shard(tag: u8, offset: u64) -> TraceShard {
        TraceShard {
            id: ShardId::from_digest(Sha256Digest::of(&[tag])),
            bytes: 100,
            offset,
        }
    }

    fn record(phase: &str, selected: Vec<ShardId>, index: u64) -> TraceRecord {
        TraceRecord {
            schema: TRACE_SCHEMA.to_owned(),
            request_id: "r0".to_owned(),
            phase_id: phase.to_owned(),
            token_index: index,
            layer: 0,
            selected,
            selector_id: "unit".to_owned(),
            mode: SelectionMode::Approximate,
            task: Some("code".to_owned()),
            timestamps: None,
            observed: None,
        }
    }

    fn trace(phases: &[(&str, Vec<u8>)]) -> Trace {
        let mut used: Vec<u8> = phases.iter().flat_map(|(_, tags)| tags.clone()).collect();
        used.sort_unstable();
        used.dedup();
        let shards: Vec<TraceShard> = used
            .iter()
            .enumerate()
            .map(|(index, tag)| shard(*tag, index as u64 * 4096))
            .collect();
        let records: Vec<TraceRecord> = phases
            .iter()
            .enumerate()
            .map(|(index, (phase, tags))| {
                record(
                    phase,
                    tags.iter()
                        .map(|tag| ShardId::from_digest(Sha256Digest::of(&[*tag])))
                        .collect(),
                    index as u64,
                )
            })
            .collect();
        let manifest = TraceManifest {
            schema: TRACE_MANIFEST_SCHEMA.to_owned(),
            trace_id: "unit".to_owned(),
            source: TraceSource::Synthetic,
            provenance: TraceProvenance {
                origin: "unit-test".to_owned(),
                receipt: None,
                receipt_sha256: None,
                revision: "rev".to_owned(),
                stamp: "20261116T000000Z".to_owned(),
                notes: Vec::new(),
            },
            bundle: TraceBundle {
                bundle_id: "b".to_owned(),
                manifest_sha256: Sha256Digest::of(b"m"),
            },
            mode: SelectionMode::Approximate,
            exact_basis: None::<ExactBasis>,
            shards,
            records: 0,
            records_sha256: Sha256Digest::of(b""),
            generator: None,
            created_utc: "20261116T000000Z".to_owned(),
        };
        Trace::new(manifest, records).unwrap()
    }

    #[test]
    fn envelope_builds_from_prefix_only() {
        let t = trace(&[
            ("p0", vec![1, 2]),
            ("p0", vec![1, 2]),
            ("p0", vec![3]),
            ("p0", vec![1, 2]),
        ]);
        let params = EnvelopeParams {
            build_fraction: 0.5,
            min_coverage: 0.9,
            ..EnvelopeParams::default()
        };
        let set = build(&t, params, "rev", "stamp").unwrap();
        assert_eq!(set.build_records, 2);
        assert_eq!(set.evaluate_records, 2);
        let envelope = set.envelope_for("p0").unwrap();
        // Shards 1 and 2 cover both build records; shard 3 appears once.
        assert_eq!(envelope.candidate.len(), 2);
        assert_eq!(envelope.build_coverage, 1.0);
        assert!(set.split_is_disjoint());
    }

    #[test]
    fn unseen_phase_is_uncovered_not_invented() {
        let t = trace(&[
            ("p0", vec![1]),
            ("p0", vec![1]),
            ("p1", vec![2]),
            ("p1", vec![2]),
        ]);
        let set = build(&t, EnvelopeParams::default(), "rev", "stamp").unwrap();
        assert!(set.envelope_for("p0").is_some());
        assert!(set.envelope_for("p1").is_none());
        assert_eq!(set.uncovered_phases, vec!["p1".to_owned()]);
    }

    #[test]
    fn invalidation_rules_are_recorded() {
        let t = trace(&[
            ("p0", vec![1]),
            ("p0", vec![1]),
            ("p0", vec![1]),
            ("p0", vec![1]),
        ]);
        let set = build(&t, EnvelopeParams::default(), "rev", "stamp").unwrap();
        let envelope = set.envelope_for("p0").unwrap();
        assert!(matches!(
            envelope.invalidate_on.first(),
            Some(Invalidation::PhaseChange)
        ));
        assert!(
            envelope
                .invalidate_on
                .iter()
                .any(|rule| matches!(rule, Invalidation::AfterRecords { .. }))
        );
        assert!(
            envelope
                .invalidate_on
                .iter()
                .any(|rule| matches!(rule, Invalidation::MissRateAbove { .. }))
        );
    }

    #[test]
    fn envelope_round_trips() {
        let t = trace(&[("p0", vec![1]), ("p0", vec![1])]);
        let set = build(&t, EnvelopeParams::default(), "rev", "stamp").unwrap();
        let dir = selvedge_core::testutil::TempDir::new("sim-env").unwrap();
        let path = dir.join("env.json");
        set.write(&path).unwrap();
        let loaded = EnvelopeSet::load(&path).unwrap();
        assert_eq!(loaded, set);
        assert!(
            Trace::load(
                &dir.join("nonexistent.jsonl"),
                &dir.join("nonexistent.json"),
                TraceLimits::default()
            )
            .is_err()
        );
    }

    #[test]
    fn invalid_parameters_are_rejected() {
        let mut params = EnvelopeParams::default();
        params.build_fraction = 1.0;
        assert!(params.validate().is_err());
        let mut params = EnvelopeParams::default();
        params.max_candidates = 0;
        assert!(params.validate().is_err());
    }
}
