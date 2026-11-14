//! Selection decisions and the exact/approximate distinction.
//!
//! Exact mode is not a string flag: [`ExactSelection`] can only be produced by
//! attesting that the selected shard set equals the native computation set, or
//! by an explicit, documented reclassification constructor. A selector that
//! predicts a subset cannot accidentally produce an "exact" decision.

use core::fmt;

use serde::{Deserialize, Serialize};

use crate::error::{CoreError, Result};
use crate::ids::ShardId;

/// Whether omission of any shard may change model semantics.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SelectionMode {
    /// The selected set is the native computation set; caching and prefetching
    /// may change latency but never the operators executed.
    Exact,
    /// A selector chose a subset or substitute; outputs may differ and must be
    /// evaluated and labelled separately.
    Approximate,
}

impl fmt::Display for SelectionMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Exact => f.write_str("exact"),
            Self::Approximate => f.write_str("approximate"),
        }
    }
}

/// One shard's predicted selector score.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ShardScore {
    /// Scored shard.
    pub shard: ShardId,
    /// Selector score (higher means more preferred). Serialized as a float.
    pub score: f64,
}

/// Provenance attached to every selection decision.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SelectionProvenance {
    /// Stable identifier of the selector that produced the decision.
    pub selector_id: String,
    /// Exact or approximate mode.
    pub mode: SelectionMode,
    /// Predicted scores, in the selector's own order.
    pub scores: Vec<ShardScore>,
    /// Shards actually used, when this is recorded post-execution.
    pub realized: Option<Vec<ShardId>>,
    /// Justification recorded when an approximate decision was deliberately
    /// reclassified as exact (for example after external verification).
    pub reclassified_exact: Option<String>,
}

impl SelectionProvenance {
    /// Construct empty provenance.
    ///
    /// # Errors
    ///
    /// Rejects empty selector ids and empty reclassification justifications.
    pub fn new(
        selector_id: impl Into<String>,
        mode: SelectionMode,
        scores: Vec<ShardScore>,
    ) -> Result<Self> {
        let selector_id = selector_id.into();
        if selector_id.is_empty() {
            return Err(CoreError::invalid("selector_id", "selector id is empty"));
        }
        if selector_id.len() > 256 {
            return Err(CoreError::invalid(
                "selector_id",
                "selector id longer than 256 bytes",
            ));
        }
        Ok(Self {
            selector_id,
            mode,
            scores,
            realized: None,
            reclassified_exact: None,
        })
    }

    /// Record the realized set after execution.
    pub fn set_realized(&mut self, realized: Vec<ShardId>) -> Result<()> {
        if realized.is_empty() {
            return Err(CoreError::invalid("realized", "realized set is empty"));
        }
        self.realized = Some(realized);
        Ok(())
    }
}

/// A set of shards selected for one step, in canonical order.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SelectionDecision {
    shards: Vec<ShardId>,
    provenance: SelectionProvenance,
}

impl SelectionDecision {
    /// Construct a decision from a shard set.
    ///
    /// Shards are sorted and must be unique; a duplicate indicates a selector
    /// bug and is rejected rather than silently deduplicated.
    ///
    /// # Errors
    ///
    /// Rejects empty sets and duplicate shard ids.
    pub fn new(mut shards: Vec<ShardId>, provenance: SelectionProvenance) -> Result<Self> {
        if shards.is_empty() {
            return Err(CoreError::invalid("selection", "selected set is empty"));
        }
        shards.sort_unstable();
        for pair in shards.windows(2) {
            if pair[0] == pair[1] {
                return Err(CoreError::invalid(
                    "selection",
                    format!("duplicate shard {}", pair[0]),
                ));
            }
        }
        Ok(Self { shards, provenance })
    }

    /// The native (unmodified) computation set for a step.
    ///
    /// # Errors
    ///
    /// Rejects empty sets and duplicates through [`SelectionDecision::new`].
    pub fn native(shards: Vec<ShardId>) -> Result<Self> {
        let provenance = SelectionProvenance::new("native", SelectionMode::Exact, Vec::new())?;
        Self::new(shards, provenance)
    }

    /// Selected shards, sorted ascending.
    #[must_use]
    pub fn shards(&self) -> &[ShardId] {
        &self.shards
    }

    /// Provenance.
    #[must_use]
    pub fn provenance(&self) -> &SelectionProvenance {
        &self.provenance
    }

    /// Mutable provenance, for recording the realized set.
    pub fn provenance_mut(&mut self) -> &mut SelectionProvenance {
        &mut self.provenance
    }

    /// Number of selected shards.
    #[must_use]
    pub fn len(&self) -> usize {
        self.shards.len()
    }

    /// Always `false`: decisions are never empty.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        false
    }

    /// Whether a shard is selected.
    #[must_use]
    pub fn contains(&self, shard: &ShardId) -> bool {
        self.shards.binary_search(shard).is_ok()
    }

    /// Selection mode.
    #[must_use]
    pub fn mode(&self) -> SelectionMode {
        self.provenance.mode
    }

    /// Whether the recorded realized set equals the selected set.
    ///
    /// Returns `None` until the realized set has been recorded.
    #[must_use]
    pub fn realized_matches(&self) -> Option<bool> {
        self.provenance.realized.as_ref().map(|realized| {
            let mut realized = realized.clone();
            realized.sort_unstable();
            realized.dedup();
            realized == self.shards
        })
    }
}

/// Proof that a decision's selected set equals the native computation set.
///
/// The fields are private. The only ways to obtain a value are
/// [`NativeSetAttestation::attest`], which compares the sets and requires exact
/// mode, and [`ExactSelection::reclassify_from_approximate`], which records an
/// explicit justification.
#[derive(Clone, Debug, PartialEq)]
pub struct NativeSetAttestation {
    shard_count: usize,
    justification: Option<String>,
}

impl NativeSetAttestation {
    /// Attest that `decision` selects exactly `native_set`.
    ///
    /// # Errors
    ///
    /// Fails when the decision is approximate (an approximate decision must be
    /// explicitly reclassified) or the sets differ.
    pub fn attest(decision: &SelectionDecision, native_set: &[ShardId]) -> Result<Self> {
        if decision.mode() != SelectionMode::Exact {
            return Err(CoreError::invalid(
                "selection",
                "approximate decision cannot be attested as exact; use reclassification",
            ));
        }
        let mut native = native_set.to_vec();
        native.sort_unstable();
        native.dedup();
        if native != decision.shards {
            return Err(CoreError::invalid(
                "selection",
                "selected set does not equal the native computation set",
            ));
        }
        Ok(Self {
            shard_count: decision.shards.len(),
            justification: None,
        })
    }

    /// Number of attested shards.
    #[must_use]
    pub const fn shard_count(&self) -> usize {
        self.shard_count
    }

    /// Justification when the attestation came from reclassification.
    #[must_use]
    pub fn justification(&self) -> Option<&str> {
        self.justification.as_deref()
    }
}

/// A decision proven to execute the native computation set.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ExactSelection {
    decision: SelectionDecision,
}

impl ExactSelection {
    /// Build from a decision whose mode is exact and whose set equals the
    /// native set.
    ///
    /// # Errors
    ///
    /// Propagates [`NativeSetAttestation::attest`] failures.
    pub fn from_verified(decision: SelectionDecision, native_set: &[ShardId]) -> Result<Self> {
        let attestation = NativeSetAttestation::attest(&decision, native_set)?;
        debug_assert_eq!(attestation.shard_count, decision.shards.len());
        Ok(Self { decision })
    }

    /// Explicitly reclassify an approximate decision as exact.
    ///
    /// This constructor documents the reclassification in the provenance. It
    /// is the only path from an approximate selector to exact mode, and it must
    /// name why the selected set is believed to be the native set.
    ///
    /// # Errors
    ///
    /// Rejects an empty justification, a decision already exact, and an empty
    /// selected set.
    pub fn reclassify_from_approximate(
        mut decision: SelectionDecision,
        justification: impl Into<String>,
    ) -> Result<Self> {
        let justification = justification.into();
        if justification.is_empty() {
            return Err(CoreError::invalid(
                "reclassification",
                "justification is empty",
            ));
        }
        if decision.mode() == SelectionMode::Exact {
            return Err(CoreError::invalid(
                "reclassification",
                "decision is already exact",
            ));
        }
        if decision.shards.is_empty() {
            return Err(CoreError::invalid("reclassification", "selection is empty"));
        }
        decision.provenance.mode = SelectionMode::Exact;
        decision.provenance.reclassified_exact = Some(justification);
        Ok(Self { decision })
    }

    /// The underlying decision.
    #[must_use]
    pub fn decision(&self) -> &SelectionDecision {
        &self.decision
    }

    /// Selected shards.
    #[must_use]
    pub fn shards(&self) -> &[ShardId] {
        self.decision.shards()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn id(seed: u8) -> ShardId {
        ShardId::from_digest(crate::hash::Sha256Digest::of(&[seed]))
    }

    fn approximate() -> SelectionDecision {
        let provenance =
            SelectionProvenance::new("predictor", SelectionMode::Approximate, Vec::new()).unwrap();
        SelectionDecision::new(vec![id(2), id(1)], provenance).unwrap()
    }

    #[test]
    fn decision_is_sorted_and_unique() {
        let decision = approximate();
        assert_eq!(decision.shards(), &[id(1), id(2)]);
        let provenance =
            SelectionProvenance::new("predictor", SelectionMode::Approximate, Vec::new()).unwrap();
        assert!(SelectionDecision::new(vec![id(1), id(1)], provenance).is_err());
    }

    #[test]
    fn exact_requires_native_set_equality() {
        let decision = SelectionDecision::native(vec![id(1), id(2)]).unwrap();
        assert!(ExactSelection::from_verified(decision.clone(), &[id(2), id(1)]).is_ok());
        assert!(ExactSelection::from_verified(decision, &[id(1)]).is_err());
    }

    #[test]
    fn approximate_cannot_be_attested_without_reclassification() {
        let decision = approximate();
        assert!(NativeSetAttestation::attest(&decision, &[id(1), id(2)]).is_err());
        let exact =
            ExactSelection::reclassify_from_approximate(decision, "verified externally").unwrap();
        assert_eq!(exact.decision().mode(), SelectionMode::Exact);
        assert_eq!(
            exact.decision().provenance().reclassified_exact.as_deref(),
            Some("verified externally")
        );
    }

    #[test]
    fn realized_mismatch_is_detectable() {
        let mut decision = approximate();
        assert_eq!(decision.realized_matches(), None);
        decision.provenance_mut().set_realized(vec![id(2)]).unwrap();
        assert_eq!(decision.realized_matches(), Some(false));
        decision
            .provenance_mut()
            .set_realized(vec![id(1), id(2)])
            .unwrap();
        assert_eq!(decision.realized_matches(), Some(true));
    }
}
