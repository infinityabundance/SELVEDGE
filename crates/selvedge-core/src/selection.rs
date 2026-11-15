//! Selection decisions and the exact/approximate distinction.
//!
//! # Two different claims, two different types
//!
//! * **Scope integrity** is a storage property: the store returns bytes only for
//!   shards inside a supplied scope. It is encoded by
//!   `selvedge_store::ExecutionScope` and proven by a [`crate::selection`] read
//!   that refuses an out-of-scope shard. A correct scoped read says nothing
//!   about whether the scope is the model's native computation.
//! * **Semantic exactness** is a model property: the executed shard set is
//!   provably equal to the model's declared native computation set. It is
//!   encoded by [`ExactSelection`], whose only constructor consumes a
//!   [`NativeSetWitness`] produced by comparing the executed set with a
//!   [`NativeComputationSet`] declared by the model bundle.
//!
//! There is deliberately no constructor that turns a selector's approximate
//! decision into an exact one from a string justification, and no conversion
//! from a store scope to [`ExactSelection`]. A selector that predicts a subset
//! can only ever produce [`SemanticExactness::Approximate`].

use core::fmt;

use serde::{Deserialize, Serialize};

use crate::canonical::CanonicalWriter;
use crate::error::{CoreError, Result};
use crate::hash::Sha256Digest;
use crate::ids::ShardId;

/// Domain tag for the digest of a declared native computation set.
pub const NATIVE_SET_DOMAIN: &str = "selvedge/native-computation-set/v1";

/// Canonical digest of `(bundle_id, sorted unique shard ids)`.
///
/// # Errors
///
/// Propagates a malformed domain tag.
pub fn native_set_digest(bundle_id: &str, shards: &[ShardId]) -> Result<Sha256Digest> {
    let mut writer = CanonicalWriter::new(NATIVE_SET_DOMAIN)?;
    writer.write_str(bundle_id);
    writer.write_u64(shards.len() as u64);
    for shard in shards {
        writer.write_bytes(shard.digest().as_bytes());
    }
    Ok(Sha256Digest::of(&writer.into_bytes()))
}

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
}

impl SelectionProvenance {
    /// Provenance for a selector that predicts a subset.
    ///
    /// A deserialized record can still carry `mode: exact`; that field is a
    /// declaration, never evidence. Only [`ExactSelection`] is evidence.
    ///
    /// # Errors
    ///
    /// Rejects an empty or over-long selector id.
    pub fn approximate(selector_id: impl Into<String>, scores: Vec<ShardScore>) -> Result<Self> {
        Self::build(selector_id, SelectionMode::Approximate, scores)
    }

    /// Provenance for the model's own native set, used by
    /// [`SelectionDecision::native`]. It is crate-private so no selector can
    /// mint an exact-mode decision through the public API.
    pub(crate) fn native_provenance() -> Result<Self> {
        Self::build("native", SelectionMode::Exact, Vec::new())
    }

    fn build(
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
        let provenance = SelectionProvenance::native_provenance()?;
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

/// The native computation set a model bundle declares for one step.
///
/// The bundle names its own full set of shards: this is the reference against
/// which an executed set can be compared. The digest is computed here, from
/// the bundle id and the sorted unique shard ids, and cannot be supplied by a
/// caller. The type is serializable for receipts but deliberately not
/// deserializable, so a receipt cannot be replayed as a declaration.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct NativeComputationSet {
    bundle_id: String,
    shards: Vec<ShardId>,
    digest: Sha256Digest,
}

impl NativeComputationSet {
    /// Declare the native set of `bundle_id`.
    ///
    /// # Errors
    ///
    /// Rejects an empty bundle id, an empty set and duplicate shard ids. The
    /// set is sorted canonically before the digest is computed.
    pub fn declare(
        bundle_id: impl Into<String>,
        shards: impl IntoIterator<Item = ShardId>,
    ) -> Result<Self> {
        let bundle_id = bundle_id.into();
        if bundle_id.is_empty() {
            return Err(CoreError::invalid("bundle_id", "bundle id is empty"));
        }
        if bundle_id.len() > 256 {
            return Err(CoreError::invalid(
                "bundle_id",
                "bundle id longer than 256 bytes",
            ));
        }
        let mut shards: Vec<ShardId> = shards.into_iter().collect();
        if shards.is_empty() {
            return Err(CoreError::invalid(
                "native_set",
                "declared native computation set is empty",
            ));
        }
        shards.sort_unstable();
        for pair in shards.windows(2) {
            if pair[0] == pair[1] {
                return Err(CoreError::invalid(
                    "native_set",
                    format!("duplicate shard {} in declared native set", pair[0]),
                ));
            }
        }
        let digest = native_set_digest(&bundle_id, &shards)?;
        Ok(Self {
            bundle_id,
            shards,
            digest,
        })
    }

    /// Bundle the declaration belongs to.
    #[must_use]
    pub fn bundle_id(&self) -> &str {
        &self.bundle_id
    }

    /// Declared shards, sorted ascending.
    #[must_use]
    pub fn shards(&self) -> &[ShardId] {
        &self.shards
    }

    /// Canonical digest of the declaration.
    #[must_use]
    pub const fn digest(&self) -> Sha256Digest {
        self.digest
    }

    /// Number of declared shards.
    #[must_use]
    pub fn len(&self) -> usize {
        self.shards.len()
    }

    /// Always `false`: a declaration is never empty.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        false
    }
}

/// Proof that an executed shard set was compared, element for element, with a
/// model's declared native computation set.
///
/// The fields are private and the only constructor is
/// [`NativeSetWitness::compare`], which requires full set equality *and* bundle
/// identity. A store `ExecutionScope` cannot produce one: it has no native
/// computation set to compare against. Neither can a selector's decision; the
/// comparison takes the executed set, not the selector's claim.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct NativeSetWitness {
    bundle_id: String,
    declared_digest: Sha256Digest,
    executed_digest: Sha256Digest,
    shard_count: u64,
}

impl NativeSetWitness {
    /// Compare the executed set with the declared native set.
    ///
    /// Equality is required in both directions: every executed shard is in the
    /// declaration and every declared shard was executed. The bundle ids must
    /// match, and the executed set must not contain duplicates.
    ///
    /// # Errors
    ///
    /// Fails on a bundle mismatch, an empty executed set, duplicate executed
    /// shards, or any element that differs between the two sets.
    pub fn compare(
        executed_bundle_id: &str,
        executed: &[ShardId],
        native: &NativeComputationSet,
    ) -> Result<Self> {
        if executed_bundle_id != native.bundle_id() {
            return Err(CoreError::invalid(
                "native_set_witness",
                format!(
                    "executed bundle {executed_bundle_id:?} is not the declared bundle {:?}",
                    native.bundle_id()
                ),
            ));
        }
        if executed.is_empty() {
            return Err(CoreError::invalid(
                "native_set_witness",
                "executed shard set is empty",
            ));
        }
        let mut executed_sorted = executed.to_vec();
        executed_sorted.sort_unstable();
        for pair in executed_sorted.windows(2) {
            if pair[0] == pair[1] {
                return Err(CoreError::invalid(
                    "native_set_witness",
                    format!("duplicate executed shard {}; refusing a witness", pair[0]),
                ));
            }
        }
        if executed_sorted != native.shards() {
            let missing = native
                .shards()
                .iter()
                .filter(|shard| !executed_sorted.contains(shard))
                .count();
            let extra = executed_sorted
                .iter()
                .filter(|shard| !native.shards().contains(shard))
                .count();
            return Err(CoreError::invalid(
                "native_set_witness",
                format!(
                    "executed set is not the native computation set: {} executed, {} declared,
                     {missing} declared shard(s) missing, {extra} executed shard(s) outside the declaration",
                    executed_sorted.len(),
                    native.len()
                ),
            ));
        }
        let executed_digest = native_set_digest(executed_bundle_id, &executed_sorted)?;
        if executed_digest != native.digest() {
            return Err(CoreError::invalid(
                "native_set_witness",
                "executed set digest differs from the declared digest",
            ));
        }
        Ok(Self {
            bundle_id: executed_bundle_id.to_owned(),
            declared_digest: native.digest(),
            executed_digest,
            shard_count: executed_sorted.len() as u64,
        })
    }

    /// Bundle both sets belong to.
    #[must_use]
    pub fn bundle_id(&self) -> &str {
        &self.bundle_id
    }

    /// Digest of the declared native set.
    #[must_use]
    pub const fn declared_digest(&self) -> Sha256Digest {
        self.declared_digest
    }

    /// Digest of the executed set.
    #[must_use]
    pub const fn executed_digest(&self) -> Sha256Digest {
        self.executed_digest
    }

    /// Number of shards proven equal.
    #[must_use]
    pub const fn shard_count(&self) -> u64 {
        self.shard_count
    }
}

/// A decision proven to execute the native computation set.
///
/// # Invariant
///
/// Values of this type are only constructible from a [`NativeSetWitness`]. The
/// witness pins the bundle id, the declared digest and the executed digest, and
/// [`ExactSelection::from_witness`] re-derives the decision's own digest and
/// requires it to equal the witnessed executed digest. There is no string
/// justification, no declared-mode shortcut and no store-scope conversion.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ExactSelection {
    decision: SelectionDecision,
    witness: NativeSetWitness,
}

impl ExactSelection {
    /// Build from a decision and a witness that covers exactly that decision.
    ///
    /// # Errors
    ///
    /// Rejects a witness whose bundle or executed digest does not match the
    /// decision's shard set, and a witness whose declared and executed digests
    /// disagree.
    pub fn from_witness(decision: SelectionDecision, witness: NativeSetWitness) -> Result<Self> {
        if witness.declared_digest() != witness.executed_digest() {
            return Err(CoreError::invalid(
                "exact_selection",
                "witness digests disagree; the sets were not proven equal",
            ));
        }
        let digest = native_set_digest(witness.bundle_id(), decision.shards())?;
        if digest != witness.executed_digest() {
            return Err(CoreError::invalid(
                "exact_selection",
                "decision shard set is not the set the witness compared",
            ));
        }
        if u64::try_from(decision.shards().len()).ok() != Some(witness.shard_count()) {
            return Err(CoreError::invalid(
                "exact_selection",
                "decision shard count does not match the witness",
            ));
        }
        Ok(Self { decision, witness })
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

    /// The witness this selection was built from.
    #[must_use]
    pub fn witness(&self) -> &NativeSetWitness {
        &self.witness
    }
}

/// Why a selection could not be proven to execute the native computation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SemanticGap {
    /// No native computation set was declared, so no comparison is possible.
    NoNativeDeclaration,
    /// The executed set is not element-for-element the declared native set.
    SetMismatch,
    /// The executed set belongs to a different bundle than the declaration.
    BundleMismatch,
}

impl fmt::Display for SemanticGap {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoNativeDeclaration => f.write_str("no_native_declaration"),
            Self::SetMismatch => f.write_str("set_mismatch"),
            Self::BundleMismatch => f.write_str("bundle_mismatch"),
        }
    }
}

/// The semantic exactness verdict for one evaluated execution.
///
/// `Proven` carries the [`ExactSelection`] proof; `Approximate` carries the
/// decision and a named gap. There is no third "reclassified" state.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(tag = "status", rename_all = "snake_case", deny_unknown_fields)]
pub enum SemanticExactness {
    /// The executed set was proven equal to the declared native set.
    Proven {
        /// The proof.
        exact: ExactSelection,
    },
    /// The selection is not the native computation; outputs may differ.
    Approximate {
        /// The unproven decision.
        decision: SelectionDecision,
        /// Why exactness could not be proven.
        gap: SemanticGap,
    },
}

impl SemanticExactness {
    /// Classify a decision by comparing its executed set with a declaration.
    ///
    /// This is the only path from a decision to either variant. A `None`
    /// declaration always yields `Approximate::NoNativeDeclaration`.
    #[must_use]
    pub fn classify(
        decision: SelectionDecision,
        executed_bundle_id: &str,
        native: Option<&NativeComputationSet>,
    ) -> Self {
        let Some(native) = native else {
            return Self::Approximate {
                decision,
                gap: SemanticGap::NoNativeDeclaration,
            };
        };
        if executed_bundle_id != native.bundle_id() {
            return Self::Approximate {
                decision,
                gap: SemanticGap::BundleMismatch,
            };
        }
        match NativeSetWitness::compare(executed_bundle_id, decision.shards(), native) {
            Ok(witness) => match ExactSelection::from_witness(decision.clone(), witness) {
                Ok(exact) => Self::Proven { exact },
                Err(_) => Self::Approximate {
                    decision,
                    gap: SemanticGap::SetMismatch,
                },
            },
            Err(_) => Self::Approximate {
                decision,
                gap: SemanticGap::SetMismatch,
            },
        }
    }

    /// Whether exactness is proven.
    #[must_use]
    pub fn is_proven(&self) -> bool {
        matches!(self, Self::Proven { .. })
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
        let provenance = SelectionProvenance::approximate("predictor", Vec::new()).unwrap();
        SelectionDecision::new(vec![id(2), id(1)], provenance).unwrap()
    }

    #[test]
    fn decision_is_sorted_and_unique() {
        let decision = approximate();
        assert_eq!(decision.shards(), &[id(1), id(2)]);
        let provenance = SelectionProvenance::approximate("predictor", Vec::new()).unwrap();
        assert!(SelectionDecision::new(vec![id(1), id(1)], provenance).is_err());
    }

    #[test]
    fn exact_selection_requires_a_full_set_witness() {
        let native = NativeComputationSet::declare("bundle-a", vec![id(1), id(2)]).unwrap();
        let decision = approximate();
        let witness = NativeSetWitness::compare("bundle-a", decision.shards(), &native).unwrap();
        assert_eq!(witness.shard_count(), 2);
        assert_eq!(witness.declared_digest(), witness.executed_digest());
        let exact = ExactSelection::from_witness(decision.clone(), witness.clone()).unwrap();
        assert_eq!(exact.shards(), decision.shards());

        // A subset is not the native set, even though it is entirely contained
        // in it; there is no approximate-to-exact conversion.
        assert!(NativeSetWitness::compare("bundle-a", &[id(1)], &native).is_err());
        // An extra shard is not the native set either.
        assert!(NativeSetWitness::compare("bundle-a", &[id(1), id(2), id(3)], &native).is_err());
        // A witness from another bundle cannot be attached to this decision.
        let other = NativeComputationSet::declare("bundle-b", vec![id(1), id(2)]).unwrap();
        assert!(NativeSetWitness::compare("bundle-b", &[id(1), id(2)], &native).is_err());
        assert!(NativeSetWitness::compare("bundle-b", &[id(1), id(2)], &other).is_ok());
    }

    #[test]
    fn witness_for_a_different_set_is_rejected_by_exact_selection() {
        let native = NativeComputationSet::declare("bundle-a", vec![id(1), id(2)]).unwrap();
        let witness = NativeSetWitness::compare("bundle-a", &[id(1), id(2)], &native).unwrap();
        let other = SelectionDecision::native(vec![id(1), id(3)]).unwrap();
        assert!(ExactSelection::from_witness(other, witness).is_err());
    }

    #[test]
    fn duplicate_or_empty_executed_sets_cannot_produce_a_witness() {
        let native = NativeComputationSet::declare("bundle-a", vec![id(1), id(2)]).unwrap();
        assert!(NativeSetWitness::compare("bundle-a", &[id(1), id(1)], &native).is_err());
        assert!(NativeSetWitness::compare("bundle-a", &[], &native).is_err());
        assert!(NativeComputationSet::declare("bundle-a", vec![id(1), id(1)]).is_err());
        assert!(NativeComputationSet::declare("", vec![id(1)]).is_err());
    }

    #[test]
    fn semantic_exactness_is_only_proven_by_a_witness() {
        let native = NativeComputationSet::declare("bundle-a", vec![id(1), id(2)]).unwrap();
        let decision = approximate();
        let proven = SemanticExactness::classify(decision.clone(), "bundle-a", Some(&native));
        assert!(proven.is_proven());
        let unproven = SemanticExactness::classify(decision, "bundle-a", None);
        assert!(!unproven.is_proven());
        assert!(matches!(
            unproven,
            SemanticExactness::Approximate {
                gap: SemanticGap::NoNativeDeclaration,
                ..
            }
        ));
        let mismatch = SelectionDecision::native(vec![id(1)]).unwrap();
        assert!(matches!(
            SemanticExactness::classify(mismatch, "bundle-a", Some(&native)),
            SemanticExactness::Approximate {
                gap: SemanticGap::SetMismatch,
                ..
            }
        ));
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
