//! Execution modes as types.
//!
//! [`ModeState::Exact`] can only be built for a plan whose manifest declares a
//! complete native set, and it can only produce a
//! [`selvedge_core::SemanticExactness::Proven`] value by comparing the shards it
//! actually read, per token, with that declaration. There is no method that
//! relabels a subset as exact, and no way to construct the exact variant from an
//! approximate selection.
//!
//! [`ModeState::Approximate`] chooses a subset with the control selector
//! documented in [`crate::config::SelectorSpec`] and can only ever classify as
//! [`selvedge_core::SemanticExactness::Approximate`].
//!
//! Both variants are deterministic: exact mode executes every group in
//! ascending `(layer, group)` order, and approximate mode selects by descending
//! score with ties broken by group index and then executes the selected groups
//! in ascending group order, so the reduction order never depends on the scores.

use std::collections::BTreeSet;

use selvedge_core::{
    ArmVerdict, CausalAvailability, CausalEvidence, CausalFailingReason, DecisionInput,
    ExactSelection, NativeComputationSet, NativeSetWitness, PhysicalCompliance, PhysicalEvidence,
    ReadMeasurement, SelectionDecision, SelectionProvenance, SemanticExactness, ShardId,
};
use selvedge_store::ExecutionScope;

use crate::bundle::BundlePlan;
use crate::config::{ExecutionMode, SelectorSpec};
use crate::error::{Result, RuntimeError};

/// Audit string recorded with the measured read counts.
///
/// It names exactly what produced the numbers: the store's own per-read reports
/// plus the ledger's union over the same identities.
pub const READ_AUDIT: &str = "ShardStore::read_required ReadReport bytes_used/device_bytes counters plus ParameterLedger union over the executed ParamUse set; test court_e";

/// Audit string recorded with the causal evidence.
pub const CAUSAL_AUDIT: &str = "crates/selvedge-runtime/src/modes.rs::decide reads only the current pre-MLP hidden state, the layer index and the position; test court_e_selector_reads_only_current_state";

/// One group's selection with the score that produced it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GroupChoice {
    /// Group index inside the layer.
    pub index: usize,
    /// Selector score. `0.0` in exact mode, where the score plays no part.
    pub score: f32,
}

/// The exact-mode state.
#[derive(Clone, Debug)]
pub struct ExactState {
    native: NativeComputationSet,
    last_witness: Option<NativeSetWitness>,
}

/// The approximate-mode state.
#[derive(Clone, Debug)]
pub struct ApproximateState {
    spec: SelectorSpec,
    /// Groups the selector asked for but the budget gate removed.
    trimmed: u64,
}

/// Which mode a run is in.
#[derive(Clone, Debug)]
pub enum ModeState {
    /// Every group of every layer.
    Exact(ExactState),
    /// A selector-chosen subset.
    Approximate(ApproximateState),
}

impl ModeState {
    /// Build the mode state for a plan.
    ///
    /// # Errors
    ///
    /// Rejects an approximate selector that keeps more groups than a layer has.
    pub fn new(plan: &BundlePlan, mode: ExecutionMode) -> Result<Self> {
        match mode {
            ExecutionMode::Exact => Ok(Self::Exact(ExactState {
                native: plan.native.clone(),
                last_witness: None,
            })),
            ExecutionMode::Approximate(spec) => {
                let groups = plan
                    .layers
                    .first()
                    .map(|layer| layer.groups.len())
                    .unwrap_or(0);
                if spec.keep_per_layer > groups {
                    return Err(RuntimeError::config(
                        "selector.keep_per_layer",
                        format!(
                            "keeps {} groups, but a layer has {groups}",
                            spec.keep_per_layer
                        ),
                    ));
                }
                for layer in &plan.layers {
                    if layer.groups.len() != groups {
                        return Err(RuntimeError::bundle(
                            "layers declare different group counts; the selector keeps a uniform count per layer",
                        ));
                    }
                }
                Ok(Self::Approximate(ApproximateState { spec, trimmed: 0 }))
            }
        }
    }

    /// Whether this is the exact variant.
    #[must_use]
    pub const fn is_exact(&self) -> bool {
        matches!(self, Self::Exact(_))
    }

    /// Stable mode label.
    #[must_use]
    pub const fn label(&self) -> &'static str {
        match self {
            Self::Exact(_) => "exact",
            Self::Approximate(_) => "approximate",
        }
    }

    /// Groups trimmed by the budget fallback so far.
    #[must_use]
    pub const fn trimmed_groups(&self) -> u64 {
        match self {
            Self::Exact(_) => 0,
            Self::Approximate(state) => state.trimmed,
        }
    }

    /// Record that `count` groups were removed by the within-budget fallback.
    pub fn note_trimmed(&mut self, count: u64) {
        if let Self::Approximate(state) = self {
            state.trimmed = state.trimmed.saturating_add(count);
        }
    }

    /// The selector's choice for one layer, sorted ascending by group index.
    ///
    /// `hidden` is the current pre-MLP hidden state. Exact mode ignores it.
    ///
    /// # Errors
    ///
    /// Rejects a hidden state whose length does not match the plan, and an
    /// approximate selection that keeps nothing.
    pub fn decide(
        &self,
        layer: usize,
        groups: &[crate::bundle::GroupPlan],
        hidden: &[f32],
        plan_hidden: u64,
    ) -> Result<Vec<GroupChoice>> {
        if hidden.len() as u64 != plan_hidden {
            return Err(RuntimeError::model(format!(
                "hidden state has {} values, the plan expects {plan_hidden}",
                hidden.len()
            )));
        }
        match self {
            Self::Exact(_) => Ok(groups
                .iter()
                .map(|group| GroupChoice {
                    index: group.index,
                    score: 0.0,
                })
                .collect()),
            Self::Approximate(state) => {
                let layer_index = u64::try_from(layer)
                    .map_err(|_| RuntimeError::model("layer index does not fit u64"))?;
                let mut scored: Vec<GroupChoice> = groups
                    .iter()
                    .map(|group| GroupChoice {
                        index: group.index,
                        score: projection_score(
                            state.spec.seed,
                            layer_index,
                            group.index as u64,
                            hidden,
                        ),
                    })
                    .collect();
                scored.sort_by(|left, right| {
                    right
                        .score
                        .total_cmp(&left.score)
                        .then_with(|| left.index.cmp(&right.index))
                });
                scored.truncate(state.spec.keep_per_layer.min(scored.len()));
                if scored.is_empty() {
                    return Err(RuntimeError::mode(
                        "the selector kept no groups; a layer must execute at least one",
                    ));
                }
                scored.sort_by_key(|choice| choice.index);
                Ok(scored)
            }
        }
    }

    /// Selector FLOPs for one layer: two per element of the projection.
    #[must_use]
    pub fn selector_flops(&self, groups: usize, hidden: u64) -> u64 {
        match self {
            Self::Exact(_) => 0,
            Self::Approximate(_) => 2 * groups as u64 * hidden,
        }
    }

    /// Learned selector parameter elements. Always zero: the control selector
    /// reads no learned parameters, only the current hidden state.
    #[must_use]
    pub const fn selector_parameter_elements(&self) -> u64 {
        0
    }

    /// Selector name for receipts.
    #[must_use]
    pub fn selector_name(&self) -> String {
        match self {
            Self::Exact(_) => "exact:native-set".to_owned(),
            Self::Approximate(state) => state.spec.name(),
        }
    }

    /// The store scope a token's reads are confined to.
    ///
    /// # Errors
    ///
    /// Rejects an empty selection.
    pub fn scope_for(
        &self,
        plan: &BundlePlan,
        selection: &BTreeSet<ShardId>,
    ) -> Result<ExecutionScope> {
        match self {
            Self::Exact(_) => plan.native_scope(),
            Self::Approximate(_) => Ok(ExecutionScope::exact(selection.iter().copied())?),
        }
    }

    /// Classify the shards one token actually read.
    ///
    /// Exact mode requires element-for-element equality with the declared native
    /// set and returns `Err` otherwise: a partial pass is a failed run, not a
    /// weaker claim.
    ///
    /// # Errors
    ///
    /// Rejects an empty executed set, a witness failure, and a decision whose
    /// digest disagrees with the witness.
    pub fn classify_token(
        &mut self,
        plan: &BundlePlan,
        executed: &BTreeSet<ShardId>,
    ) -> Result<SemanticExactness> {
        let shards: Vec<ShardId> = executed.iter().copied().collect();
        if shards.is_empty() {
            return Err(RuntimeError::exactness("the token executed no shards"));
        }
        let selector = self.selector_name();
        let decision = SelectionDecision::new(
            shards,
            SelectionProvenance::approximate(selector, Vec::new())?,
        )?;
        match self {
            Self::Exact(state) => {
                let witness =
                    NativeSetWitness::compare(&plan.bundle_id, decision.shards(), &state.native)?;
                let exact = ExactSelection::from_witness(decision, witness.clone())?;
                if exact.witness().declared_digest() != plan.native.digest() {
                    return Err(RuntimeError::exactness(
                        "witness declared digest does not match the plan declaration",
                    ));
                }
                state.last_witness = Some(witness);
                Ok(SemanticExactness::Proven { exact })
            }
            Self::Approximate(_) => Ok(SemanticExactness::classify(
                decision,
                &plan.bundle_id,
                Some(&plan.native),
            )),
        }
    }

    /// The witness of the last exact token, if any.
    #[must_use]
    pub fn last_witness(&self) -> Option<&NativeSetWitness> {
        match self {
            Self::Exact(state) => state.last_witness.as_ref(),
            Self::Approximate(_) => None,
        }
    }

    /// The two evidence-bearing verdicts for an approximate run.
    ///
    /// Exact runs return `None`: their evidence is the
    /// [`selvedge_core::ExactSelection`], which is a stronger and different
    /// claim than a compliance record.
    ///
    /// # Errors
    ///
    /// Rejects inconsistent evidence.
    pub fn verdict(
        &self,
        record: &str,
        accounting: &selvedge_core::TokenAccounting,
        controller_flops: u64,
    ) -> Result<Option<ArmVerdict>> {
        match self {
            Self::Exact(_) => Ok(None),
            Self::Approximate(_) => {
                let causal = CausalAvailability::available(CausalEvidence::new(
                    record,
                    vec![
                        DecisionInput::CurrentHiddenState,
                        DecisionInput::LayerIndex,
                        DecisionInput::PositionIndex,
                        DecisionInput::TokenPrefix,
                    ],
                    CAUSAL_AUDIT,
                    true,
                )?)?;
                let measured = ReadMeasurement::Measured {
                    audit: READ_AUDIT.to_owned(),
                };
                let reads: u64 = accounting.learned_bytes_by_tier.values().sum::<u64>() / 4;
                let physical = PhysicalCompliance::compliant(PhysicalEvidence::new(
                    self.selector_parameter_elements(),
                    0,
                    reads,
                    accounting.active_unique_params,
                    controller_flops,
                    accounting.flops,
                    measured,
                )?)?;
                Ok(Some(ArmVerdict::new(causal, physical)))
            }
        }
    }

    /// A reason a mode cannot proceed, used when a caller asks for exactness it
    /// cannot have.
    ///
    /// # Errors
    ///
    /// Always returns for the approximate variant.
    pub fn require_exact(&self) -> Result<()> {
        match self {
            Self::Exact(_) => Ok(()),
            Self::Approximate(_) => Err(RuntimeError::mode(
                "approximate execution cannot produce a semantic-exactness proof",
            )),
        }
    }

    /// The causal failing reason this runtime could produce, for documentation
    /// and receipts. Not user-facing logic.
    #[must_use]
    pub const fn forbidden_inputs() -> [CausalFailingReason; 2] {
        [
            CausalFailingReason::FutureTokens,
            CausalFailingReason::FutureLayers,
        ]
    }
}

/// A deterministic Rademacher projection of the hidden state.
///
/// The direction is derived from `(seed, layer, group)` with SplitMix64; there
/// is no learned parameter and no floating-point environment dependence.
fn projection_score(seed: u64, layer: u64, group: u64, hidden: &[f32]) -> f32 {
    let mut state = splitmix64(seed ^ splitmix64(layer ^ splitmix64(group)));
    let mut sum = 0.0_f32;
    for value in hidden {
        state = splitmix64(state);
        let sign = if state & 1 == 0 { 1.0_f32 } else { -1.0_f32 };
        sum += value * sign;
    }
    sum
}

const fn splitmix64(state: u64) -> u64 {
    let mut z = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn projection_is_deterministic_and_seed_dependent() {
        let hidden: Vec<f32> = (0..64).map(|index| (index as f32) * 0.01).collect();
        let first = projection_score(7, 1, 2, &hidden);
        assert_eq!(first, projection_score(7, 1, 2, &hidden));
        assert_ne!(first, projection_score(8, 1, 2, &hidden));
        assert_ne!(first, projection_score(7, 2, 2, &hidden));
    }

    #[test]
    fn splitmix_is_not_constant() {
        let mut state = 1_u64;
        let mut seen = BTreeSet::new();
        for _ in 0..64 {
            state = splitmix64(state);
            seen.insert(state);
        }
        assert_eq!(seen.len(), 64);
    }
}
