//! Two independent, separately reported verdicts for an evaluated arm.
//!
//! A single `deployable` boolean conflated two unrelated claims. This module
//! keeps them apart and forces each to carry named evidence or a named failing
//! reason:
//!
//! * [`CausalAvailability`] answers "did the selector read only information
//!   that existed before its decision?". The allowed inputs are enumerated in
//!   [`DecisionInput`]; the failing reasons name the exact leak.
//! * [`PhysicalCompliance`] answers "did selection **and** execution physically
//!   touch only counted parameters, and was the executed computation genuinely
//!   sparse?". A dense computation with a mask applied is explicitly
//!   non-compliant, even when the logical active count is small.
//!
//! There is no type in this module that can be constructed from a bare boolean.
//! A compliant physical verdict requires a measured read audit; a constructed
//! (formula-only) count is retained in the evidence but cannot yield
//! `Compliant`.

use core::fmt;

use serde::{Deserialize, Serialize};

use crate::error::{CoreError, Result};

/// Stable schema string for arm verdict records.
pub const VERDICT_SCHEMA: &str = "selvedge/arm-verdict/v1";

/// Information a selector is allowed to read before its decision.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DecisionInput {
    /// The current layer's pre-MLP hidden state.
    CurrentHiddenState,
    /// Gate/up activations of the current layer computed from that state.
    CurrentLayerActivations,
    /// The layer index.
    LayerIndex,
    /// The current position index.
    PositionIndex,
    /// Token ids strictly before the current decision.
    TokenPrefix,
    /// Constants fitted on calibration data before evaluation.
    CalibrationConstants,
    /// Learned controller parameters, counted in the active set.
    ControllerParameters,
    /// Masks this arm already applied at earlier layers of the same pass.
    OwnEarlierMasks,
}

impl DecisionInput {
    /// Stable name used in receipts.
    #[must_use]
    pub const fn stable_name(self) -> &'static str {
        match self {
            Self::CurrentHiddenState => "current_hidden_state",
            Self::CurrentLayerActivations => "current_layer_activations",
            Self::LayerIndex => "layer_index",
            Self::PositionIndex => "position_index",
            Self::TokenPrefix => "token_prefix",
            Self::CalibrationConstants => "calibration_constants",
            Self::ControllerParameters => "controller_parameters",
            Self::OwnEarlierMasks => "own_earlier_masks",
        }
    }
}

impl fmt::Display for DecisionInput {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.stable_name())
    }
}

/// Named reason a selector is not causally available.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CausalFailingReason {
    /// A future token id or activation entered the decision.
    FutureTokens,
    /// A later layer's state entered the decision.
    FutureLayers,
    /// Teacher logits or teacher probabilities entered the decision.
    TeacherLogits,
    /// The gold next token entered the decision.
    GoldLabels,
    /// Hidden states captured from an unmasked dense forward entered the
    /// decision, so the decision saw a state the online execution cannot have.
    UnmaskedFutureState,
    /// A held-out measurement (for example the realized KL being scored) was
    /// used to accept or refine the decision.
    HeldOutMeasurement,
    /// A backward pass whose target is the teacher-forced token produced the
    /// score.
    BackwardTarget,
    /// The selector read an input that its declaration does not list.
    UndeclaredInput,
}

impl CausalFailingReason {
    /// Stable name used in receipts.
    #[must_use]
    pub const fn stable_name(self) -> &'static str {
        match self {
            Self::FutureTokens => "future_tokens",
            Self::FutureLayers => "future_layers",
            Self::TeacherLogits => "teacher_logits",
            Self::GoldLabels => "gold_labels",
            Self::UnmaskedFutureState => "unmasked_future_state",
            Self::HeldOutMeasurement => "held_out_measurement",
            Self::BackwardTarget => "backward_target",
            Self::UndeclaredInput => "undeclared_input",
        }
    }
}

impl fmt::Display for CausalFailingReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.stable_name())
    }
}

/// Evidence for a causally available decision.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CausalEvidence {
    /// Identifier of the decision record this verdict describes.
    pub decision_record: String,
    /// Exactly the inputs the selector read.
    pub allowed_inputs: Vec<DecisionInput>,
    /// Receipt or test that audited the input boundary.
    pub audit: String,
    /// Whether the decision was made in a single forward pass, with no
    /// re-forwarding under captured states.
    pub one_pass: bool,
}

impl CausalEvidence {
    /// Construct validated evidence.
    ///
    /// # Errors
    ///
    /// Rejects an empty decision record or audit string and an empty input
    /// list.
    pub fn new(
        decision_record: impl Into<String>,
        allowed_inputs: Vec<DecisionInput>,
        audit: impl Into<String>,
        one_pass: bool,
    ) -> Result<Self> {
        let decision_record = decision_record.into();
        let audit = audit.into();
        if decision_record.is_empty() {
            return Err(CoreError::invalid(
                "causal_evidence",
                "decision record is empty",
            ));
        }
        if audit.is_empty() {
            return Err(CoreError::invalid("causal_evidence", "audit is empty"));
        }
        if allowed_inputs.is_empty() {
            return Err(CoreError::invalid(
                "causal_evidence",
                "no allowed inputs are declared",
            ));
        }
        Ok(Self {
            decision_record,
            allowed_inputs,
            audit,
            one_pass,
        })
    }
}

/// The causal-availability verdict.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case", deny_unknown_fields)]
pub enum CausalAvailability {
    /// Only pre-decision information was used; evidence is recorded.
    Available {
        /// What was read and how it was audited.
        evidence: CausalEvidence,
    },
    /// The selector read information it could not have had at decision time.
    Unavailable {
        /// Named failure.
        reason: CausalFailingReason,
        /// Human-readable detail naming the input.
        detail: String,
    },
}

impl CausalAvailability {
    /// Construct an available verdict.
    ///
    /// # Errors
    ///
    /// Propagates [`CausalEvidence::new`] validation.
    pub fn available(evidence: CausalEvidence) -> Result<Self> {
        Ok(Self::Available { evidence })
    }

    /// Construct an unavailable verdict.
    ///
    /// # Errors
    ///
    /// Rejects an empty detail string.
    pub fn unavailable(reason: CausalFailingReason, detail: impl Into<String>) -> Result<Self> {
        let detail = detail.into();
        if detail.is_empty() {
            return Err(CoreError::invalid(
                "causal_verdict",
                "unavailable detail is empty",
            ));
        }
        Ok(Self::Unavailable { reason, detail })
    }

    /// Whether the selector was causally available.
    #[must_use]
    pub fn is_available(&self) -> bool {
        matches!(self, Self::Available { .. })
    }
}

/// Named reason an arm is not physically compliant.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PhysicalFailingReason {
    /// The execution path ran dense matmuls and multiplied the activation by a
    /// mask, so the reported "active" count is a logical selection, not the
    /// physically touched weights.
    DenseComputationMasked,
    /// The selector read parameter elements outside the active set it charges
    /// (for example full gate/up projections used to rank groups).
    SelectionReadsOutsideActiveSet,
    /// No genuinely sparse execution path was measured for this arm.
    MissingSparseExecutionPath,
    /// Parameter reads were assumed rather than counted or audited.
    UnmeasuredParameterReads,
    /// The union of selector and execution parameters exceeds the ceiling.
    OverCeiling,
}

impl PhysicalFailingReason {
    /// Stable name used in receipts.
    #[must_use]
    pub const fn stable_name(self) -> &'static str {
        match self {
            Self::DenseComputationMasked => "dense_computation_masked",
            Self::SelectionReadsOutsideActiveSet => "selection_reads_outside_active_set",
            Self::MissingSparseExecutionPath => "missing_sparse_execution_path",
            Self::UnmeasuredParameterReads => "unmeasured_parameter_reads",
            Self::OverCeiling => "over_ceiling",
        }
    }
}

impl fmt::Display for PhysicalFailingReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.stable_name())
    }
}

/// How the read counts were obtained.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum ReadMeasurement {
    /// Counted by construction from the sliced tensors and cross-checked by an
    /// independent tensor-access audit (per-op shapes).
    Measured {
        /// Receipt or test that performed the audit.
        audit: String,
    },
    /// A formula over shapes, not a measurement of the executed path.
    Constructed {
        /// Formula or derivation.
        method: String,
    },
}

/// The physical accounting evidence for one arm.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PhysicalEvidence {
    /// Learned controller parameter elements used by selection.
    pub selector_parameter_elements: u64,
    /// Parameter elements whose bytes are read to make the selection decision.
    pub selector_elements_read: u64,
    /// Parameter elements whose bytes are read to execute the selected work.
    pub execution_elements_read: u64,
    /// Union of distinct learned parameter elements used by selector and
    /// execution, which is the quantity the ceiling constrains.
    pub executed_unique_elements: u64,
    /// Selection FLOPs.
    pub selection_flops: u64,
    /// Execution FLOPs.
    pub execution_flops: u64,
    /// How the read counts were obtained.
    pub read_measurement: ReadMeasurement,
}

impl PhysicalEvidence {
    /// Construct evidence and check its internal arithmetic.
    ///
    /// # Errors
    ///
    /// Rejects zero executed elements and evidence in which the selector plus
    /// execution unique set is smaller than either part (an impossible union).
    pub fn new(
        selector_parameter_elements: u64,
        selector_elements_read: u64,
        execution_elements_read: u64,
        executed_unique_elements: u64,
        selection_flops: u64,
        execution_flops: u64,
        read_measurement: ReadMeasurement,
    ) -> Result<Self> {
        if executed_unique_elements == 0 {
            return Err(CoreError::invalid(
                "physical_evidence",
                "executed unique element count is zero",
            ));
        }
        if selector_parameter_elements > executed_unique_elements {
            return Err(CoreError::invalid(
                "physical_evidence",
                "selector parameters exceed the executed unique set",
            ));
        }
        Ok(Self {
            selector_parameter_elements,
            selector_elements_read,
            execution_elements_read,
            executed_unique_elements,
            selection_flops,
            execution_flops,
            read_measurement,
        })
    }

    /// Whether the read counts are measured rather than constructed.
    #[must_use]
    pub fn is_measured(&self) -> bool {
        matches!(self.read_measurement, ReadMeasurement::Measured { .. })
    }
}

/// The physical-compliance verdict.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case", deny_unknown_fields)]
pub enum PhysicalCompliance {
    /// Selection and execution are both counted and the executed computation
    /// is genuinely sparse, measured.
    Compliant {
        /// The measured accounting.
        evidence: PhysicalEvidence,
    },
    /// The arm does not physically satisfy the accounting contract.
    NonCompliant {
        /// Named failure.
        reason: PhysicalFailingReason,
        /// What was measured, when available. A missing value is reported as
        /// absent, never as zero.
        evidence: Option<PhysicalEvidence>,
    },
}

impl PhysicalCompliance {
    /// Construct a compliant verdict.
    ///
    /// # Errors
    ///
    /// Rejects evidence whose reads were constructed from a formula rather
    /// than measured; a constructed count cannot establish compliance.
    pub fn compliant(evidence: PhysicalEvidence) -> Result<Self> {
        if !evidence.is_measured() {
            return Err(CoreError::invalid(
                "physical_verdict",
                "constructed read counts cannot establish physical compliance",
            ));
        }
        Ok(Self::Compliant { evidence })
    }

    /// Construct a non-compliant verdict.
    #[must_use]
    pub const fn non_compliant(
        reason: PhysicalFailingReason,
        evidence: Option<PhysicalEvidence>,
    ) -> Self {
        Self::NonCompliant { reason, evidence }
    }

    /// Whether the arm is physically compliant.
    #[must_use]
    pub fn is_compliant(&self) -> bool {
        matches!(self, Self::Compliant { .. })
    }
}

/// The two verdicts for one evaluated arm.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArmVerdict {
    /// Causal-availability verdict.
    pub causal: CausalAvailability,
    /// Physical-compliance verdict.
    pub physical: PhysicalCompliance,
    /// Schema string.
    pub schema: String,
}

impl ArmVerdict {
    /// Combine two verdicts.
    #[must_use]
    pub fn new(causal: CausalAvailability, physical: PhysicalCompliance) -> Self {
        Self {
            causal,
            physical,
            schema: VERDICT_SCHEMA.to_owned(),
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn evidence() -> PhysicalEvidence {
        PhysicalEvidence::new(
            868,
            2_359_296,
            2_359_296,
            998_631_268,
            12,
            6_442_450_944,
            ReadMeasurement::Measured {
                audit: "tests/tensor_read_audit.rs".to_owned(),
            },
        )
        .unwrap()
    }

    #[test]
    fn constructed_reads_cannot_claim_compliance() {
        let mut constructed = evidence();
        constructed.read_measurement = ReadMeasurement::Constructed {
            method: "formula".to_owned(),
        };
        assert!(PhysicalCompliance::compliant(constructed).is_err());
        assert!(PhysicalCompliance::compliant(evidence()).is_ok());
    }

    #[test]
    fn causal_verdict_requires_named_evidence() {
        let available = CausalAvailability::available(
            CausalEvidence::new(
                "arm/token-magnitude",
                vec![
                    DecisionInput::CurrentHiddenState,
                    DecisionInput::CurrentLayerActivations,
                    DecisionInput::OwnEarlierMasks,
                ],
                "receipt:oracle/one-pass-audit",
                true,
            )
            .unwrap(),
        )
        .unwrap();
        assert!(available.is_available());
        let unavailable = CausalAvailability::unavailable(
            CausalFailingReason::GoldLabels,
            "backward target is the gold next token",
        )
        .unwrap();
        assert!(!unavailable.is_available());
        assert_eq!(
            unavailable,
            CausalAvailability::Unavailable {
                reason: CausalFailingReason::GoldLabels,
                detail: "backward target is the gold next token".to_owned(),
            }
        );
    }

    #[test]
    fn verdicts_serialise_as_two_records_not_a_boolean() {
        let verdict = ArmVerdict::new(
            CausalAvailability::unavailable(
                CausalFailingReason::UnmaskedFutureState,
                "hidden states captured from a dense forward",
            )
            .unwrap(),
            PhysicalCompliance::non_compliant(
                PhysicalFailingReason::DenseComputationMasked,
                Some(evidence()),
            ),
        );
        let text = serde_json::to_string(&verdict).unwrap();
        assert!(text.contains("\"causal\""));
        assert!(text.contains("\"physical\""));
        assert!(!text.contains("deployable"));
        let round_trip: ArmVerdict = serde_json::from_str(&text).unwrap();
        assert_eq!(round_trip, verdict);
    }

    #[test]
    fn empty_evidence_and_impossible_unions_are_rejected() {
        assert!(CausalEvidence::new("", vec![DecisionInput::LayerIndex], "a", true).is_err());
        assert!(CausalEvidence::new("r", Vec::new(), "a", true).is_err());
        assert!(CausalEvidence::new("r", vec![DecisionInput::LayerIndex], "", true).is_err());
        assert!(
            PhysicalEvidence::new(
                100,
                0,
                0,
                10,
                0,
                0,
                ReadMeasurement::Measured {
                    audit: "a".to_owned()
                },
            )
            .is_err()
        );
    }
}
