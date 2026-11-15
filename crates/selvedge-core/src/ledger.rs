//! The learned-parameter ledger and the pre-execution budget gate.
//!
//! The central invariant of SELVEDGE is
//! `active_unique_params(t) = |union of underlying learned parameter elements
//! used to compute token t| <= 1_000_000_000` in compliance mode. This module
//! computes that union over [`ParamIdentity`] values, keeps repeated reads and
//! FLOPs as separate counters, accounts quantization metadata as non-learned
//! auxiliary bytes, and refuses an over-budget decision in compliance mode.
//!
//! The FLOP formula used by [`FlopEstimate::matmul`] is `2 * M * N * K`: one
//! multiply and one add per multiply-accumulate. Controller/router FLOPs are
//! tracked separately and are never mixed into the model FLOP counter.

use core::fmt;

use serde::de::{self, Visitor};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use thiserror::Error;

use crate::error::{CoreError, Result};
use crate::residency::StorageTier;
use crate::selection::SelectionDecision;
use crate::tensor::ParamIdentity;
use crate::units::{checked_add, checked_mul};

/// Category of non-learned auxiliary bytes.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum AuxiliaryKind {
    /// Quantization scales.
    Scales,
    /// Quantization zero points.
    ZeroPoints,
    /// Other quantization/repacking metadata.
    QuantMetadata,
    /// Anything else that is stored but is not a learned parameter.
    Other(String),
}

impl AuxiliaryKind {
    /// Stable wire name.
    #[must_use]
    pub fn stable_name(&self) -> String {
        match self {
            Self::Scales => "scales".to_owned(),
            Self::ZeroPoints => "zero_points".to_owned(),
            Self::QuantMetadata => "quant_metadata".to_owned(),
            Self::Other(raw) => format!("other:{raw}"),
        }
    }
}

impl TryFrom<String> for AuxiliaryKind {
    type Error = CoreError;

    fn try_from(value: String) -> Result<Self> {
        match value.as_str() {
            "scales" => Ok(Self::Scales),
            "zero_points" => Ok(Self::ZeroPoints),
            "quant_metadata" => Ok(Self::QuantMetadata),
            raw => {
                let name = raw.strip_prefix("other:").unwrap_or(raw);
                if name.is_empty() || name.len() > 64 {
                    return Err(CoreError::invalid(
                        "auxiliary_kind",
                        "auxiliary kind must be 1..=64 bytes",
                    ));
                }
                Ok(Self::Other(name.to_owned()))
            }
        }
    }
}

impl fmt::Display for AuxiliaryKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.stable_name())
    }
}

impl Serialize for AuxiliaryKind {
    fn serialize<S: Serializer>(&self, serializer: S) -> core::result::Result<S::Ok, S::Error> {
        serializer.serialize_str(&self.stable_name())
    }
}

impl<'de> Deserialize<'de> for AuxiliaryKind {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> core::result::Result<Self, D::Error> {
        struct KindVisitor;

        impl Visitor<'_> for KindVisitor {
            type Value = AuxiliaryKind;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("a stable SELVEDGE auxiliary byte kind")
            }

            fn visit_str<E: de::Error>(self, value: &str) -> core::result::Result<Self::Value, E> {
                AuxiliaryKind::try_from(value.to_owned()).map_err(E::custom)
            }
        }

        deserializer.deserialize_str(KindVisitor)
    }
}

/// Whether the 1B ceiling is a hard refusal or a labelled teacher exception.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ComplianceMode {
    /// A plan above the ceiling must not execute.
    Compliance,
    /// Oracle/teacher measurements may exceed the ceiling but are labelled and
    /// must never appear in compliant results.
    ResearchNoncompliantTeacher,
}

impl fmt::Display for ComplianceMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Compliance => f.write_str("compliance"),
            Self::ResearchNoncompliantTeacher => f.write_str("research_noncompliant_teacher"),
        }
    }
}

/// Active-unique-parameter ceiling and compliance mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Budget {
    /// Maximum unique learned parameter elements per output token.
    pub active_unique_limit: u64,
    /// Compliance or labelled teacher mode.
    pub mode: ComplianceMode,
}

impl Budget {
    /// Construct a validated budget.
    ///
    /// # Errors
    ///
    /// Rejects a zero limit.
    pub fn new(active_unique_limit: u64, mode: ComplianceMode) -> Result<Self> {
        if active_unique_limit == 0 {
            return Err(CoreError::invalid("budget", "limit is zero"));
        }
        Ok(Self {
            active_unique_limit,
            mode,
        })
    }
}

/// One learned-parameter use inside a token computation.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ParamUse {
    /// Underlying parameter identity (tensor key + element range).
    pub identity: ParamIdentity,
    /// Element reads including repeats.
    pub reads: u64,
    /// Bytes read from the tier.
    pub bytes: u64,
    /// Tier the bytes came from.
    pub tier: StorageTier,
}

impl ParamUse {
    /// Construct and validate a use.
    ///
    /// # Errors
    ///
    /// Rejects zero reads and fewer reads than the identity length (a used
    /// parameter must be read at least once).
    pub fn new(identity: ParamIdentity, reads: u64, bytes: u64, tier: StorageTier) -> Result<Self> {
        let use_value = Self {
            identity,
            reads,
            bytes,
            tier,
        };
        use_value.validate()?;
        Ok(use_value)
    }

    /// Construct a use that reads each element exactly once.
    ///
    /// # Errors
    ///
    /// Propagates [`ParamUse::new`].
    pub fn once(identity: ParamIdentity, bytes: u64, tier: StorageTier) -> Result<Self> {
        let reads = identity.len();
        Self::new(identity, reads, bytes, tier)
    }

    /// Validate read accounting.
    ///
    /// # Errors
    ///
    /// Rejects zero reads or reads below the identity length.
    pub fn validate(&self) -> Result<()> {
        if self.reads < self.identity.len() {
            return Err(CoreError::invalid(
                "param_use",
                format!(
                    "reads {} below identity length {}",
                    self.reads,
                    self.identity.len()
                ),
            ));
        }
        Ok(())
    }
}

/// Caller-supplied FLOP estimate for one step.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FlopEstimate {
    /// Dense matrix-multiply FLOPs (`2 * M * N * K` per call).
    pub matmul_flops: u64,
    /// Other model FLOPs (norms, activations, elementwise).
    pub other_flops: u64,
    /// Router/controller FLOPs, reported separately.
    pub controller_flops: u64,
}

impl FlopEstimate {
    /// FLOPs of one dense matmul with the documented `2*M*N*K` formula.
    ///
    /// # Errors
    ///
    /// Rejects overflow.
    pub fn matmul(m: u64, n: u64, k: u64) -> Result<Self> {
        let mn = checked_mul(m, n, "matmul M*N")?;
        let mnk = checked_mul(mn, k, "matmul M*N*K")?;
        let flops = checked_mul(mnk, 2, "matmul FLOPs")?;
        Ok(Self {
            matmul_flops: flops,
            other_flops: 0,
            controller_flops: 0,
        })
    }

    /// Model FLOPs (matmul + other), excluding controller FLOPs.
    ///
    /// # Errors
    ///
    /// Rejects overflow.
    pub fn model_flops(&self) -> Result<u64> {
        checked_add(self.matmul_flops, self.other_flops, "model FLOPs")
    }

    /// Accumulate another estimate.
    ///
    /// # Errors
    ///
    /// Rejects overflow in any counter.
    pub fn accumulate(&mut self, other: &Self) -> Result<()> {
        self.matmul_flops = checked_add(self.matmul_flops, other.matmul_flops, "matmul FLOPs")?;
        self.other_flops = checked_add(self.other_flops, other.other_flops, "other FLOPs")?;
        self.controller_flops = checked_add(
            self.controller_flops,
            other.controller_flops,
            "controller FLOPs",
        )?;
        Ok(())
    }
}

/// Placeholder quantization/dequantization work counters.
///
/// These are intentionally counts, not time estimates. A later phase may
/// replace them with measured kernel work; until then they are reported
/// verbatim and never folded into FLOPs or learned parameters.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QuantDequantWork {
    /// Logical elements dequantized for compute.
    pub dequantized_elements: u64,
    /// Logical elements requantized for storage.
    pub requantized_elements: u64,
}

/// Per-token accounting snapshot.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TokenAccounting {
    /// Union of learned parameter elements used by this token.
    pub active_unique_params: u64,
    /// Union of learned parameter elements resident at the peak.
    pub resident_unique_params: u64,
    /// Total accessible learned parameters in the substrate.
    pub total_accessible_params: u64,
    /// Element reads including repeats.
    pub parsed_parameter_reads: u64,
    /// Learned bytes read per tier.
    pub learned_bytes_by_tier: std::collections::BTreeMap<StorageTier, u64>,
    /// Non-learned auxiliary bytes (scales, zero points, metadata).
    pub auxiliary_bytes: std::collections::BTreeMap<AuxiliaryKind, u64>,
    /// Model FLOPs (matmul + other), excluding controller FLOPs.
    pub flops: u64,
    /// Router/controller FLOPs.
    pub controller_flops: u64,
    /// Quantization/dequantization work placeholder.
    pub quant_dequant_work: QuantDequantWork,
    /// Mode in which this accounting was produced.
    pub mode: ComplianceMode,
    /// Whether `active_unique_params <= budget_limit`.
    pub within_budget: bool,
    /// Ceiling the accounting was checked against.
    pub budget_limit: u64,
}

/// A plan that is checked against the budget before execution.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DecisionPlan {
    /// Selected shards and their provenance.
    pub decision: SelectionDecision,
    /// Learned parameter uses predicted for the step.
    pub uses: Vec<ParamUse>,
    /// Learned parameters resident while the plan executes.
    pub resident: Vec<ParamIdentity>,
    /// Predicted FLOPs.
    pub flops: FlopEstimate,
    /// Predicted non-learned auxiliary bytes.
    pub auxiliary: std::collections::BTreeMap<AuxiliaryKind, u64>,
    /// Predicted quantization/dequantization work.
    pub quant_dequant: QuantDequantWork,
}

impl DecisionPlan {
    /// Empty plan for a validated decision.
    #[must_use]
    pub fn new(decision: SelectionDecision) -> Self {
        Self {
            decision,
            uses: Vec::new(),
            resident: Vec::new(),
            flops: FlopEstimate::default(),
            auxiliary: std::collections::BTreeMap::new(),
            quant_dequant: QuantDequantWork::default(),
        }
    }

    /// Add a learned parameter use.
    ///
    /// # Errors
    ///
    /// Propagates [`ParamUse::validate`].
    pub fn push_use(&mut self, use_value: ParamUse) -> Result<()> {
        use_value.validate()?;
        self.uses.push(use_value);
        Ok(())
    }

    /// Union of learned parameter elements used by the plan.
    ///
    /// # Errors
    ///
    /// Rejects invalid uses and overflow.
    pub fn active_unique_params(&self) -> Result<u64> {
        self.uses.iter().try_for_each(ParamUse::validate)?;
        union_element_count(self.uses.iter().map(|use_value| &use_value.identity))
    }

    /// Element reads including repeats.
    ///
    /// # Errors
    ///
    /// Rejects overflow.
    pub fn parsed_parameter_reads(&self) -> Result<u64> {
        let mut total = 0_u64;
        for use_value in &self.uses {
            total = checked_add(total, use_value.reads, "parameter reads")?;
        }
        Ok(total)
    }
}

/// Union of element ranges, deduplicated per underlying tensor key.
///
/// Ranges under the same tensor key are merged whether they overlap or are
/// adjacent; different tensor keys never merge, so two distinct tensors that
/// happen to cover `0..N` count as `2N`. An empty iterator yields `0`.
///
/// # Errors
///
/// Rejects `u64` overflow.
pub fn union_element_count<'a>(
    identities: impl IntoIterator<Item = &'a ParamIdentity>,
) -> Result<u64> {
    let mut by_tensor: std::collections::BTreeMap<&str, Vec<(u64, u64)>> =
        std::collections::BTreeMap::new();
    for identity in identities {
        by_tensor
            .entry(identity.tensor().as_str())
            .or_default()
            .push((identity.elements().start, identity.elements().end));
    }
    let mut total = 0_u64;
    for mut ranges in by_tensor.into_values() {
        ranges.sort_unstable();
        let mut current: Option<(u64, u64)> = None;
        for (start, end) in ranges {
            current = match current {
                None => Some((start, end)),
                Some((current_start, current_end)) if start <= current_end => {
                    Some((current_start, current_end.max(end)))
                }
                Some((current_start, current_end)) => {
                    total = checked_add(
                        total,
                        current_end - current_start,
                        "unique parameter elements",
                    )?;
                    Some((start, end))
                }
            };
        }
        if let Some((start, end)) = current {
            total = checked_add(total, end - start, "unique parameter elements")?;
        }
    }
    Ok(total)
}

/// Why a plan was rejected.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum BudgetError {
    /// The plan exceeds the active-unique-parameter ceiling in compliance mode.
    #[error(
        "plan exceeds active unique parameter limit: {active_unique_params} > {limit} (mode {mode})"
    )]
    Exceeded {
        /// Computed unique parameter count.
        active_unique_params: u64,
        /// Configured ceiling.
        limit: u64,
        /// Mode that rejected the plan.
        mode: ComplianceMode,
    },
    /// A fallback plan was produced but is still over budget.
    #[error("within-budget fallback still exceeds limit: {active_unique_params} > {limit}")]
    FallbackExceeded {
        /// Computed unique parameter count of the fallback.
        active_unique_params: u64,
        /// Configured ceiling.
        limit: u64,
    },
    /// The plan itself is malformed.
    #[error("plan is inconsistent: {reason}")]
    InvalidPlan {
        /// Why the plan is malformed.
        reason: String,
    },
}

impl BudgetError {
    /// Whether this error is a ceiling violation rather than malformed input.
    #[must_use]
    pub fn is_exceeded(&self) -> bool {
        matches!(self, Self::Exceeded { .. } | Self::FallbackExceeded { .. })
    }
}

/// Result of a successful budget check.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum BudgetVerdict {
    /// The plan is inside the ceiling.
    Within {
        /// Computed unique parameter count.
        active_unique_params: u64,
        /// Configured ceiling.
        limit: u64,
    },
    /// The plan is over the ceiling and is labelled as a non-compliant teacher
    /// measurement. It must not be reported as a compliant result.
    NoncompliantTeacher {
        /// Computed unique parameter count.
        active_unique_params: u64,
        /// Configured ceiling.
        limit: u64,
        /// Stable label for the exception.
        label: String,
    },
}

impl BudgetVerdict {
    /// Whether the plan is inside the ceiling.
    #[must_use]
    pub fn is_within(&self) -> bool {
        matches!(self, Self::Within { .. })
    }
}

/// Hook that can substitute an in-budget plan before execution.
pub trait WithinBudgetFallback {
    /// Return a replacement plan, or `None` if no in-budget plan is known.
    ///
    /// The returned plan is re-checked; a fallback that remains over budget is
    /// rejected with [`BudgetError::FallbackExceeded`].
    fn fallback(&self, plan: &DecisionPlan) -> Option<DecisionPlan>;
}

impl<F> WithinBudgetFallback for F
where
    F: Fn(&DecisionPlan) -> Option<DecisionPlan>,
{
    fn fallback(&self, plan: &DecisionPlan) -> Option<DecisionPlan> {
        self(plan)
    }
}

/// A plan that passed the gate, possibly after substitution.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResolvedPlan {
    /// Plan that may execute.
    pub plan: DecisionPlan,
    /// Verdict for that plan.
    pub verdict: BudgetVerdict,
    /// Whether the fallback hook replaced the original plan.
    pub fallback_applied: bool,
}

/// Learned parameter uses of a selector path and of the execution it enables.
///
/// The ceiling constrains the *union*, not the execution alone: a selector that
/// reads a projection outside the active set has already spent the budget
/// before any FFN group is executed. Keeping the two lists separate makes the
/// selector's cost auditable instead of invisible.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SelectionPath {
    /// Parameters the selector/controller reads to make its decision.
    pub selector_uses: Vec<ParamUse>,
    /// Parameters the selected execution reads.
    pub execution_uses: Vec<ParamUse>,
    /// Selector/controller FLOPs.
    pub selector_flops: FlopEstimate,
    /// Execution FLOPs.
    pub execution_flops: FlopEstimate,
}

impl SelectionPath {
    /// Construct and validate a path.
    ///
    /// # Errors
    ///
    /// Rejects invalid [`ParamUse`] entries.
    pub fn new(
        selector_uses: Vec<ParamUse>,
        execution_uses: Vec<ParamUse>,
        selector_flops: FlopEstimate,
        execution_flops: FlopEstimate,
    ) -> Result<Self> {
        for use_value in selector_uses.iter().chain(execution_uses.iter()) {
            use_value.validate()?;
        }
        Ok(Self {
            selector_uses,
            execution_uses,
            selector_flops,
            execution_flops,
        })
    }

    /// Union of every learned parameter element touched by selection and
    /// execution. This is the quantity the ceiling constrains.
    ///
    /// # Errors
    ///
    /// Rejects overflow.
    pub fn active_unique_params(&self) -> Result<u64> {
        union_element_count(
            self.selector_uses
                .iter()
                .chain(self.execution_uses.iter())
                .map(|use_value| &use_value.identity),
        )
    }

    /// Union of the selector's own parameter elements.
    ///
    /// # Errors
    ///
    /// Rejects overflow.
    pub fn selector_unique_params(&self) -> Result<u64> {
        union_element_count(
            self.selector_uses
                .iter()
                .map(|use_value| &use_value.identity),
        )
    }

    /// Union of the executed parameter elements.
    ///
    /// # Errors
    ///
    /// Rejects overflow.
    pub fn execution_unique_params(&self) -> Result<u64> {
        union_element_count(
            self.execution_uses
                .iter()
                .map(|use_value| &use_value.identity),
        )
    }

    /// Element reads performed to make the decision, including repeats.
    ///
    /// # Errors
    ///
    /// Rejects overflow.
    pub fn selector_elements_read(&self) -> Result<u64> {
        sum_reads(&self.selector_uses)
    }

    /// Element reads performed to execute the selection, including repeats.
    ///
    /// # Errors
    ///
    /// Rejects overflow.
    pub fn execution_elements_read(&self) -> Result<u64> {
        sum_reads(&self.execution_uses)
    }

    /// Total model FLOPs (selector + execution), excluding neither part.
    ///
    /// # Errors
    ///
    /// Rejects overflow.
    pub fn total_flops(&self) -> Result<FlopEstimate> {
        let mut total = self.selector_flops;
        total.accumulate(&self.execution_flops)?;
        Ok(total)
    }
}

fn sum_reads(uses: &[ParamUse]) -> Result<u64> {
    let mut total = 0_u64;
    for use_value in uses {
        total = checked_add(total, use_value.reads, "parameter reads")?;
    }
    Ok(total)
}

/// A selection path admitted by the budget gate before execution.
///
/// The fields are private and there is no public constructor: a runtime that
/// accepts this type can assume both the selector and the execution it enables
/// were checked as one union. A path that cannot be admitted has no value to
/// pass to execution, so an over-budget selection cannot be executed first and
/// accounted later.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct AdmittedSelectionPath {
    path: SelectionPath,
    verdict: BudgetVerdict,
}

impl AdmittedSelectionPath {
    /// The admitted path.
    #[must_use]
    pub fn path(&self) -> &SelectionPath {
        &self.path
    }

    /// The gate's verdict.
    #[must_use]
    pub fn verdict(&self) -> &BudgetVerdict {
        &self.verdict
    }
}

/// Accumulates accounting for one token and gates plans against the ceiling.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ParameterLedger {
    budget: Budget,
    total_accessible: u64,
    uses: Vec<ParamUse>,
    resident: Vec<ParamIdentity>,
    flops: FlopEstimate,
    tier_bytes: std::collections::BTreeMap<StorageTier, u64>,
    auxiliary: std::collections::BTreeMap<AuxiliaryKind, u64>,
    quant_dequant: QuantDequantWork,
}

impl ParameterLedger {
    /// Start a ledger.
    ///
    /// # Errors
    ///
    /// Rejects an empty accessible substrate.
    pub fn begin(budget: Budget, total_accessible: u64) -> Result<Self> {
        if total_accessible == 0 {
            return Err(CoreError::invalid(
                "total_accessible",
                "accessible parameter count is zero",
            ));
        }
        Ok(Self {
            budget,
            total_accessible,
            uses: Vec::new(),
            resident: Vec::new(),
            flops: FlopEstimate::default(),
            tier_bytes: std::collections::BTreeMap::new(),
            auxiliary: std::collections::BTreeMap::new(),
            quant_dequant: QuantDequantWork::default(),
        })
    }

    /// Configured budget.
    #[must_use]
    pub const fn budget(&self) -> Budget {
        self.budget
    }

    /// Total accessible learned parameters.
    #[must_use]
    pub const fn total_accessible(&self) -> u64 {
        self.total_accessible
    }

    /// Record one learned parameter use.
    ///
    /// # Errors
    ///
    /// Rejects invalid uses and byte-count overflow.
    pub fn record_use(&mut self, use_value: ParamUse) -> Result<()> {
        use_value.validate()?;
        let tier_total = self.tier_bytes.entry(use_value.tier).or_insert(0);
        *tier_total = checked_add(*tier_total, use_value.bytes, "tier bytes")?;
        self.uses.push(use_value);
        Ok(())
    }

    /// Record a learned parameter as resident.
    pub fn record_resident(&mut self, identity: ParamIdentity) {
        self.resident.push(identity);
    }

    /// Record FLOPs.
    ///
    /// # Errors
    ///
    /// Rejects overflow.
    pub fn record_flops(&mut self, estimate: &FlopEstimate) -> Result<()> {
        self.flops.accumulate(estimate)
    }

    /// Record non-learned auxiliary bytes.
    ///
    /// # Errors
    ///
    /// Rejects overflow.
    pub fn record_auxiliary(&mut self, kind: AuxiliaryKind, bytes: u64) -> Result<()> {
        let total = self.auxiliary.entry(kind).or_insert(0);
        *total = checked_add(*total, bytes, "auxiliary bytes")?;
        Ok(())
    }

    /// Record quantization/dequantization work.
    ///
    /// # Errors
    ///
    /// Rejects overflow.
    pub fn record_quant_work(&mut self, work: &QuantDequantWork) -> Result<()> {
        self.quant_dequant.dequantized_elements = checked_add(
            self.quant_dequant.dequantized_elements,
            work.dequantized_elements,
            "dequantized elements",
        )?;
        self.quant_dequant.requantized_elements = checked_add(
            self.quant_dequant.requantized_elements,
            work.requantized_elements,
            "requantized elements",
        )?;
        Ok(())
    }

    /// Record every predicted counter from a plan.
    ///
    /// # Errors
    ///
    /// Rejects invalid uses and overflow.
    pub fn record_plan(&mut self, plan: &DecisionPlan) -> Result<()> {
        for use_value in &plan.uses {
            self.record_use(use_value.clone())?;
        }
        for identity in &plan.resident {
            self.record_resident(identity.clone());
        }
        self.record_flops(&plan.flops)?;
        for (kind, bytes) in &plan.auxiliary {
            self.record_auxiliary(kind.clone(), *bytes)?;
        }
        self.record_quant_work(&plan.quant_dequant)
    }

    /// Union of learned parameter elements used so far.
    ///
    /// # Errors
    ///
    /// Rejects overflow.
    pub fn active_unique_params(&self) -> Result<u64> {
        union_element_count(self.uses.iter().map(|use_value| &use_value.identity))
    }

    /// Union of learned parameter elements recorded resident.
    ///
    /// # Errors
    ///
    /// Rejects overflow.
    pub fn resident_unique_params(&self) -> Result<u64> {
        union_element_count(self.resident.iter())
    }

    /// Element reads including repeats.
    ///
    /// # Errors
    ///
    /// Rejects overflow.
    pub fn parsed_parameter_reads(&self) -> Result<u64> {
        let mut total = 0_u64;
        for use_value in &self.uses {
            total = checked_add(total, use_value.reads, "parameter reads")?;
        }
        Ok(total)
    }

    /// Snapshot all counters the mandate requires.
    ///
    /// # Errors
    ///
    /// Rejects counter overflow.
    pub fn snapshot(&self) -> Result<TokenAccounting> {
        let active_unique_params = self.active_unique_params()?;
        let model_flops = self.flops.model_flops()?;
        Ok(TokenAccounting {
            active_unique_params,
            resident_unique_params: self.resident_unique_params()?,
            total_accessible_params: self.total_accessible,
            parsed_parameter_reads: self.parsed_parameter_reads()?,
            learned_bytes_by_tier: self.tier_bytes.clone(),
            auxiliary_bytes: self.auxiliary.clone(),
            flops: model_flops,
            controller_flops: self.flops.controller_flops,
            quant_dequant_work: self.quant_dequant,
            mode: self.budget.mode,
            within_budget: active_unique_params <= self.budget.active_unique_limit,
            budget_limit: self.budget.active_unique_limit,
        })
    }

    /// Check a plan against the ceiling before execution.
    ///
    /// In [`ComplianceMode::Compliance`] an over-budget plan produces
    /// [`BudgetError::Exceeded`]; there is no boolean that a runtime could
    /// ignore. In [`ComplianceMode::ResearchNoncompliantTeacher`] an over-budget
    /// plan is returned as a labelled verdict.
    ///
    /// # Errors
    ///
    /// Returns [`BudgetError::Exceeded`] for an over-budget compliant plan and
    /// [`BudgetError::InvalidPlan`] for a malformed plan.
    pub fn check_decision(
        &self,
        plan: &DecisionPlan,
    ) -> core::result::Result<BudgetVerdict, BudgetError> {
        let active_unique_params =
            plan.active_unique_params()
                .map_err(|err| BudgetError::InvalidPlan {
                    reason: err.to_string(),
                })?;
        let limit = self.budget.active_unique_limit;
        if active_unique_params <= limit {
            return Ok(BudgetVerdict::Within {
                active_unique_params,
                limit,
            });
        }
        match self.budget.mode {
            ComplianceMode::Compliance => Err(BudgetError::Exceeded {
                active_unique_params,
                limit,
                mode: self.budget.mode,
            }),
            ComplianceMode::ResearchNoncompliantTeacher => Ok(BudgetVerdict::NoncompliantTeacher {
                active_unique_params,
                limit,
                label: "research_noncompliant_teacher".to_owned(),
            }),
        }
    }

    /// Check a plan, substituting an in-budget fallback when the hook supplies
    /// one.
    ///
    /// The fallback is only consulted in compliance mode. The returned plan is
    /// always within budget; a fallback that is still over budget is rejected.
    ///
    /// # Errors
    ///
    /// Returns [`BudgetError::Exceeded`] when no fallback exists,
    /// [`BudgetError::FallbackExceeded`] when the fallback is still over
    /// budget, and [`BudgetError::InvalidPlan`] for malformed plans.
    pub fn check_decision_with_fallback<F: WithinBudgetFallback>(
        &self,
        plan: &DecisionPlan,
        fallback: &F,
    ) -> core::result::Result<ResolvedPlan, BudgetError> {
        match self.check_decision(plan) {
            Ok(verdict) => Ok(ResolvedPlan {
                plan: plan.clone(),
                verdict,
                fallback_applied: false,
            }),
            Err(error) => {
                if self.budget.mode == ComplianceMode::ResearchNoncompliantTeacher {
                    return Err(error);
                }
                let Some(replacement) = fallback.fallback(plan) else {
                    return Err(error);
                };
                let active_unique_params =
                    replacement
                        .active_unique_params()
                        .map_err(|err| BudgetError::InvalidPlan {
                            reason: err.to_string(),
                        })?;
                if active_unique_params > self.budget.active_unique_limit {
                    return Err(BudgetError::FallbackExceeded {
                        active_unique_params,
                        limit: self.budget.active_unique_limit,
                    });
                }
                let verdict = self.check_decision(&replacement)?;
                Ok(ResolvedPlan {
                    plan: replacement,
                    verdict,
                    fallback_applied: true,
                })
            }
        }
    }

    /// Admit a selector path against the ceiling before execution.
    ///
    /// The union of selector and execution parameters is checked, so a selector
    /// that reads a projection outside the selected set is rejected instead of
    /// being charged only after the fact. In compliance mode an over-budget
    /// path yields [`BudgetError::Exceeded`] and no [`AdmittedSelectionPath`]
    /// exists to execute.
    ///
    /// # Errors
    ///
    /// Returns [`BudgetError::Exceeded`] for an over-budget compliant path and
    /// [`BudgetError::InvalidPlan`] when the path's accounting is malformed.
    pub fn admit_selection_path(
        &self,
        path: SelectionPath,
    ) -> core::result::Result<AdmittedSelectionPath, BudgetError> {
        let active_unique_params =
            path.active_unique_params()
                .map_err(|err| BudgetError::InvalidPlan {
                    reason: err.to_string(),
                })?;
        let limit = self.budget.active_unique_limit;
        let verdict = if active_unique_params <= limit {
            BudgetVerdict::Within {
                active_unique_params,
                limit,
            }
        } else {
            match self.budget.mode {
                ComplianceMode::Compliance => {
                    return Err(BudgetError::Exceeded {
                        active_unique_params,
                        limit,
                        mode: self.budget.mode,
                    });
                }
                ComplianceMode::ResearchNoncompliantTeacher => BudgetVerdict::NoncompliantTeacher {
                    active_unique_params,
                    limit,
                    label: "research_noncompliant_teacher".to_owned(),
                },
            }
        };
        Ok(AdmittedSelectionPath { path, verdict })
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::hash::Sha256Digest;
    use crate::ids::ShardId;
    use crate::selection::{SelectionDecision, SelectionProvenance};
    use crate::tensor::{ParamIdentity, ParamTensorId};

    fn tensor(key: &str) -> ParamTensorId {
        ParamTensorId::new(key).unwrap()
    }

    fn identity(key: &str, start: u64, end: u64) -> ParamIdentity {
        ParamIdentity::new(tensor(key), start..end).unwrap()
    }

    #[test]
    fn union_merges_overlaps_and_adjacent_ranges() {
        let ranges = [
            identity("t", 0, 100),
            identity("t", 50, 150),
            identity("t", 100, 200),
            identity("u", 0, 10),
        ];
        assert_eq!(union_element_count(ranges.iter()).unwrap(), 210);
    }

    #[test]
    fn budget_boundary_semantics() {
        let limit = 1_000_000_000_u64;
        let budget = Budget::new(limit, ComplianceMode::Compliance).unwrap();
        let ledger = ParameterLedger::begin(budget, limit * 2).unwrap();
        let decision = SelectionDecision::new(
            vec![ShardId::from_digest(Sha256Digest::of(b"s"))],
            SelectionProvenance::approximate("test", Vec::new()).unwrap(),
        )
        .unwrap();
        let mut plan = DecisionPlan::new(decision);
        plan.push_use(
            ParamUse::once(identity("a", 0, limit), limit * 4, StorageTier::Nvme).unwrap(),
        )
        .unwrap();
        assert!(ledger.check_decision(&plan).unwrap().is_within());
        plan.uses[0] = ParamUse::once(
            identity("a", 0, limit + 1),
            (limit + 1) * 4,
            StorageTier::Nvme,
        )
        .unwrap();
        assert!(matches!(
            ledger.check_decision(&plan),
            Err(BudgetError::Exceeded { .. })
        ));
    }

    #[test]
    fn auxiliary_kind_serialises_as_stable_strings() {
        for kind in [
            AuxiliaryKind::Scales,
            AuxiliaryKind::ZeroPoints,
            AuxiliaryKind::QuantMetadata,
            AuxiliaryKind::Other("sidecar".to_owned()),
        ] {
            let text = serde_json::to_string(&kind).unwrap();
            let parsed: AuxiliaryKind = serde_json::from_str(&text).unwrap();
            assert_eq!(parsed, kind);
        }
        let map = std::collections::BTreeMap::from([(AuxiliaryKind::Scales, 8_u64)]);
        let text = serde_json::to_string(&map).unwrap();
        let parsed: std::collections::BTreeMap<AuxiliaryKind, u64> =
            serde_json::from_str(&text).unwrap();
        assert_eq!(parsed, map);
    }

    #[test]
    fn over_budget_selection_path_is_rejected_before_execution() {
        let limit = 1_000_000_u64;
        let budget = Budget::new(limit, ComplianceMode::Compliance).unwrap();
        let ledger = ParameterLedger::begin(budget, limit * 4).unwrap();
        // The execution alone fits; the selector reads a projection outside it.
        let execution = vec![
            ParamUse::once(identity("ffn", 0, limit / 2), limit * 2, StorageTier::Nvme).unwrap(),
        ];
        let selector = vec![
            ParamUse::once(identity("gate-up", 0, limit), limit * 4, StorageTier::Nvme).unwrap(),
        ];
        let path = SelectionPath::new(
            selector,
            execution,
            FlopEstimate::default(),
            FlopEstimate::default(),
        )
        .unwrap();
        assert!(matches!(
            ledger.admit_selection_path(path.clone()),
            Err(BudgetError::Exceeded { .. })
        ));
        // A selector inside the same union is admitted, and the admitted type
        // is the only thing execution could accept.
        let small_path = SelectionPath::new(
            vec![
                ParamUse::once(identity("controller", 0, 100), 400, StorageTier::HostRam).unwrap(),
            ],
            vec![
                ParamUse::once(identity("ffn", 0, limit / 2), limit * 2, StorageTier::Nvme)
                    .unwrap(),
            ],
            FlopEstimate::default(),
            FlopEstimate::default(),
        )
        .unwrap();
        let admitted = ledger.admit_selection_path(small_path).unwrap();
        assert!(admitted.verdict().is_within());
        assert_eq!(
            admitted.path().active_unique_params().unwrap(),
            limit / 2 + 100
        );
    }

    #[test]
    fn teacher_mode_labels_an_over_budget_selection_path_instead_of_executing_it() {
        let limit = 1_000_u64;
        let budget = Budget::new(limit, ComplianceMode::ResearchNoncompliantTeacher).unwrap();
        let ledger = ParameterLedger::begin(budget, limit * 4).unwrap();
        let path = SelectionPath::new(
            vec![ParamUse::once(identity("a", 0, limit), limit * 4, StorageTier::Nvme).unwrap()],
            vec![ParamUse::once(identity("b", 0, limit), limit * 4, StorageTier::Nvme).unwrap()],
            FlopEstimate::default(),
            FlopEstimate::default(),
        )
        .unwrap();
        let admitted = ledger.admit_selection_path(path).unwrap();
        assert!(matches!(
            admitted.verdict(),
            BudgetVerdict::NoncompliantTeacher { .. }
        ));
    }
}
