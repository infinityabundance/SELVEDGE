//! The two-view head contract.
//!
//! The mandate requires independent heads for the value of executing a shard
//! and the cost of materialising it now. This module fixes the types and the
//! decision rule; the value head is trained on interventions in M2 and the
//! cost head is filled by real scheduler traces in M3. Until then the cost
//! head returns an explicit `unavailable` estimate, so a caller cannot mistake
//! a missing cost model for a measured zero.

use serde::{Deserialize, Serialize};

use crate::error::{AtlasError, Result};

/// A calibrated value estimate.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ValueEstimate {
    /// Mean predicted delta.
    pub mean: f64,
    /// Predictive variance.
    pub variance: f64,
    /// Lower bound of the declared interval.
    pub ci_low: f64,
    /// Upper bound of the declared interval.
    pub ci_high: f64,
    /// Provenance, e.g. the observation receipt or model id.
    pub provenance: String,
}

impl ValueEstimate {
    /// Construct and validate a value estimate.
    ///
    /// # Errors
    ///
    /// Rejects non-finite numbers, a negative variance and an inverted interval.
    pub fn new(
        mean: f64,
        variance: f64,
        ci_low: f64,
        ci_high: f64,
        provenance: impl Into<String>,
    ) -> Result<Self> {
        if ![mean, variance, ci_low, ci_high]
            .iter()
            .all(|value| value.is_finite())
        {
            return Err(AtlasError::invalid("value", "contains a non-finite number"));
        }
        if variance < 0.0 {
            return Err(AtlasError::invalid("variance", "is negative"));
        }
        if ci_low > ci_high {
            return Err(AtlasError::invalid("ci", "is inverted"));
        }
        Ok(Self {
            mean,
            variance,
            ci_low,
            ci_high,
            provenance: provenance.into(),
        })
    }
}

/// A cost estimate for materialising a shard now.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CostEstimate {
    /// Exposed seconds (not bytes over advertised bandwidth).
    pub exposed_seconds: f64,
    /// Bytes that must be read.
    pub bytes: u64,
    /// Relative uncertainty.
    pub uncertainty: f64,
    /// Whether the estimate is backed by a measured cost model.
    pub available: bool,
    /// Reason the estimate is unavailable.
    pub unavailable_reason: Option<String>,
}

impl CostEstimate {
    /// An explicit "no cost model yet" estimate.
    #[must_use]
    pub fn unavailable(reason: impl Into<String>) -> Self {
        Self {
            exposed_seconds: f64::NAN,
            bytes: 0,
            uncertainty: f64::NAN,
            available: false,
            unavailable_reason: Some(reason.into()),
        }
    }

    /// Construct a measured estimate.
    ///
    /// # Errors
    ///
    /// Rejects non-finite or negative numbers.
    pub fn measured(exposed_seconds: f64, bytes: u64, uncertainty: f64) -> Result<Self> {
        if !exposed_seconds.is_finite() || exposed_seconds < 0.0 {
            return Err(AtlasError::invalid("exposed_seconds", "is invalid"));
        }
        if !uncertainty.is_finite() || uncertainty < 0.0 {
            return Err(AtlasError::invalid("uncertainty", "is invalid"));
        }
        Ok(Self {
            exposed_seconds,
            bytes,
            uncertainty,
            available: true,
            unavailable_reason: None,
        })
    }
}

/// Decision status.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DecisionStatus {
    /// Both views are available.
    Ready,
    /// The value view is available but the cost view is not; the decision must
    /// be marked provisional.
    CostUnavailable,
}

/// The combined two-view decision.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TwoViewDecision {
    /// Value if executed.
    pub value_if_executed: ValueEstimate,
    /// Cost to materialise now.
    pub cost_to_materialize_now: CostEstimate,
    /// Combined score `value.mean - lambda * cost.exposed_seconds`.
    pub score: Option<f64>,
    /// Whether the decision is complete.
    pub status: DecisionStatus,
}

/// A value head.
pub trait ValueHead {
    /// Estimate the value of executing a shard in a context.
    ///
    /// # Errors
    ///
    /// Returns a typed error when the context cannot be scored.
    fn value_if_executed(&self, context: &str) -> Result<ValueEstimate>;
}

/// A cost head.
pub trait CostHead {
    /// Estimate the cost of materialising a shard now.
    ///
    /// # Errors
    ///
    /// Returns a typed error when the shard cannot be costed.
    fn cost_to_materialize_now(&self, shard: &str) -> Result<CostEstimate>;
}

/// A cost head that is explicitly not implemented yet.
#[derive(Debug, Default, Clone, Copy)]
pub struct UnavailableCostHead {
    reason: Option<&'static str>,
}

impl UnavailableCostHead {
    /// Construct the placeholder.
    #[must_use]
    pub fn new(reason: &'static str) -> Self {
        Self {
            reason: Some(reason),
        }
    }
}

impl CostHead for UnavailableCostHead {
    fn cost_to_materialize_now(&self, _shard: &str) -> Result<CostEstimate> {
        Ok(CostEstimate::unavailable(self.reason.unwrap_or(
            "cost head is filled by the M3 trace scheduler",
        )))
    }
}

/// Combine the two views.
///
/// # Errors
///
/// Rejects a non-finite or negative `lambda`, or a non-finite value mean.
pub fn combine(value: ValueEstimate, cost: CostEstimate, lambda: f64) -> Result<TwoViewDecision> {
    if !lambda.is_finite() || lambda < 0.0 {
        return Err(AtlasError::invalid(
            "lambda",
            "must be finite and non-negative",
        ));
    }
    if cost.available {
        let score = value.mean - lambda * cost.exposed_seconds;
        if !score.is_finite() {
            return Err(AtlasError::invalid("score", "is not finite"));
        }
        Ok(TwoViewDecision {
            value_if_executed: value,
            cost_to_materialize_now: cost,
            score: Some(score),
            status: DecisionStatus::Ready,
        })
    } else {
        Ok(TwoViewDecision {
            value_if_executed: value,
            cost_to_materialize_now: cost,
            score: None,
            status: DecisionStatus::CostUnavailable,
        })
    }
}
