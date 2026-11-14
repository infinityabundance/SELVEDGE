//! Intervention-labelled quality observations.
//!
//! An observation records one measured causal quantity for one context. It is
//! deliberately narrow: a ledger record, not a claim. Every observation names
//! the receipt that produced it, so a report can never cite an observation
//! without naming its source.

use serde::{Deserialize, Serialize};

use crate::error::{AtlasError, Result};

/// One measured quality intervention.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QualityObservation {
    /// Model digest the observation was measured on.
    pub model_digest: String,
    /// Decoder layer.
    pub layer: u32,
    /// Group (or shard) index.
    pub group: u32,
    /// Stable context identifier (document, window, position, layer).
    pub context_id: String,
    /// Base set the intervention was measured against, sorted ascending.
    pub base_set: Vec<u32>,
    /// Metric name, e.g. `teacher_kl` or `local_squared_error`.
    pub metric: String,
    /// Measured marginal quantity. Sign convention is metric-specific and is
    /// stated by the producing receipt.
    pub delta: f64,
    /// One-sigma (or declared) uncertainty of `delta`.
    pub uncertainty: f64,
    /// Tracked receipt that produced this observation.
    pub source_receipt: String,
    /// Receipt stamp (UTC, `YYYYMMDDTHHMMSSZ`).
    pub stamp: String,
}

impl QualityObservation {
    /// Validate the record against the model's layer and group counts.
    ///
    /// # Errors
    ///
    /// Rejects out-of-range indices, non-finite numbers, negative uncertainty,
    /// duplicate or unsorted base sets, and a base set containing the group
    /// being measured (which would make the intervention a no-op).
    pub fn validate(&self, layers: u32, groups: u32) -> Result<()> {
        if self.model_digest.is_empty() {
            return Err(AtlasError::invalid("model_digest", "is empty"));
        }
        if self.context_id.is_empty() {
            return Err(AtlasError::invalid("context_id", "is empty"));
        }
        if self.metric.is_empty() {
            return Err(AtlasError::invalid("metric", "is empty"));
        }
        if self.source_receipt.is_empty() {
            return Err(AtlasError::invalid("source_receipt", "is empty"));
        }
        if self.stamp.is_empty() {
            return Err(AtlasError::invalid("stamp", "is empty"));
        }
        if self.layer >= layers {
            return Err(AtlasError::invalid("layer", "is outside the model"));
        }
        if self.group >= groups {
            return Err(AtlasError::invalid("group", "is outside the model"));
        }
        if !self.delta.is_finite() {
            return Err(AtlasError::invalid("delta", "is not finite"));
        }
        if !self.uncertainty.is_finite() || self.uncertainty < 0.0 {
            return Err(AtlasError::invalid(
                "uncertainty",
                "must be finite and non-negative",
            ));
        }
        for pair in self.base_set.windows(2) {
            if pair[0] >= pair[1] {
                return Err(AtlasError::invalid(
                    "base_set",
                    "must be strictly ascending and unique",
                ));
            }
        }
        if self.base_set.iter().any(|value| *value >= groups) {
            return Err(AtlasError::invalid(
                "base_set",
                "contains a group outside the model",
            ));
        }
        if self.base_set.contains(&self.group) {
            return Err(AtlasError::invalid(
                "base_set",
                "contains the group being measured",
            ));
        }
        Ok(())
    }
}
