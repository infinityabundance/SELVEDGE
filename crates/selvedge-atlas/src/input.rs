//! Pre-decision selector inputs and their leakage boundary.
//!
//! [`SelectorInput`] is the only record a deployed selector may consume. It
//! carries the pre-MLP hidden state, the layer's own group features, the layer
//! index, the current position and a sequence key derived from tokens that
//! precede the decision. It has no field for teacher logits, future tokens,
//! gold labels or the realized mask outcome, and it rejects unknown fields, so
//! a record that tries to carry future state fails to deserialize instead of
//! being silently ignored.

use serde::{Deserialize, Serialize};

use crate::error::{AtlasError, Result};

/// Number of pre-decision group features.
pub const GROUP_FEATURE_COUNT: usize = 5;

/// One group's pre-decision features.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GroupFeature {
    /// Group index.
    pub group: u32,
    /// `[log1p(||a_g||), share, mean(a_g), max(a_g), min(a_g)]`.
    pub features: Vec<f64>,
}

/// Everything a selector may read at decision time.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SelectorInput {
    /// Decoder layer where the decision is made.
    pub layer: u32,
    /// Number of layers in the model.
    pub layers: u32,
    /// Token position in the current sequence.
    pub position: u32,
    /// Stable key derived only from tokens at positions `<= position` (for the
    /// deterministic random baseline).
    pub sequence_key: u64,
    /// One record per group, in group order.
    pub groups: Vec<GroupFeature>,
    /// Optional raw hidden state, for auditability; the registered learned
    /// models do not read it.
    #[serde(default)]
    pub hidden: Option<Vec<f32>>,
}

impl SelectorInput {
    /// Validate dimensions and finiteness.
    ///
    /// # Errors
    ///
    /// Rejects a wrong group count, non-contiguous or duplicated group indices,
    /// a feature vector of the wrong length, non-finite features, an
    /// out-of-range layer and a non-finite hidden vector.
    pub fn validate(&self, layers: u32, groups: u32) -> Result<()> {
        if self.layers != layers {
            return Err(AtlasError::invalid(
                "layers",
                format!("expected {layers}, got {}", self.layers),
            ));
        }
        if self.layer >= self.layers {
            return Err(AtlasError::invalid("layer", "is outside the model"));
        }
        if self.groups.len() != groups as usize {
            return Err(AtlasError::invalid(
                "groups",
                format!("expected {groups} group records, got {}", self.groups.len()),
            ));
        }
        for (index, group) in self.groups.iter().enumerate() {
            if group.group != index as u32 {
                return Err(AtlasError::invalid(
                    "groups",
                    "records must be in ascending group order without gaps",
                ));
            }
            if group.features.len() != GROUP_FEATURE_COUNT {
                return Err(AtlasError::invalid(
                    "features",
                    format!(
                        "group {} has {} features, expected {GROUP_FEATURE_COUNT}",
                        group.group,
                        group.features.len()
                    ),
                ));
            }
            if group.features.iter().any(|value| !value.is_finite()) {
                return Err(AtlasError::invalid(
                    "features",
                    format!("group {} has a non-finite feature", group.group),
                ));
            }
        }
        if let Some(hidden) = &self.hidden
            && hidden.iter().any(|value| !value.is_finite())
        {
            return Err(AtlasError::invalid("hidden", "contains a non-finite value"));
        }
        Ok(())
    }

    /// Feature row for one group.
    #[must_use]
    pub fn features(&self, group: usize) -> &[f64] {
        &self.groups[group].features
    }
}
