//! Fitted selector weights and their versioned JSON contract.
//!
//! The Python oracle exports `results/selectors/m2-weights.json` in this
//! shape. `atlas fit` writes the same shape from sufficient statistics, so a
//! Rust service can load either file. The numeric fields are per layer:
//! a 6-vector unary model (five standardised group features plus a bias), the
//! standardisation constants, and a 5x5 bilinear interaction matrix.

use serde::{Deserialize, Serialize};

use crate::error::{AtlasError, Result};
use crate::input::GROUP_FEATURE_COUNT;

fn default_lambda_mix() -> f64 {
    1.0
}

/// The three calibration rankings exported by the Python harness.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Rankings {
    /// Descending magnitude ranking per layer.
    pub magnitude: Vec<Vec<u32>>,
    /// Descending frequency ranking per layer.
    pub frequency: Vec<Vec<u32>>,
    /// Seeded random permutation per layer.
    pub random: Vec<Vec<u32>>,
}

/// The versioned selector model.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AtlasWeights {
    /// Schema identifier.
    pub schema: String,
    /// Number of layers.
    pub layers: usize,
    /// Number of groups per layer.
    pub groups: usize,
    /// Feature names, in order.
    pub feature_names: Vec<String>,
    /// Per-layer unary weights (five group features plus a bias).
    pub unary_weights: Vec<Vec<f64>>,
    /// Per-layer feature means (5 group features plus bias).
    pub feature_mean: Vec<Vec<f64>>,
    /// Per-layer feature scales (5 group features plus bias).
    pub feature_scale: Vec<Vec<f64>>,
    /// Per-layer 5x5 interaction matrices.
    pub interaction: Vec<Vec<Vec<f64>>>,
    /// Mixing weight for the pairwise term.
    #[serde(default = "default_lambda_mix")]
    pub lambda_mix: f64,
    /// Optional calibration rankings for the static selectors.
    #[serde(default)]
    pub rankings: Option<Rankings>,
    /// Number of learned controller elements counted in the budget.
    #[serde(default)]
    pub parameter_elements: usize,
}

impl AtlasWeights {
    /// Validate all dimensions and finiteness.
    ///
    /// # Errors
    ///
    /// Rejects a zero layer/group count, a feature-name mismatch, per-layer
    /// vectors of the wrong length, a non-finite weight or a zero feature
    /// scale.
    pub fn validate(&self) -> Result<()> {
        let expected_features = GROUP_FEATURE_COUNT + 1;
        if self.schema.is_empty() {
            return Err(AtlasError::invalid("schema", "is empty"));
        }
        if self.layers == 0 || self.groups == 0 {
            return Err(AtlasError::invalid("layers", "must be non-zero"));
        }
        if self.feature_names.len() != expected_features {
            return Err(AtlasError::invalid(
                "feature_names",
                format!("expected {expected_features} names"),
            ));
        }
        if self.unary_weights.len() != self.layers
            || self.feature_mean.len() != self.layers
            || self.feature_scale.len() != self.layers
            || self.interaction.len() != self.layers
        {
            return Err(AtlasError::invalid(
                "layers",
                "per-layer vectors do not match the layer count",
            ));
        }
        for layer in 0..self.layers {
            if self.unary_weights[layer].len() != expected_features
                || self.feature_mean[layer].len() != expected_features
                || self.feature_scale[layer].len() != expected_features
            {
                return Err(AtlasError::invalid(
                    "unary_weights",
                    format!("layer {layer} does not have {expected_features} entries"),
                ));
            }
            if self.interaction[layer].len() != GROUP_FEATURE_COUNT
                || self.interaction[layer]
                    .iter()
                    .any(|row| row.len() != GROUP_FEATURE_COUNT)
            {
                return Err(AtlasError::invalid(
                    "interaction",
                    format!(
                        "layer {layer} is not a {GROUP_FEATURE_COUNT}x{GROUP_FEATURE_COUNT} matrix"
                    ),
                ));
            }
            let values = self.unary_weights[layer]
                .iter()
                .chain(self.feature_mean[layer].iter())
                .chain(self.feature_scale[layer].iter())
                .chain(self.interaction[layer].iter().flatten());
            if values.into_iter().any(|value| !value.is_finite()) {
                return Err(AtlasError::invalid(
                    "weights",
                    format!("layer {layer} contains a non-finite value"),
                ));
            }
            if self.feature_scale[layer].contains(&0.0) {
                return Err(AtlasError::invalid(
                    "feature_scale",
                    format!("layer {layer} has a zero scale"),
                ));
            }
            if !self.lambda_mix.is_finite() {
                return Err(AtlasError::invalid("lambda_mix", "is not finite"));
            }
        }
        if let Some(rankings) = &self.rankings {
            if rankings.magnitude.len() != self.layers
                || rankings.frequency.len() != self.layers
                || rankings.random.len() != self.layers
            {
                return Err(AtlasError::invalid(
                    "rankings",
                    "per-layer rankings do not match the layer count",
                ));
            }
            for layer in 0..self.layers {
                for ranking in [
                    &rankings.magnitude[layer],
                    &rankings.frequency[layer],
                    &rankings.random[layer],
                ] {
                    if ranking.len() != self.groups {
                        return Err(AtlasError::invalid(
                            "rankings",
                            format!("layer {layer} ranking does not cover every group"),
                        ));
                    }
                }
            }
        }
        Ok(())
    }

    /// Unary score for one group.
    ///
    /// # Errors
    ///
    /// Rejects an out-of-range layer or a feature vector of the wrong length.
    pub fn unary_score(&self, layer: usize, features: &[f64]) -> Result<f64> {
        if layer >= self.layers {
            return Err(AtlasError::invalid("layer", "is outside the fitted model"));
        }
        if features.len() != GROUP_FEATURE_COUNT {
            return Err(AtlasError::invalid(
                "features",
                format!("expected {GROUP_FEATURE_COUNT} features"),
            ));
        }
        let mean = &self.feature_mean[layer];
        let scale = &self.feature_scale[layer];
        let weights = &self.unary_weights[layer];
        let mut score = weights[GROUP_FEATURE_COUNT]
            * ((1.0 - mean[GROUP_FEATURE_COUNT]) / scale[GROUP_FEATURE_COUNT]);
        for index in 0..GROUP_FEATURE_COUNT {
            score += weights[index] * ((features[index] - mean[index]) / scale[index]);
        }
        Ok(score)
    }

    /// Standardised group features (the interaction model's input).
    ///
    /// # Errors
    ///
    /// Rejects an out-of-range layer or a feature vector of the wrong length.
    pub fn standardised_features(&self, layer: usize, features: &[f64]) -> Result<Vec<f64>> {
        if layer >= self.layers {
            return Err(AtlasError::invalid("layer", "is outside the fitted model"));
        }
        if features.len() != GROUP_FEATURE_COUNT {
            return Err(AtlasError::invalid(
                "features",
                format!("expected {GROUP_FEATURE_COUNT} features"),
            ));
        }
        Ok((0..GROUP_FEATURE_COUNT)
            .map(|index| {
                (features[index] - self.feature_mean[layer][index])
                    / self.feature_scale[layer][index]
            })
            .collect())
    }
}
