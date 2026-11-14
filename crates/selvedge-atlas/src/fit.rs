//! Ridge fitting from the Python harness's sufficient statistics.
//!
//! `oracle/m2_selectors.py prepare` exports the normal equations of the
//! standardised intervention labels. Fitting from sufficient statistics keeps
//! the deployment artefact small while remaining a real closed-form solve; the
//! Python and Rust paths are parity-tested on the same file.

use serde::{Deserialize, Serialize};

use crate::error::{AtlasError, Result};
use crate::input::GROUP_FEATURE_COUNT;
use crate::weights::AtlasWeights;

fn default_lambda_mix() -> f64 {
    1.0
}

/// One layer's sufficient statistics.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FitLayer {
    /// Decoder layer.
    pub layer: usize,
    /// `X^T X` for the standardised unary features.
    pub unary_xtx: Vec<Vec<f64>>,
    /// `X^T y` for the unary labels.
    pub unary_xty: Vec<f64>,
    /// Number of unary labels.
    pub unary_count: usize,
    /// Ridge coefficient selected on the validation split.
    pub unary_lambda: f64,
    /// `X^T X` for the standardised pairwise products.
    pub pair_xtx: Vec<Vec<f64>>,
    /// `X^T y` for the pairwise labels.
    pub pair_xty: Vec<f64>,
    /// Number of pairwise labels.
    pub pair_count: usize,
    /// Ridge coefficient selected on the validation split.
    pub pair_lambda: f64,
    /// Standardisation means (five group features plus bias).
    pub feature_mean: Vec<f64>,
    /// Standardisation scales (five group features plus bias).
    pub feature_scale: Vec<f64>,
}

/// The fit-data file.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FitData {
    /// Schema identifier.
    pub schema: String,
    /// Per-layer sufficient statistics.
    pub layers: Vec<FitLayer>,
    /// Feature names.
    pub feature_names: Vec<String>,
    /// Number of groups per layer.
    pub groups: usize,
    /// Mixing weight for the pairwise term.
    #[serde(default = "default_lambda_mix")]
    pub lambda_mix: f64,
}

/// Solve `(A + lambda I) x = b` by Gaussian elimination with partial pivoting.
///
/// # Errors
///
/// Rejects a non-square matrix, a mismatched right-hand side, a non-finite
/// entry, or a singular system.
pub fn solve_ridge(matrix: &[Vec<f64>], rhs: &[f64], lambda: f64) -> Result<Vec<f64>> {
    let n = matrix.len();
    if n == 0 || rhs.len() != n || matrix.iter().any(|row| row.len() != n) {
        return Err(AtlasError::invalid(
            "matrix",
            "must be non-empty, square and match the right-hand side",
        ));
    }
    if !lambda.is_finite() || lambda < 0.0 {
        return Err(AtlasError::invalid(
            "lambda",
            "must be finite and non-negative",
        ));
    }
    let mut augmented = vec![vec![0.0_f64; n + 1]; n];
    for row in 0..n {
        for column in 0..n {
            let value = matrix[row][column];
            if !value.is_finite() {
                return Err(AtlasError::invalid("matrix", "contains a non-finite value"));
            }
            augmented[row][column] = value + if row == column { lambda } else { 0.0 };
        }
        if !rhs[row].is_finite() {
            return Err(AtlasError::invalid("rhs", "contains a non-finite value"));
        }
        augmented[row][n] = rhs[row];
    }
    for column in 0..n {
        let mut pivot = column;
        let mut best = augmented[column][column].abs();
        for (row, values) in augmented.iter().enumerate().skip(column + 1) {
            let value = values[column].abs();
            if value > best {
                best = value;
                pivot = row;
            }
        }
        if best < 1e-14 {
            return Err(AtlasError::NotFitted(format!(
                "singular system at column {column}"
            )));
        }
        if pivot != column {
            augmented.swap(column, pivot);
        }
        let divisor = augmented[column][column];
        let pivot_row = augmented[column].clone();
        for values in augmented.iter_mut().skip(column + 1) {
            let factor = values[column] / divisor;
            if factor == 0.0 {
                continue;
            }
            for (index, value) in values.iter_mut().enumerate().skip(column) {
                *value -= factor * pivot_row[index];
            }
        }
    }
    let mut solution = vec![0.0_f64; n];
    for row in (0..n).rev() {
        let mut value = augmented[row][n];
        for column in (row + 1)..n {
            value -= augmented[row][column] * solution[column];
        }
        solution[row] = value / augmented[row][row];
        if !solution[row].is_finite() {
            return Err(AtlasError::NotFitted("non-finite solution".into()));
        }
    }
    Ok(solution)
}

/// Fit the full selector model from sufficient statistics.
///
/// # Errors
///
/// Rejects a malformed fit data file, a non-positive group count, a layer
/// outside the declared set, or a singular ridge system.
pub fn fit(data: &FitData) -> Result<AtlasWeights> {
    if data.groups == 0 {
        return Err(AtlasError::invalid("groups", "must be positive"));
    }
    let expected_features = GROUP_FEATURE_COUNT + 1;
    let pair_dim = GROUP_FEATURE_COUNT * GROUP_FEATURE_COUNT;
    let mut layers = 0_usize;
    for layer in &data.layers {
        layers = layers.max(layer.layer + 1);
        if layer.feature_mean.len() != expected_features
            || layer.feature_scale.len() != expected_features
        {
            return Err(AtlasError::invalid(
                "feature_mean",
                format!("layer {} has the wrong feature dimension", layer.layer),
            ));
        }
        if layer.unary_xtx.len() != expected_features
            || layer.unary_xty.len() != expected_features
            || layer.pair_xtx.len() != pair_dim
            || layer.pair_xty.len() != pair_dim
        {
            return Err(AtlasError::invalid(
                "fit",
                format!(
                    "layer {} has the wrong normal-equation dimension",
                    layer.layer
                ),
            ));
        }
    }
    if layers == 0 {
        return Err(AtlasError::invalid("layers", "fit data has no layers"));
    }
    let mut weights = AtlasWeights {
        schema: "selvedge/atlas/weights-v1".into(),
        layers,
        groups: data.groups,
        feature_names: data.feature_names.clone(),
        unary_weights: vec![vec![0.0; expected_features]; layers],
        feature_mean: vec![vec![0.0; expected_features]; layers],
        feature_scale: vec![vec![1.0; expected_features]; layers],
        interaction: vec![vec![vec![0.0; GROUP_FEATURE_COUNT]; GROUP_FEATURE_COUNT]; layers],
        lambda_mix: data.lambda_mix,
        rankings: None,
        parameter_elements: layers * (expected_features + pair_dim),
    };
    for layer in &data.layers {
        if layer.layer >= layers {
            return Err(AtlasError::invalid(
                "layer",
                "is outside the fitted layer count",
            ));
        }
        let unary = solve_ridge(&layer.unary_xtx, &layer.unary_xty, layer.unary_lambda)?;
        let pair = solve_ridge(&layer.pair_xtx, &layer.pair_xty, layer.pair_lambda)?;
        weights.unary_weights[layer.layer] = unary;
        weights.feature_mean[layer.layer] = layer.feature_mean.clone();
        weights.feature_scale[layer.layer] = layer.feature_scale.clone();
        for row in 0..GROUP_FEATURE_COUNT {
            for column in 0..GROUP_FEATURE_COUNT {
                let forward = pair[row * GROUP_FEATURE_COUNT + column];
                let backward = pair[column * GROUP_FEATURE_COUNT + row];
                weights.interaction[layer.layer][row][column] = 0.5 * (forward + backward);
            }
        }
    }
    weights.validate()?;
    Ok(weights)
}
