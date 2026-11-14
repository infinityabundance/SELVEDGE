//! Deployable selectors.
//!
//! Every selector implements [`Selector`], sees only [`SelectorInput`], and
//! returns a sorted group set. The implementations mirror the Python arms of
//! `oracle/m2_selectors.py` and are parity-tested on the exported fixtures.
//! The oracle is deliberately not implemented here: it needs the per-layer
//! group contributions and is not deployable.

use std::cmp::Ordering;

use crate::error::{AtlasError, Result};
use crate::input::{GROUP_FEATURE_COUNT, SelectorInput};
use crate::weights::AtlasWeights;

/// Deterministic seed shared with the Python harness.
pub const SEED: u64 = 20_261_114;

/// A selector that maps a pre-decision input to a group set.
pub trait Selector {
    /// Stable selector identifier.
    fn id(&self) -> &'static str;
    /// Select `k` groups for this input.
    ///
    /// # Errors
    ///
    /// Returns an error for a malformed input, a missing ranking, a missing
    /// fitted model, or `k` larger than the group count.
    fn select(&self, input: &SelectorInput, k: usize) -> Result<Vec<u32>>;
}

/// SplitMix64, matching the Python implementation byte for byte.
#[must_use]
pub fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9E37_79B9_7F4A_7C15);
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}

fn candidate_order(left: &(f64, u32), right: &(f64, u32)) -> Ordering {
    match right.0.partial_cmp(&left.0) {
        Some(Ordering::Equal) | None => left.1.cmp(&right.1),
        Some(order) => order,
    }
}

/// Deterministic top-k: descending score, ties by ascending group index.
///
/// # Errors
///
/// Rejects a non-finite score, an empty candidate list or `k == 0`.
pub fn top_k(scores: &[(f64, u32)], k: usize) -> Result<Vec<u32>> {
    if k == 0 {
        return Err(AtlasError::invalid("k", "must be positive"));
    }
    if scores.is_empty() {
        return Err(AtlasError::invalid("scores", "are empty"));
    }
    if scores.iter().any(|(score, _)| !score.is_finite()) {
        return Err(AtlasError::invalid("scores", "contain a non-finite value"));
    }
    let mut candidates = scores.to_vec();
    candidates.sort_by(candidate_order);
    let mut selected: Vec<u32> = candidates
        .into_iter()
        .take(k.min(scores.len()))
        .map(|(_, group)| group)
        .collect();
    selected.sort_unstable();
    Ok(selected)
}

/// Activation magnitude.
#[derive(Debug, Default, Clone, Copy)]
pub struct MagnitudeSelector;

impl Selector for MagnitudeSelector {
    fn id(&self) -> &'static str {
        "magnitude"
    }

    fn select(&self, input: &SelectorInput, k: usize) -> Result<Vec<u32>> {
        let scores: Vec<(f64, u32)> = input
            .groups
            .iter()
            .map(|group| (group.features[0], group.group))
            .collect();
        top_k(&scores, k)
    }
}

/// Per-token activation share.
#[derive(Debug, Default, Clone, Copy)]
pub struct FrequencySelector;

impl Selector for FrequencySelector {
    fn id(&self) -> &'static str {
        "frequency"
    }

    fn select(&self, input: &SelectorInput, k: usize) -> Result<Vec<u32>> {
        let scores: Vec<(f64, u32)> = input
            .groups
            .iter()
            .map(|group| (group.features[1], group.group))
            .collect();
        top_k(&scores, k)
    }
}

/// Deterministic random baseline.
#[derive(Debug, Default, Clone, Copy)]
pub struct RandomSelector;

impl Selector for RandomSelector {
    fn id(&self) -> &'static str {
        "random"
    }

    fn select(&self, input: &SelectorInput, k: usize) -> Result<Vec<u32>> {
        let scores: Vec<(f64, u32)> = input
            .groups
            .iter()
            .map(|group| {
                let seed = SEED
                    ^ (u64::from(input.layer)).wrapping_mul(0x9E37_79B9_7F4A_7C15)
                    ^ (u64::from(input.position)).wrapping_mul(0xBF58_476D_1CE4_E5B9)
                    ^ (u64::from(group.group)).wrapping_mul(0x94D0_49BB_1331_11EB)
                    ^ input.sequence_key;
                let value = splitmix64(seed) as f64 / 18_446_744_073_709_551_616.0;
                (value, group.group)
            })
            .collect();
        top_k(&scores, k)
    }
}

/// A static calibration ranking.
#[derive(Debug, Clone, Copy)]
pub enum StaticKind {
    /// Magnitude ranking.
    Magnitude,
    /// Frequency ranking.
    Frequency,
}

/// Static calibration selector.
#[derive(Debug, Clone, Copy)]
pub struct StaticSelector<'a> {
    weights: &'a AtlasWeights,
    kind: StaticKind,
}

impl<'a> StaticSelector<'a> {
    /// Construct the selector.
    pub fn new(weights: &'a AtlasWeights, kind: StaticKind) -> Self {
        Self { weights, kind }
    }
}

impl Selector for StaticSelector<'_> {
    fn id(&self) -> &'static str {
        match self.kind {
            StaticKind::Magnitude => "static-magnitude",
            StaticKind::Frequency => "static-frequency",
        }
    }

    fn select(&self, input: &SelectorInput, k: usize) -> Result<Vec<u32>> {
        let rankings =
            self.weights.rankings.as_ref().ok_or_else(|| {
                AtlasError::NotFitted("weights carry no calibration rankings".into())
            })?;
        let ranking = match self.kind {
            StaticKind::Magnitude => &rankings.magnitude,
            StaticKind::Frequency => &rankings.frequency,
        }
        .get(input.layer as usize)
        .ok_or_else(|| AtlasError::invalid("layer", "is outside the ranking"))?;
        let mut selected: Vec<u32> = ranking.iter().copied().take(k.min(ranking.len())).collect();
        selected.sort_unstable();
        Ok(selected)
    }
}

/// Learned unary selector.
#[derive(Debug, Clone, Copy)]
pub struct LearnedUnarySelector<'a> {
    weights: &'a AtlasWeights,
}

impl<'a> LearnedUnarySelector<'a> {
    /// Construct the selector.
    pub fn new(weights: &'a AtlasWeights) -> Self {
        Self { weights }
    }
}

impl Selector for LearnedUnarySelector<'_> {
    fn id(&self) -> &'static str {
        "learned-unary"
    }

    fn select(&self, input: &SelectorInput, k: usize) -> Result<Vec<u32>> {
        let layer = input.layer as usize;
        let mut scores = Vec::with_capacity(input.groups.len());
        for group in &input.groups {
            scores.push((
                self.weights.unary_score(layer, &group.features)?,
                group.group,
            ));
        }
        top_k(&scores, k)
    }
}

/// Learned interaction-aware selector: greedy maximisation of
/// `unary + lambda * phi_g^T M phi_j` over the selected set.
#[derive(Debug, Clone, Copy)]
pub struct LearnedInteractionSelector<'a> {
    weights: &'a AtlasWeights,
}

impl<'a> LearnedInteractionSelector<'a> {
    /// Construct the selector.
    pub fn new(weights: &'a AtlasWeights) -> Self {
        Self { weights }
    }
}

impl Selector for LearnedInteractionSelector<'_> {
    fn id(&self) -> &'static str {
        "learned-interaction"
    }

    fn select(&self, input: &SelectorInput, k: usize) -> Result<Vec<u32>> {
        let layer = input.layer as usize;
        if layer >= self.weights.layers {
            return Err(AtlasError::invalid("layer", "is outside the fitted model"));
        }
        let groups = input.groups.len();
        if k > groups {
            return Err(AtlasError::invalid("k", "is larger than the group count"));
        }
        let mut unary = Vec::with_capacity(groups);
        let mut phi = Vec::with_capacity(groups);
        for group in &input.groups {
            unary.push(self.weights.unary_score(layer, &group.features)?);
            phi.push(self.weights.standardised_features(layer, &group.features)?);
        }
        let matrix = &self.weights.interaction[layer];
        let mut projected = vec![vec![0.0_f64; GROUP_FEATURE_COUNT]; groups];
        for group in 0..groups {
            for column in 0..GROUP_FEATURE_COUNT {
                let mut value = 0.0;
                for row in 0..GROUP_FEATURE_COUNT {
                    value += phi[group][row] * matrix[row][column];
                }
                projected[group][column] = value;
            }
        }
        let mut selected = vec![false; groups];
        let mut sum_projected = [0.0_f64; GROUP_FEATURE_COUNT];
        let mut result = Vec::with_capacity(k);
        for _ in 0..k {
            let mut best: Option<(f64, usize)> = None;
            for group in 0..groups {
                if selected[group] {
                    continue;
                }
                let mut pair = 0.0;
                for index in 0..GROUP_FEATURE_COUNT {
                    pair += phi[group][index] * sum_projected[index];
                }
                let score = unary[group] + self.weights.lambda_mix * pair;
                if !score.is_finite() {
                    return Err(AtlasError::invalid("score", "is not finite"));
                }
                match best {
                    None => best = Some((score, group)),
                    Some((best_score, best_group)) => {
                        if score > best_score || (score == best_score && group < best_group) {
                            best = Some((score, group));
                        }
                    }
                }
            }
            if let Some((_score, group)) = best {
                selected[group] = true;
                result.push(input.groups[group].group);
                for index in 0..GROUP_FEATURE_COUNT {
                    sum_projected[index] += projected[group][index];
                }
            }
        }
        result.sort_unstable();
        Ok(result)
    }
}

/// Named selector kinds accepted by the CLI and the fixture tests.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SelectorKind {
    /// Per-token magnitude.
    Magnitude,
    /// Per-token frequency.
    Frequency,
    /// Deterministic random.
    Random,
    /// Static magnitude calibration ranking.
    StaticMagnitude,
    /// Static frequency calibration ranking.
    StaticFrequency,
    /// Learned unary.
    LearnedUnary,
    /// Learned interaction-aware.
    LearnedInteraction,
}

impl SelectorKind {
    /// Parse a CLI name.
    ///
    /// # Errors
    ///
    /// Rejects unknown names rather than guessing.
    pub fn parse(name: &str) -> Result<Self> {
        match name {
            "magnitude" => Ok(Self::Magnitude),
            "frequency" => Ok(Self::Frequency),
            "random" => Ok(Self::Random),
            "static-magnitude" => Ok(Self::StaticMagnitude),
            "static-frequency" => Ok(Self::StaticFrequency),
            "learned-unary" => Ok(Self::LearnedUnary),
            "learned-interaction" => Ok(Self::LearnedInteraction),
            other => Err(AtlasError::invalid(
                "selector",
                format!("unknown selector {other:?}"),
            )),
        }
    }

    /// The canonical name.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Magnitude => "magnitude",
            Self::Frequency => "frequency",
            Self::Random => "random",
            Self::StaticMagnitude => "static-magnitude",
            Self::StaticFrequency => "static-frequency",
            Self::LearnedUnary => "learned-unary",
            Self::LearnedInteraction => "learned-interaction",
        }
    }
}

/// Run one named selector.
///
/// # Errors
///
/// Propagates the selector's dimensional, fitting and finiteness errors.
pub fn select_named(
    kind: SelectorKind,
    input: &SelectorInput,
    weights: &AtlasWeights,
    k: usize,
) -> Result<Vec<u32>> {
    match kind {
        SelectorKind::Magnitude => MagnitudeSelector.select(input, k),
        SelectorKind::Frequency => FrequencySelector.select(input, k),
        SelectorKind::Random => RandomSelector.select(input, k),
        SelectorKind::StaticMagnitude => {
            StaticSelector::new(weights, StaticKind::Magnitude).select(input, k)
        }
        SelectorKind::StaticFrequency => {
            StaticSelector::new(weights, StaticKind::Frequency).select(input, k)
        }
        SelectorKind::LearnedUnary => LearnedUnarySelector::new(weights).select(input, k),
        SelectorKind::LearnedInteraction => {
            LearnedInteractionSelector::new(weights).select(input, k)
        }
    }
}
