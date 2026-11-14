//! Independent Rust verification of the matched-budget selector table.
//!
//! `atlas compare` re-derives every arm's realized active unique parameters
//! from its realized group counts and the receipt-sourced model geometry, then
//! performs the paired bootstrap in Rust. The Python analysis is the published
//! table; this is an independent accounting check.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::error::{AtlasError, Result};
use crate::selector::{SEED, splitmix64};

/// One arm's realized allocation and per-item metrics.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CompareArm {
    /// Arm identifier.
    pub id: String,
    /// Registered budget name.
    pub budget: String,
    /// Realized groups per layer.
    pub groups_per_layer: Vec<u32>,
    /// Group width in channels.
    pub group_width: u32,
    /// Learned controller elements counted for this arm.
    pub controller_elements: u64,
    /// Per-document mean KL.
    pub items: Vec<CompareItem>,
}

/// One document's mean teacher KL.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CompareItem {
    /// Document identifier.
    pub doc_id: String,
    /// Mean KL in nats/token.
    pub kl_mean: f64,
}

/// The comparison input.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CompareInput {
    /// Schema identifier.
    pub schema: String,
    /// Active-parameter ceiling.
    pub ceiling: u64,
    /// Always-active backbone elements.
    pub backbone_elements: u64,
    /// Hidden width.
    pub hidden: u64,
    /// Intermediate width.
    pub intermediate: u64,
    /// Baseline arm id.
    pub baseline: String,
    /// Arms to compare.
    pub arms: Vec<CompareArm>,
}

/// Per-arm accounting row.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CompareArmReport {
    /// Arm identifier.
    pub id: String,
    /// Budget name.
    pub budget: String,
    /// Realized active unique elements, controller included.
    pub active_unique_elements: u64,
    /// Whether the arm is inside the ceiling.
    pub compliant: bool,
    /// Difference from the baseline's active count.
    pub active_delta_from_baseline: i64,
    /// Whether the comparison rule treats this arm as matched.
    pub matched: bool,
}

/// One paired comparison.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CompareReportEntry {
    /// Arm identifier.
    pub arm: String,
    /// Baseline identifier.
    pub baseline: String,
    /// Number of shared documents.
    pub items: usize,
    /// Whether the comparison could be computed.
    pub available: bool,
    /// Baseline mean KL.
    pub base_kl: Option<f64>,
    /// Arm mean KL.
    pub arm_kl: Option<f64>,
    /// Mean absolute reduction (baseline minus arm).
    pub absolute_reduction: Option<f64>,
    /// 95% CI of the absolute reduction.
    pub absolute_ci: Option<[f64; 2]>,
    /// Mean relative reduction.
    pub relative_reduction: Option<f64>,
    /// 95% CI of the relative reduction.
    pub relative_ci: Option<[f64; 2]>,
    /// Whether the registered 5% rule passes.
    pub passes_five_percent: bool,
}

/// The comparison report.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CompareReport {
    /// Schema identifier.
    pub schema: String,
    /// All arms grouped by realized active count.
    pub budget_groups: BTreeMap<u64, Vec<String>>,
    /// Per-arm accounting rows.
    pub arms: Vec<CompareArmReport>,
    /// Paired comparisons against the baseline.
    pub comparisons: Vec<CompareReportEntry>,
    /// True when every arm is compliant and the baseline exists.
    pub ok: bool,
    /// Ceiling used.
    pub ceiling: u64,
    /// Resamples used.
    pub resamples: usize,
}

/// Realized active unique elements per output token.
///
/// # Errors
///
/// Rejects an empty allocation.
pub fn active_unique_elements(backbone: u64, hidden: u64, arm: &CompareArm) -> Result<u64> {
    if arm.groups_per_layer.is_empty() || arm.group_width == 0 {
        return Err(AtlasError::invalid(
            "groups_per_layer",
            "must be non-empty and the group width non-zero",
        ));
    }
    let mut total = backbone;
    for count in &arm.groups_per_layer {
        total = total
            .checked_add(u64::from(*count) * 3 * u64::from(arm.group_width) * hidden)
            .ok_or_else(|| AtlasError::Budget("active count overflowed u64".into()))?;
    }
    total = total
        .checked_add(arm.controller_elements)
        .ok_or_else(|| AtlasError::Budget("active count overflowed u64".into()))?;
    Ok(total)
}

fn percentile_ci(mut values: Vec<f64>) -> Option<[f64; 2]> {
    if values.is_empty() {
        return None;
    }
    values.sort_by(|left, right| left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal));
    let last = values.len() - 1;
    let low = values[(0.025 * last as f64).round() as usize];
    let high = values[(0.975 * last as f64).round() as usize];
    Some([low, high])
}

fn bootstrap_means(values: &[f64], resamples: usize) -> (f64, Option<[f64; 2]>) {
    let count = values.len();
    if count == 0 {
        return (0.0, None);
    }
    let mean = values.iter().sum::<f64>() / count as f64;
    if resamples == 0 {
        return (mean, None);
    }
    let mut means = Vec::with_capacity(resamples);
    for resample in 0..resamples {
        let mut sum = 0.0;
        for draw in 0..count {
            let state = splitmix64(
                SEED ^ (resample as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
                    ^ (draw as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9),
            );
            let index = (state % count as u64) as usize;
            sum += values[index];
        }
        means.push(sum / count as f64);
    }
    (mean, percentile_ci(means))
}

/// Run the comparison.
///
/// # Errors
///
/// Rejects a missing baseline, an over-ceiling arm, a non-finite metric, or a
/// malformed input.
pub fn compare(input: &CompareInput, resamples: usize) -> Result<CompareReport> {
    if input.arms.is_empty() {
        return Err(AtlasError::invalid("arms", "are empty"));
    }
    let baseline = input
        .arms
        .iter()
        .find(|arm| arm.id == input.baseline)
        .ok_or_else(|| AtlasError::invalid("baseline", "is not present in the arms"))?;
    let baseline_active = active_unique_elements(input.backbone_elements, input.hidden, baseline)?;
    let mut budget_groups: BTreeMap<u64, Vec<String>> = BTreeMap::new();
    let mut reports = Vec::with_capacity(input.arms.len());
    for arm in &input.arms {
        let active = active_unique_elements(input.backbone_elements, input.hidden, arm)?;
        if active > input.ceiling {
            return Err(AtlasError::Budget(format!(
                "arm {} has {active} active parameters, above the ceiling {}",
                arm.id, input.ceiling
            )));
        }
        if arm.items.iter().any(|item| !item.kl_mean.is_finite()) {
            return Err(AtlasError::invalid("items", "contain a non-finite KL"));
        }
        budget_groups
            .entry(active)
            .or_default()
            .push(arm.id.clone());
        let delta = i64::try_from(active)
            .ok()
            .and_then(|value| i64::try_from(baseline_active).ok().map(|base| value - base))
            .ok_or_else(|| AtlasError::Budget("active count does not fit i64".into()))?;
        reports.push(CompareArmReport {
            id: arm.id.clone(),
            budget: arm.budget.clone(),
            active_unique_elements: active,
            compliant: true,
            active_delta_from_baseline: delta,
            matched: arm.budget == baseline.budget && delta.unsigned_abs() <= 6_144,
        });
    }
    let baseline_items: BTreeMap<&str, f64> = baseline
        .items
        .iter()
        .map(|item| (item.doc_id.as_str(), item.kl_mean))
        .collect();
    let mut comparisons = Vec::new();
    for arm in &input.arms {
        if arm.id == baseline.id {
            continue;
        }
        let mut shared = BTreeSet::new();
        for item in &arm.items {
            if baseline_items.contains_key(item.doc_id.as_str()) {
                shared.insert(item.doc_id.clone());
            }
        }
        let mut absolute = Vec::new();
        let mut relative = Vec::new();
        let mut base_values = Vec::new();
        let mut arm_values = Vec::new();
        for doc_id in &shared {
            let base = baseline_items
                .get(doc_id.as_str())
                .copied()
                .ok_or_else(|| AtlasError::invalid("item", "disappeared from the baseline"))?;
            let value = arm
                .items
                .iter()
                .find(|item| item.doc_id == *doc_id)
                .map(|item| item.kl_mean)
                .ok_or_else(|| AtlasError::invalid("item", "disappeared from the arm"))?;
            base_values.push(base);
            arm_values.push(value);
            absolute.push(base - value);
            relative.push(if base > 0.0 {
                (base - value) / base
            } else {
                0.0
            });
        }
        let available = absolute.len() >= 3;
        let (absolute_mean, absolute_ci) = bootstrap_means(&absolute, resamples);
        let (relative_mean, relative_ci) = bootstrap_means(&relative, resamples);
        let base_mean = if base_values.is_empty() {
            None
        } else {
            Some(base_values.iter().sum::<f64>() / base_values.len() as f64)
        };
        let arm_mean = if arm_values.is_empty() {
            None
        } else {
            Some(arm_values.iter().sum::<f64>() / arm_values.len() as f64)
        };
        comparisons.push(CompareReportEntry {
            arm: arm.id.clone(),
            baseline: baseline.id.clone(),
            items: absolute.len(),
            available,
            base_kl: base_mean,
            arm_kl: arm_mean,
            absolute_reduction: if available { Some(absolute_mean) } else { None },
            absolute_ci: if available { absolute_ci } else { None },
            relative_reduction: if available { Some(relative_mean) } else { None },
            relative_ci: if available { relative_ci } else { None },
            passes_five_percent: available
                && relative_mean >= 0.05
                && relative_ci.map(|ci| ci[0] > 0.0).unwrap_or(false),
        });
    }
    let ok = reports.iter().all(|arm| arm.compliant);
    Ok(CompareReport {
        schema: "selvedge/atlas/compare-v1".into(),
        budget_groups,
        arms: reports,
        comparisons,
        ok,
        ceiling: input.ceiling,
        resamples,
    })
}
