//! Reliability/calibration checks for the two-view head.
//!
//! A selector that emits a value estimate must be calibrated: higher stated
//! confidence should correspond to a higher observed success rate. These
//! helpers are deliberately small; they test the property, they do not fit a
//! calibrator.

use serde::{Deserialize, Serialize};

use crate::error::{AtlasError, Result};

/// One reliability bin.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ReliabilityPoint {
    /// Mean stated confidence.
    pub confidence: f64,
    /// Observed success rate.
    pub accuracy: f64,
    /// Bin population.
    pub count: usize,
}

/// Build a reliability curve from `(confidence, success)` pairs.
///
/// # Errors
///
/// Rejects empty input, non-finite values, confidences outside `[0, 1]`, or a
/// zero bin count.
pub fn reliability_curve(pairs: &[(f64, bool)], bins: usize) -> Result<Vec<ReliabilityPoint>> {
    if pairs.is_empty() {
        return Err(AtlasError::invalid("pairs", "are empty"));
    }
    if bins == 0 {
        return Err(AtlasError::invalid("bins", "must be positive"));
    }
    for (confidence, _) in pairs {
        if !confidence.is_finite() || *confidence < 0.0 || *confidence > 1.0 {
            return Err(AtlasError::invalid(
                "confidence",
                "must be finite and in [0, 1]",
            ));
        }
    }
    let mut buckets: Vec<Vec<(f64, bool)>> = vec![Vec::new(); bins];
    for (confidence, success) in pairs {
        let mut index = (*confidence * bins as f64).floor() as usize;
        if index >= bins {
            index = bins - 1;
        }
        buckets[index].push((*confidence, *success));
    }
    let mut points = Vec::new();
    for bucket in buckets {
        if bucket.is_empty() {
            continue;
        }
        let confidence = bucket.iter().map(|(value, _)| *value).sum::<f64>() / bucket.len() as f64;
        let success = bucket.iter().filter(|(_, hit)| *hit).count() as f64 / bucket.len() as f64;
        points.push(ReliabilityPoint {
            confidence,
            accuracy: success,
            count: bucket.len(),
        });
    }
    Ok(points)
}

/// Whether accuracy is non-decreasing with confidence.
#[must_use]
pub fn is_monotone(points: &[ReliabilityPoint]) -> bool {
    points
        .windows(2)
        .all(|pair| pair[0].accuracy <= pair[1].accuracy + 1e-12)
}

/// Calibration summary.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CalibrationReport {
    /// Reliability bins.
    pub points: Vec<ReliabilityPoint>,
    /// Monotonicity result.
    pub monotone: bool,
}

/// Build the report.
///
/// # Errors
///
/// Propagates [`reliability_curve`] errors.
pub fn report(pairs: &[(f64, bool)], bins: usize) -> Result<CalibrationReport> {
    let points = reliability_curve(pairs, bins)?;
    let monotone = is_monotone(&points);
    Ok(CalibrationReport { points, monotone })
}
