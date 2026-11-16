//! Deterministic randomness for the simulator.
//!
//! The simulator must be reproducible from a seed alone, so it uses a small
//! explicit `SplitMix64` generator rather than a platform generator. Sampling is
//! inverse-transform over a sorted empirical sample (or over an explicit
//! parametric quantile function), so a given seed and a given sample always
//! produce the same value on every platform.

use crate::error::{Result, SimError};

/// `SplitMix64`: a small, well-known, fully specified 64-bit generator.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    /// Create a generator from a seed.
    #[must_use]
    pub const fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Next 64-bit value.
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform value in `[0, 1)`, exactly 53 random bits.
    pub fn next_f64(&mut self) -> f64 {
        // 2^-53 keeps the value strictly below one.
        (self.next_u64() >> 11) as f64 * (1.0 / 9_007_199_254_740_992.0)
    }

    /// Uniform integer in `[0, n)`; `n == 0` is rejected.
    ///
    /// # Errors
    ///
    /// Rejects a zero bound.
    pub fn below(&mut self, n: u64) -> Result<u64> {
        if n == 0 {
            return Err(SimError::invalid("rng_bound", "must be non-zero"));
        }
        // Rejection sampling removes the modulo bias and is deterministic.
        let threshold = n.wrapping_neg() % n;
        loop {
            let value = self.next_u64();
            if value >= threshold {
                return Ok(value % n);
            }
        }
    }
}

/// Linear-interpolated empirical quantile of a sorted sample.
///
/// `q` is clamped to `[0, 1]`. An empty sample is rejected.
///
/// # Errors
///
/// Rejects an empty sample.
pub fn empirical_quantile(sorted: &[u64], q: f64) -> Result<f64> {
    if sorted.is_empty() {
        return Err(SimError::invalid("sample", "is empty"));
    }
    let q = q.clamp(0.0, 1.0);
    if sorted.len() == 1 {
        return Ok(sorted[0] as f64);
    }
    let position = q * (sorted.len() - 1) as f64;
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    let low = sorted[lower] as f64;
    let high = sorted[upper] as f64;
    Ok(low + (high - low) * (position - lower as f64))
}

/// Percentile of a sorted `u64` sample, returned as the raw value.
///
/// Uses the nearest-rank definition: the smallest element whose rank covers the
/// requested percentile. An empty sample yields zero, which callers must label.
#[must_use]
pub fn percentile_u64(sorted: &[u64], percentile: f64) -> u64 {
    if sorted.is_empty() {
        return 0;
    }
    let q = (percentile / 100.0).clamp(0.0, 1.0);
    let rank = (q * sorted.len() as f64).ceil() as usize;
    let index = rank.saturating_sub(1).min(sorted.len() - 1);
    sorted[index]
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn splitmix_is_reproducible() {
        let mut a = SplitMix64::new(7);
        let mut b = SplitMix64::new(7);
        for _ in 0..32 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
        let mut c = SplitMix64::new(8);
        assert_ne!(SplitMix64::new(7).next_u64(), c.next_u64());
    }

    #[test]
    fn uniform_is_in_unit_interval() {
        let mut rng = SplitMix64::new(1);
        for _ in 0..10_000 {
            let value = rng.next_f64();
            assert!((0.0..1.0).contains(&value));
        }
    }

    #[test]
    fn below_is_bounded_and_rejects_zero() {
        let mut rng = SplitMix64::new(2);
        for _ in 0..1000 {
            assert!(rng.below(7).unwrap() < 7);
        }
        assert!(rng.below(0).is_err());
    }

    #[test]
    fn empirical_quantiles_include_endpoints() {
        let sample = [10_u64, 20, 30, 40, 50];
        assert_eq!(empirical_quantile(&sample, 0.0).unwrap(), 10.0);
        assert_eq!(empirical_quantile(&sample, 1.0).unwrap(), 50.0);
        assert_eq!(empirical_quantile(&sample, 0.5).unwrap(), 30.0);
        assert_eq!(empirical_quantile(&sample, 0.25).unwrap(), 20.0);
        assert!(empirical_quantile(&[], 0.5).is_err());
        assert_eq!(empirical_quantile(&[9], 0.9).unwrap(), 9.0);
    }

    #[test]
    fn nearest_rank_percentiles() {
        let sample: Vec<u64> = (1..=100).collect();
        assert_eq!(percentile_u64(&sample, 50.0), 50);
        assert_eq!(percentile_u64(&sample, 95.0), 95);
        assert_eq!(percentile_u64(&sample, 99.0), 99);
        assert_eq!(percentile_u64(&sample, 100.0), 100);
        assert_eq!(percentile_u64(&[], 50.0), 0);
    }
}
