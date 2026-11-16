//! Largest gap between consecutive values.

pub fn max_gap(values: &[i64]) -> i64 {
    values
        .windows(2)
        .map(|pair| pair[1] - pair[0])
        .max()
        .unwrap_or(0)
}
