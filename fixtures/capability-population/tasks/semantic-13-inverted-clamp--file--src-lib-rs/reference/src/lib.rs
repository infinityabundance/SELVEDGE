//! Clamping with an inverted-bound rule.

pub fn clamp(value: i64, low: i64, high: i64) -> i64 {
    if low > high {
        return low;
    }
    value.max(low).min(high)
}
