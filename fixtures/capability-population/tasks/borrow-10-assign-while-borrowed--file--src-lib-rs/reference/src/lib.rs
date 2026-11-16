//! Compare the previous value with the last reading.

pub fn compare_with_last(current: i64, readings: &[i64]) -> i64 {
    let previous = current;
    let last = readings.last().copied().unwrap_or(current);
    previous + last
}
