//! Compare the previous value with the last reading. The crate does
//! not build: `current` is assigned while a reference to it is alive.

pub fn compare_with_last(current: i64, readings: &[i64]) -> i64 {
    let mut current = current;
    let previous = &current;
    current = readings.last().copied().unwrap_or(current);
    *previous + current
}
