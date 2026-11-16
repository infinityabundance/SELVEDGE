//! Inclusive range sums.

pub fn range_sum(start: i32, end: i32) -> i32 {
    (start..=end).sum()
}
