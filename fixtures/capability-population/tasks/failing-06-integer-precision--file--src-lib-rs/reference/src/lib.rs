//! Percentage arithmetic.

pub fn percent_of(cents: u64, percent: u64) -> u64 {
    cents * percent / 100
}
