//! Running totals through the tally crate. This does not build: the
//! method was renamed in version 2.

pub fn running_total(values: &[i64]) -> i64 {
    let mut tally = stats::Tally::new();
    for value in values {
        tally.push(*value);
    }
    tally.sum()
}
