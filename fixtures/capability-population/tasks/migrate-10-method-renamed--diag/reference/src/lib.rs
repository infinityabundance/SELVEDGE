//! Running totals through the tally crate.

pub fn running_total(values: &[i64]) -> i64 {
    let mut tally = stats::Tally::new();
    for value in values {
        tally.push(*value);
    }
    tally.total()
}
