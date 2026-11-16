//! Totals over parsed lines.

use crate::model::load;

pub fn run(lines: &[&str]) -> i32 {
    load(lines).iter().map(|record| record.value).sum()
}
