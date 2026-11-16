//! Order-preserving de-duplication.

use std::collections::HashSet;

pub fn unique(values: &[i32]) -> Vec<i32> {
    let mut seen = HashSet::new();
    let mut result = Vec::new();
    for value in values {
        if seen.insert(*value) {
            result.push(*value);
        }
    }
    result
}
