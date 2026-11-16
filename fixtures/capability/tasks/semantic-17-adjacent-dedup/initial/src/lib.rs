//! Collapsing through a set of seen values.

use std::collections::BTreeSet;

pub fn collapse(values: &[u8]) -> Vec<u8> {
    let mut seen = BTreeSet::new();
    values
        .iter()
        .copied()
        .filter(|value| seen.insert(*value))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::collapse;

    #[test]
    fn visible_adjacent_duplicates() {
        assert_eq!(collapse(&[1, 1, 2]), vec![1, 2]);
    }
}
