//! Overlapping occurrence counting.

pub fn count_occurrences(haystack: &str, needle: &str) -> usize {
    todo!()
}

#[cfg(test)]
mod visible {
    use super::*;

    #[test]
    fn counts_overlapping_matches() {
        assert_eq!(count_occurrences("aaaa", "aa"), 3);
    }
}
