//! Overlapping occurrence counting.

// TODO(FUNCTION)

#[cfg(test)]
mod visible {
    use super::*;

    #[test]
    fn counts_overlapping_matches() {
        assert_eq!(count_occurrences("aaaa", "aa"), 3);
    }
}
