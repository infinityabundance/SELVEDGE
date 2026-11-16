//! Occurrence counting with the standard non-overlapping matcher.

pub fn count_occurrences(haystack: &str, needle: &str) -> usize {
    haystack.matches(needle).count()
}

#[cfg(test)]
mod tests {
    use super::count_occurrences;

    #[test]
    fn visible_example_is_non_overlapping() {
        assert_eq!(count_occurrences("aaaa", "aa"), 2);
    }
}
