//! Overlapping occurrence counting.

pub fn count_occurrences(haystack: &str, needle: &str) -> usize {
    if needle.is_empty() {
        return 0;
    }
    haystack
        .as_bytes()
        .windows(needle.len())
        .filter(|window| *window == needle.as_bytes())
        .count()
}

#[cfg(test)]
mod visible {
    use crate::count_occurrences;

    #[test]
    fn counts_overlapping_matches() {
        assert_eq!(count_occurrences("aaaa", "aa"), 3);
    }

    #[test]
    fn counts_every_aba_in_ababa() {
        assert_eq!(count_occurrences("ababa", "aba"), 2);
    }

    #[test]
    fn counts_single_characters() {
        assert_eq!(count_occurrences("aaa", "a"), 3);
    }

    #[test]
    fn no_match_is_zero() {
        assert_eq!(count_occurrences("abc", "d"), 0);
    }

    #[test]
    fn an_empty_needle_is_zero() {
        assert_eq!(count_occurrences("abc", ""), 0);
    }
}
