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
    use super::*;

    #[test]
    fn counts_overlapping_matches() {
        assert_eq!(count_occurrences("aaaa", "aa"), 3);
    }
}
