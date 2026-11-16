//! Truncation by character count.

// TODO(FUNCTION)

#[cfg(test)]
mod visible {
    use crate::truncate;

    #[test]
    fn truncates_ascii_exactly() {
        assert_eq!(truncate("abcdef", 3), "abc");
    }

    #[test]
    fn does_not_split_a_multibyte_character() {
        assert_eq!(truncate("héllo", 2), "hé");
    }

    #[test]
    fn zero_characters_is_empty() {
        assert_eq!(truncate("abc", 0), "");
    }

    #[test]
    fn a_longer_limit_returns_everything() {
        assert_eq!(truncate("hé", 10), "hé");
    }
}
