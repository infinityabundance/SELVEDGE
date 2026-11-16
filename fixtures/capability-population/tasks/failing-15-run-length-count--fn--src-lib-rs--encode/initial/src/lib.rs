//! Run-length encoding.

// TODO(FUNCTION)

#[cfg(test)]
mod visible {
    use crate::encode;

    #[test]
    fn every_run_carries_its_length() {
        assert_eq!(encode("aab"), "a2b1");
        assert_eq!(encode("abbcccd"), "a1b2c3d1");
    }

    #[test]
    fn a_single_character_is_a_run_of_one() {
        assert_eq!(encode("a"), "a1");
    }

    #[test]
    fn empty_input_is_empty() {
        assert_eq!(encode(""), "");
    }

    #[test]
    fn a_long_run_is_counted_exactly() {
        assert_eq!(encode("aaaa"), "a4");
    }
}
