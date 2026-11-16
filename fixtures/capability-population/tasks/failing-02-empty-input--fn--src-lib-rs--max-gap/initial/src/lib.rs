//! Largest gap between consecutive values.

// TODO(FUNCTION)

#[cfg(test)]
mod visible {
    use crate::max_gap;

    #[test]
    fn empty_slice() {
        assert_eq!(max_gap(&[]), 0);
    }

    #[test]
    fn single_value() {
        assert_eq!(max_gap(&[5]), 0);
    }

    #[test]
    fn largest_gap_wins() {
        assert_eq!(max_gap(&[1, 4, 2, 9]), 7);
    }

    #[test]
    fn negative_gaps_are_reported() {
        assert_eq!(max_gap(&[9, 1]), -8);
    }
}
