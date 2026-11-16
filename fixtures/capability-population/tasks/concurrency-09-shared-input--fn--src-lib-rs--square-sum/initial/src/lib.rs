//! Sum of squares, computed twice by two workers.

use std::sync::Arc;
use std::thread;

// TODO(FUNCTION)

#[cfg(test)]
mod visible {
    use crate::square_sum;

    #[test]
    fn sums_squares() {
        assert_eq!(square_sum(vec![1, 2, 3]), 14);
    }

    #[test]
    fn empty_input_is_zero() {
        assert_eq!(square_sum(Vec::new()), 0);
    }

    #[test]
    fn negative_values_square_positively() {
        assert_eq!(square_sum(vec![-2, 3]), 13);
    }

    #[test]
    fn zeroes_contribute_nothing() {
        assert_eq!(square_sum(vec![0, 0, 2]), 4);
    }
}
