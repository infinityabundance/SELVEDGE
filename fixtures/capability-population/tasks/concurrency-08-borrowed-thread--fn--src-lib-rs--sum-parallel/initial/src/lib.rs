//! Sum two-element chunks in parallel.

use std::thread;

// TODO(FUNCTION)

#[cfg(test)]
mod visible {
    use crate::sum_parallel;

    #[test]
    fn sums_every_value() {
        assert_eq!(sum_parallel(&[1, 2, 3, 4, 5]), 15);
    }

    #[test]
    fn an_empty_slice_is_zero() {
        assert_eq!(sum_parallel(&[]), 0);
    }

    #[test]
    fn a_single_value_is_itself() {
        assert_eq!(sum_parallel(&[7]), 7);
    }

    #[test]
    fn negative_values_are_summed() {
        assert_eq!(sum_parallel(&[-3, 4, -5]), -4);
    }
}
