//! A barrier-guarded sum.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Barrier;

// TODO(FUNCTION)

#[cfg(test)]
mod visible {
    use crate::sum_nonzero;

    #[test]
    fn a_zero_is_ignored() {
        assert_eq!(sum_nonzero(&[1, 0, 2]), 3);
    }

    #[test]
    fn every_value_is_summed() {
        assert_eq!(sum_nonzero(&[2, 3]), 5);
    }

    #[test]
    fn all_zeroes_sum_to_zero() {
        assert_eq!(sum_nonzero(&[0, 0]), 0);
    }

    #[test]
    fn empty_input_sums_to_zero() {
        assert_eq!(sum_nonzero(&[]), 0);
    }
}
