//! Sum a vector across worker threads with a thread-safe pointer.

use std::sync::Arc;

// TODO(FUNCTION)

#[cfg(test)]
mod visible {
    use crate::total_in_threads;

    #[test]
    fn sums_every_element() {
        assert_eq!(total_in_threads(vec![1, 2, 3, 4, 5]), 15);
    }

    #[test]
    fn empty_is_zero() {
        assert_eq!(total_in_threads(vec![]), 0);
    }

    #[test]
    fn one_element() {
        assert_eq!(total_in_threads(vec![7]), 7);
    }

    #[test]
    fn negative_values() {
        assert_eq!(total_in_threads(vec![-4, 1, 3]), 0);
    }
}
