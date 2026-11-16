//! Accumulate values into a counter.

use std::sync::atomic::{AtomicUsize, Ordering};

// TODO(FUNCTION)

#[cfg(test)]
mod visible {
    use std::sync::atomic::AtomicUsize;

    use crate::accumulate;

    #[test]
    fn repeated_values_add() {
        let counter = AtomicUsize::new(0);
        assert_eq!(accumulate(&counter, &[1, 1]), 2);
    }

    #[test]
    fn distinct_values_add() {
        let counter = AtomicUsize::new(0);
        assert_eq!(accumulate(&counter, &[2, 3]), 5);
    }

    #[test]
    fn the_starting_value_is_kept() {
        let counter = AtomicUsize::new(10);
        assert_eq!(accumulate(&counter, &[8, 1]), 19);
    }
}
