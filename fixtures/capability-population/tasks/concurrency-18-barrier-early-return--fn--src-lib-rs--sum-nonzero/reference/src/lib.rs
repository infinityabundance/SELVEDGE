//! A barrier-guarded sum.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Barrier;

pub fn sum_nonzero(values: &[u8]) -> usize {
    let participants = values.iter().filter(|value| **value != 0).count().max(1);
    let barrier = Barrier::new(participants);
    let total = AtomicUsize::new(0);
    std::thread::scope(|scope| {
        for value in values {
            scope.spawn(|| {
                if *value == 0 {
                    return;
                }
                barrier.wait();
                total.fetch_add(*value as usize, Ordering::SeqCst);
            });
        }
    });
    total.load(Ordering::SeqCst)
}

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
