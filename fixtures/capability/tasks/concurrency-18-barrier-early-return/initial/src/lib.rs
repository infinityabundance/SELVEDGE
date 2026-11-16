//! A barrier-guarded sum. The crate builds, but the barrier is sized
//! for workers that leave before it.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Barrier;

pub fn sum_nonzero(values: &[u8]) -> usize {
    let barrier = Barrier::new(values.len());
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
mod tests {
    #[test]
    fn visible_with_zero() {
        assert_eq!(super::sum_nonzero(&[1, 0, 2]), 3);
    }
}
