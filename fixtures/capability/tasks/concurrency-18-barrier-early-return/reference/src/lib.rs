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
