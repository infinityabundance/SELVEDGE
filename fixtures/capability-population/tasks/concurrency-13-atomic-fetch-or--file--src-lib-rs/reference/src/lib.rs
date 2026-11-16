//! Accumulate values into a counter.

use std::sync::atomic::{AtomicUsize, Ordering};

pub fn accumulate(counter: &AtomicUsize, values: &[usize]) -> usize {
    for value in values {
        counter.fetch_add(*value, Ordering::SeqCst);
    }
    counter.load(Ordering::SeqCst)
}
