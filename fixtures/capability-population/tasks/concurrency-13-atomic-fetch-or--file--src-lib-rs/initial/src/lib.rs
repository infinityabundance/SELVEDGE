//! Accumulate values into a counter. The visible test fails: the
//! counter is combined with OR instead of addition.

use std::sync::atomic::{AtomicUsize, Ordering};

pub fn accumulate(counter: &AtomicUsize, values: &[usize]) -> usize {
    for value in values {
        counter.fetch_or(*value, Ordering::SeqCst);
    }
    counter.load(Ordering::SeqCst)
}

#[cfg(test)]
mod tests {
    use super::accumulate;
    use std::sync::atomic::AtomicUsize;

    #[test]
    fn visible_pair() {
        let counter = AtomicUsize::new(0);
        assert_eq!(accumulate(&counter, &[1, 1]), 2);
    }
}
