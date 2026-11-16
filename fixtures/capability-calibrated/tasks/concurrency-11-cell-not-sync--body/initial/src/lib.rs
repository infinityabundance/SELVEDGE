//! Count from several threads.

use std::sync::atomic::{AtomicU64, Ordering};

pub fn count_with(counter: &AtomicU64, threads: usize) -> u64 {
    todo!()
}

#[cfg(test)]
mod visible {
    use super::*;

    use std::sync::atomic::{AtomicU64, Ordering};
    
    #[test]
    fn every_thread_adds_one() {
        let counter = AtomicU64::new(0);
        assert_eq!(count_with(&counter, 4), 4);
        assert_eq!(counter.load(Ordering::SeqCst), 4);
    }
}
