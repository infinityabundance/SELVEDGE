//! Count from several threads.

use std::sync::atomic::{AtomicU64, Ordering};

pub fn count_with(counter: &AtomicU64, threads: usize) -> u64 {
    std::thread::scope(|scope| {
        for _ in 0..threads {
            scope.spawn(|| {
                counter.fetch_add(1, Ordering::SeqCst);
            });
        }
    });
    counter.load(Ordering::SeqCst)
}
