//! A counter incremented atomically from two threads.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Barrier;

pub fn bump_pair(counter: &AtomicUsize) {
    let barrier = Barrier::new(2);
    std::thread::scope(|scope| {
        for _ in 0..2 {
            scope.spawn(|| {
                barrier.wait();
                counter.fetch_add(1, Ordering::SeqCst);
            });
        }
    });
}

pub fn bump_all(counter: &AtomicUsize, threads: usize, per_thread: usize) {
    std::thread::scope(|scope| {
        for _ in 0..threads {
            scope.spawn(|| {
                for _ in 0..per_thread {
                    counter.fetch_add(1, Ordering::Relaxed);
                }
            });
        }
    });
}
