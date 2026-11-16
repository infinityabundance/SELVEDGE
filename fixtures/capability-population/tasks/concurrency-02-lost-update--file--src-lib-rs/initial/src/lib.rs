//! A counter incremented from two threads.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Barrier;

pub fn bump_pair(counter: &AtomicUsize) {
    let barrier = Barrier::new(2);
    std::thread::scope(|scope| {
        for _ in 0..2 {
            scope.spawn(|| {
                let value = counter.load(Ordering::SeqCst);
                barrier.wait();
                counter.store(value + 1, Ordering::SeqCst);
            });
        }
    });
}

pub fn bump_all(counter: &AtomicUsize, threads: usize, per_thread: usize) {
    std::thread::scope(|scope| {
        for _ in 0..threads {
            scope.spawn(|| {
                for _ in 0..per_thread {
                    let value = counter.load(Ordering::Relaxed);
                    std::hint::spin_loop();
                    counter.store(value + 1, Ordering::Relaxed);
                }
            });
        }
    });
}
