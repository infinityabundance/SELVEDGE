//! Two synchronized increments.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Barrier};
use std::thread;

pub fn increment_pair() -> usize {
    let counter = Arc::new(AtomicUsize::new(0));
    let barrier = Arc::new(Barrier::new(2));
    let handles: Vec<_> = (0..2)
        .map(|_| {
            let counter = Arc::clone(&counter);
            let barrier = Arc::clone(&barrier);
            thread::spawn(move || {
                barrier.wait();
                counter.fetch_add(1, Ordering::SeqCst);
            })
        })
        .collect();
    for handle in handles {
        handle.join().unwrap();
    }
    counter.load(Ordering::SeqCst)
}
