//! Two synchronized increments under one lock.

use std::sync::{Arc, Barrier, Mutex};
use std::thread;

pub fn increment_pair(counter: &Arc<Mutex<usize>>) {
    let barrier = Arc::new(Barrier::new(2));
    thread::scope(|scope| {
        for _ in 0..2 {
            let counter = Arc::clone(counter);
            let barrier = Arc::clone(&barrier);
            scope.spawn(move || {
                barrier.wait();
                *counter.lock().unwrap() += 1;
            });
        }
    });
}
