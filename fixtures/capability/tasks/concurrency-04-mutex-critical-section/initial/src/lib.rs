//! Two synchronized increments with a mutex, and a lost update.

use std::sync::{Arc, Barrier, Mutex};
use std::thread;

pub fn increment_pair(counter: &Arc<Mutex<usize>>) {
    let barrier = Arc::new(Barrier::new(2));
    thread::scope(|scope| {
        for _ in 0..2 {
            let counter = Arc::clone(counter);
            let barrier = Arc::clone(&barrier);
            scope.spawn(move || {
                let current = *counter.lock().unwrap();
                barrier.wait();
                *counter.lock().unwrap() = current + 1;
            });
        }
    });
}
