//! A mutex-protected counter.

use std::sync::{Barrier, Mutex};

pub fn bump_pair(counter: &Mutex<u64>) {
    let barrier = Barrier::new(2);
    std::thread::scope(|scope| {
        for _ in 0..2 {
            scope.spawn(|| {
                barrier.wait();
                let mut guard = counter.lock().unwrap();
                *guard += 1;
            });
        }
    });
}

pub fn bump_many(counter: &Mutex<u64>, threads: usize, per_thread: usize) {
    std::thread::scope(|scope| {
        for _ in 0..threads {
            scope.spawn(|| {
                for _ in 0..per_thread {
                    let mut guard = counter.lock().unwrap();
                    *guard += 1;
                }
            });
        }
    });
}
