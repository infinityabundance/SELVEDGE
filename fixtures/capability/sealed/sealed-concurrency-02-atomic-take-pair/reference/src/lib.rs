//! Two concurrent atomic drains.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Barrier;

pub fn take_pair(counter: &AtomicUsize) -> usize {
    let barrier = Barrier::new(2);
    std::thread::scope(|scope| {
        let first = scope.spawn(|| {
            let value = counter.swap(0, Ordering::SeqCst);
            barrier.wait();
            value
        });
        let second = scope.spawn(|| {
            let value = counter.swap(0, Ordering::SeqCst);
            barrier.wait();
            value
        });
        first.join().unwrap() + second.join().unwrap()
    })
}
