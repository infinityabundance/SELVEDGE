//! Two concurrent drains. The visible test fails: the read and the store
//! are separate operations.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Barrier;

pub fn take_pair(counter: &AtomicUsize) -> usize {
    let barrier = Barrier::new(2);
    std::thread::scope(|scope| {
        let first = scope.spawn(|| {
            let value = counter.load(Ordering::SeqCst);
            barrier.wait();
            counter.store(0, Ordering::SeqCst);
            value
        });
        let second = scope.spawn(|| {
            let value = counter.load(Ordering::SeqCst);
            barrier.wait();
            counter.store(0, Ordering::SeqCst);
            value
        });
        first.join().unwrap() + second.join().unwrap()
    })
}

#[cfg(test)]
mod tests {
    use super::take_pair;
    use std::sync::atomic::AtomicUsize;

    #[test]
    fn visible_pair() {
        let counter = AtomicUsize::new(5);
        assert_eq!(take_pair(&counter), 5);
    }
}
