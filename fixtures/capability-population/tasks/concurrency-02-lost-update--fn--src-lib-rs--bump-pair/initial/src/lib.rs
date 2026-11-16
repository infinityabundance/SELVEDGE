//! A counter incremented atomically from two threads.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Barrier;

// TODO(FUNCTION)

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

#[cfg(test)]
mod visible {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use crate::{bump_all, bump_pair};

    #[test]
    fn pair_bumps_twice() {
        let counter = AtomicUsize::new(0);
        bump_pair(&counter);
        assert_eq!(counter.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn many_bumps_are_not_lost() {
        let counter = AtomicUsize::new(0);
        bump_all(&counter, 4, 500);
        assert_eq!(counter.load(Ordering::SeqCst), 2000);
    }

    #[test]
    fn bumps_start_from_the_current_value() {
        let counter = AtomicUsize::new(41);
        bump_pair(&counter);
        assert_eq!(counter.load(Ordering::SeqCst), 43);
    }
}
