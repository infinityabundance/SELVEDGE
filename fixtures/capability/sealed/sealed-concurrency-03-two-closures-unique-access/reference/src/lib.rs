//! Two scoped atomic increments.

use std::sync::atomic::{AtomicU32, Ordering};

pub fn bump_both(counter: &AtomicU32) -> u32 {
    std::thread::scope(|scope| {
        scope.spawn(|| {
            counter.fetch_add(1, Ordering::SeqCst);
        });
        scope.spawn(|| {
            counter.fetch_add(1, Ordering::SeqCst);
        });
    });
    counter.load(Ordering::SeqCst)
}
