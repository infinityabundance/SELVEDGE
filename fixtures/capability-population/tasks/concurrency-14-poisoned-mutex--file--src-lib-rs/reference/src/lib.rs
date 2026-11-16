//! Record bytes and survive a worker panic.

use std::sync::Mutex;

pub fn record_with_panic(shared: &Mutex<Vec<u8>>) -> usize {
    let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        std::thread::scope(|scope| {
            scope.spawn(|| {
                let mut guard = shared.lock().unwrap();
                guard.push(7);
                panic!("worker failed");
            });
        });
    }));
    let _ = caught;
    let guard = shared
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    guard.len()
}
