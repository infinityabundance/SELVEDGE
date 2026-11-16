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

#[cfg(test)]
mod visible {
    use super::*;

    use std::sync::Mutex;
    
    #[test]
    fn the_recorded_bytes_survive_the_panic() {
        let shared = Mutex::new(Vec::new());
        assert_eq!(record_with_panic(&shared), 1);
    }
}
