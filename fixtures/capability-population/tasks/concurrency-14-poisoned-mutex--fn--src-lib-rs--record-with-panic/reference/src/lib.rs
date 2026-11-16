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
    use std::sync::Mutex;

    use crate::record_with_panic;

    #[test]
    fn the_recorded_bytes_survive_the_panic() {
        let shared = Mutex::new(Vec::new());
        assert_eq!(record_with_panic(&shared), 1);
    }

    #[test]
    fn the_worker_byte_is_visible_afterwards() {
        let shared = Mutex::new(Vec::new());
        record_with_panic(&shared);
        let guard = shared.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        assert_eq!(*guard, vec![7]);
    }

    #[test]
    fn existing_bytes_are_kept() {
        let shared = Mutex::new(vec![1, 2]);
        assert_eq!(record_with_panic(&shared), 3);
    }
}
