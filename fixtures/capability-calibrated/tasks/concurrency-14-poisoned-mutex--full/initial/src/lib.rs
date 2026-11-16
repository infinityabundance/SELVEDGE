//! Record bytes and survive a worker panic. The visible test fails:
//! the poisoned mutex is unwrapped instead of recovered.

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
    let guard = shared.lock().unwrap();
    guard.len()
}

#[cfg(test)]
mod tests {
    use super::record_with_panic;
    use std::sync::Mutex;

    #[test]
    fn visible_after_panic() {
        let shared = Mutex::new(Vec::new());
        assert_eq!(record_with_panic(&shared), 1);
    }
}
