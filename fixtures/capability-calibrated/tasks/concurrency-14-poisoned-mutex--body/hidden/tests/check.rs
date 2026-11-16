use std::sync::Mutex;

use concurrency_14_poisoned_mutex::record_with_panic;

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
