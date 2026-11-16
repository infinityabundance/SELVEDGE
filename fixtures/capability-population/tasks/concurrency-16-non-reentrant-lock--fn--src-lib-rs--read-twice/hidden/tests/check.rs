use std::sync::Mutex;

use concurrency_16_non_reentrant_lock::read_twice;

#[test]
fn doubles_the_value() {
    let shared = Mutex::new(21);
    assert_eq!(read_twice(&shared), 42);
}

#[test]
fn zero_doubles_to_zero() {
    let shared = Mutex::new(0);
    assert_eq!(read_twice(&shared), 0);
}

#[test]
fn the_lock_is_released_afterwards() {
    let shared = Mutex::new(4);
    assert_eq!(read_twice(&shared), 8);
    assert_eq!(*shared.lock().unwrap(), 4);
}
