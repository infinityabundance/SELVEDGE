//! Read a mutex-protected value twice.

use std::sync::Mutex;

pub fn read_twice(shared: &Mutex<u32>) -> u32 {
    let value = *shared.lock().unwrap();
    value * 2
}
