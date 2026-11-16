//! Push from a spawned thread.

use std::sync::{Arc, Mutex};
use std::thread;

pub fn push_from_thread(values: &Arc<Mutex<Vec<i32>>>, value: i32) {
    let mut guard = values.lock().unwrap();
    thread::spawn(move || {
        guard.push(value);
    })
    .join()
    .unwrap();
}
