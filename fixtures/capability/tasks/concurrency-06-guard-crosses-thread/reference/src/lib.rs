//! Push from a spawned thread.

use std::sync::{Arc, Mutex};
use std::thread;

pub fn push_from_thread(values: &Arc<Mutex<Vec<i32>>>, value: i32) {
    let values = Arc::clone(values);
    thread::spawn(move || {
        values.lock().unwrap().push(value);
    })
    .join()
    .unwrap();
}
