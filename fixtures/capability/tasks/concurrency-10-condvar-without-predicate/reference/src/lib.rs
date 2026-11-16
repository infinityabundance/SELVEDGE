//! A readiness signal.

use std::sync::{Arc, Condvar, Mutex};

pub fn wait_for_ready(state: Arc<(Mutex<bool>, Condvar)>) -> bool {
    let (lock, ready) = &*state;
    let mut guard = lock.lock().unwrap();
    while !*guard {
        guard = ready.wait(guard).unwrap();
    }
    *guard
}

pub fn run() -> bool {
    let state = Arc::new((Mutex::new(false), Condvar::new()));
    {
        let (lock, ready) = &*state;
        let mut flag = lock.lock().unwrap();
        *flag = true;
        ready.notify_all();
    }
    wait_for_ready(state)
}
