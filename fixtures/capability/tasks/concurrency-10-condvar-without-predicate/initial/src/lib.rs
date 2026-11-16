//! A readiness signal. The crate builds, but `run` never returns: the
//! notification is sent before the waiter parks.

use std::sync::{Arc, Condvar, Mutex};

pub fn wait_for_ready(state: Arc<(Mutex<bool>, Condvar)>) -> bool {
    let (lock, ready) = &*state;
    let guard = lock.lock().unwrap();
    let _released = ready.wait(guard).unwrap();
    true
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

#[cfg(test)]
mod tests {
    #[test]
    fn visible_run_returns() {
        assert!(super::run());
    }
}
