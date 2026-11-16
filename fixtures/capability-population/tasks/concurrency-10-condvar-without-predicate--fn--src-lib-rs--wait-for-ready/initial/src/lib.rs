//! A readiness signal.

use std::sync::{Arc, Condvar, Mutex};

// TODO(FUNCTION)

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
mod visible {
    use std::sync::{Arc, Condvar, Mutex};

    use crate::{run, wait_for_ready};

    #[test]
    fn a_signal_before_the_wait_is_not_lost() {
        assert!(run());
    }

    #[test]
    fn an_already_ready_state_returns_immediately() {
        let state = Arc::new((Mutex::new(true), Condvar::new()));
        assert!(wait_for_ready(state));
    }

    #[test]
    fn a_late_signal_wakes_the_waiter() {
        let state = Arc::new((Mutex::new(false), Condvar::new()));
        let clone = Arc::clone(&state);
        let handle = std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(20));
            let (lock, ready) = &*clone;
            let mut flag = lock.lock().unwrap();
            *flag = true;
            ready.notify_all();
        });
        assert!(wait_for_ready(state));
        handle.join().unwrap();
    }
}
