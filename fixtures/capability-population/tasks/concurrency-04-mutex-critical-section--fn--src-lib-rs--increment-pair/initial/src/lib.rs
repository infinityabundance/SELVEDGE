//! Two synchronized increments under one lock.

use std::sync::{Arc, Barrier, Mutex};
use std::thread;

// TODO(FUNCTION)

#[cfg(test)]
mod visible {
    use crate::increment_pair;
    use std::sync::{Arc, Mutex};

    #[test]
    fn both_threads_increment() {
        let counter = Arc::new(Mutex::new(0));
        increment_pair(&counter);
        assert_eq!(*counter.lock().unwrap(), 2);
    }

    #[test]
    fn the_result_is_stable_across_runs() {
        for _ in 0..20 {
            let counter = Arc::new(Mutex::new(0));
            increment_pair(&counter);
            assert_eq!(*counter.lock().unwrap(), 2);
        }
    }

    #[test]
    fn increments_from_the_current_value() {
        let counter = Arc::new(Mutex::new(41));
        increment_pair(&counter);
        assert_eq!(*counter.lock().unwrap(), 43);
    }
}
