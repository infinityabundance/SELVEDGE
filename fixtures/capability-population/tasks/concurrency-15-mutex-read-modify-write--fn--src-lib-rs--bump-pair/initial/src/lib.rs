//! A mutex-protected counter.

use std::sync::{Barrier, Mutex};

// TODO(FUNCTION)

pub fn bump_many(counter: &Mutex<u64>, threads: usize, per_thread: usize) {
    std::thread::scope(|scope| {
        for _ in 0..threads {
            scope.spawn(|| {
                for _ in 0..per_thread {
                    let mut guard = counter.lock().unwrap();
                    *guard += 1;
                }
            });
        }
    });
}

#[cfg(test)]
mod visible {
    use std::sync::Mutex;

    use crate::{bump_many, bump_pair};

    #[test]
    fn a_pair_adds_two() {
        let counter = Mutex::new(0);
        bump_pair(&counter);
        assert_eq!(*counter.lock().unwrap(), 2);
    }

    #[test]
    fn many_updates_are_not_lost() {
        let counter = Mutex::new(0);
        bump_many(&counter, 4, 250);
        assert_eq!(*counter.lock().unwrap(), 1000);
    }

    #[test]
    fn updates_start_from_the_current_value() {
        let counter = Mutex::new(41);
        bump_pair(&counter);
        assert_eq!(*counter.lock().unwrap(), 43);
    }
}
