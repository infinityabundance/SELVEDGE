//! Two accounts with one global lock order.

use std::sync::{Barrier, Mutex, MutexGuard};

fn lock_pair<'a>(
    a: &'a Mutex<i64>,
    b: &'a Mutex<i64>,
) -> (MutexGuard<'a, i64>, MutexGuard<'a, i64>) {
    if (a as *const Mutex<i64>) < (b as *const Mutex<i64>) {
        let guard_a = a.lock().unwrap();
        let guard_b = b.lock().unwrap();
        (guard_a, guard_b)
    } else {
        let guard_b = b.lock().unwrap();
        let guard_a = a.lock().unwrap();
        (guard_a, guard_b)
    }
}

pub fn transfer_a_to_b(a: &Mutex<i64>, b: &Mutex<i64>, amount: i64) {
    let (mut guard_a, mut guard_b) = lock_pair(a, b);
    *guard_a -= amount;
    *guard_b += amount;
}

pub fn transfer_b_to_a(b: &Mutex<i64>, a: &Mutex<i64>, amount: i64) {
    let (mut guard_a, mut guard_b) = lock_pair(a, b);
    *guard_a += amount;
    *guard_b -= amount;
}

// TODO(FUNCTION)

#[cfg(test)]
mod visible {
    use crate::run_pair;

    #[test]
    fn two_transfers_apply() {
        assert_eq!(run_pair(100, 50, 10), (80, 70));
    }

    #[test]
    fn amounts_are_signed() {
        assert_eq!(run_pair(0, 100, -5), (10, 90));
    }

    #[test]
    fn the_helpers_compose() {
        use std::sync::Mutex;
        let a = Mutex::new(10);
        let b = Mutex::new(0);
        crate::transfer_a_to_b(&a, &b, 3);
        crate::transfer_b_to_a(&b, &a, 1);
        assert_eq!(*a.lock().unwrap(), 8);
        assert_eq!(*b.lock().unwrap(), 2);
    }
}
