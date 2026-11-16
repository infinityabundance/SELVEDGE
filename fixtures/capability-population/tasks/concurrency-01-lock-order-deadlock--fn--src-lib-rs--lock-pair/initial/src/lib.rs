//! Two accounts with one global lock order.

use std::sync::{Barrier, Mutex, MutexGuard};

// TODO(FUNCTION)

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

pub fn run_pair(x: i64, y: i64, amount: i64) -> (i64, i64) {
    let a = Mutex::new(x);
    let b = Mutex::new(y);
    let barrier = Barrier::new(2);
    std::thread::scope(|scope| {
        scope.spawn(|| {
            barrier.wait();
            transfer_a_to_b(&a, &b, amount);
        });
        scope.spawn(|| {
            barrier.wait();
            transfer_b_to_a(&b, &a, -amount);
        });
    });
    (a.into_inner().unwrap(), b.into_inner().unwrap())
}

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
