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
