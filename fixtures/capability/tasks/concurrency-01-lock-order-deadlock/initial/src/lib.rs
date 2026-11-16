//! Two accounts and two transfer helpers. Each helper sleeps while
//! holding its first lock, so opposite orders deadlock.

use std::sync::{Barrier, Mutex};
use std::time::Duration;

pub fn transfer_a_to_b(a: &Mutex<i64>, b: &Mutex<i64>, amount: i64) {
    let mut guard_a = a.lock().unwrap();
    std::thread::sleep(Duration::from_millis(10));
    let mut guard_b = b.lock().unwrap();
    *guard_a -= amount;
    *guard_b += amount;
}

pub fn transfer_b_to_a(b: &Mutex<i64>, a: &Mutex<i64>, amount: i64) {
    let mut guard_b = b.lock().unwrap();
    std::thread::sleep(Duration::from_millis(10));
    let mut guard_a = a.lock().unwrap();
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
