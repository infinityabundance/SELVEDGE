use std::sync::atomic::{AtomicUsize, Ordering};

use concurrency_02_lost_update::{bump_all, bump_pair};

#[test]
fn pair_bumps_twice() {
    let counter = AtomicUsize::new(0);
    bump_pair(&counter);
    assert_eq!(counter.load(Ordering::SeqCst), 2);
}

#[test]
fn many_bumps_are_not_lost() {
    let counter = AtomicUsize::new(0);
    bump_all(&counter, 4, 500);
    assert_eq!(counter.load(Ordering::SeqCst), 2000);
}

#[test]
fn bumps_start_from_the_current_value() {
    let counter = AtomicUsize::new(41);
    bump_pair(&counter);
    assert_eq!(counter.load(Ordering::SeqCst), 43);
}
