use std::sync::atomic::{AtomicUsize, Ordering};

use sealed_concurrency_02_atomic_take_pair::take_pair;

#[test]
fn the_pair_sums_to_the_stored_value() {
    let counter = AtomicUsize::new(5);
    assert_eq!(take_pair(&counter), 5);
    assert_eq!(counter.load(Ordering::SeqCst), 0);
}

#[test]
fn an_empty_counter_yields_zero() {
    let counter = AtomicUsize::new(0);
    assert_eq!(take_pair(&counter), 0);
    assert_eq!(counter.load(Ordering::SeqCst), 0);
}

#[test]
fn the_pair_runs_again_on_a_refilled_counter() {
    let counter = AtomicUsize::new(9);
    assert_eq!(take_pair(&counter), 9);
    counter.store(4, Ordering::SeqCst);
    assert_eq!(take_pair(&counter), 4);
}
