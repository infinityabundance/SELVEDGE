use std::sync::atomic::{AtomicU64, Ordering};

use concurrency_11_cell_not_sync::count_with;

#[test]
fn every_thread_adds_one() {
    let counter = AtomicU64::new(0);
    assert_eq!(count_with(&counter, 4), 4);
    assert_eq!(counter.load(Ordering::SeqCst), 4);
}

#[test]
fn the_count_starts_from_the_current_value() {
    let counter = AtomicU64::new(10);
    assert_eq!(count_with(&counter, 3), 13);
}

#[test]
fn zero_threads_change_nothing() {
    let counter = AtomicU64::new(7);
    assert_eq!(count_with(&counter, 0), 7);
}
