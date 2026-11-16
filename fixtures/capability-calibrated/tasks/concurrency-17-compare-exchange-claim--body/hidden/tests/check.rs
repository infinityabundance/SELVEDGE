use std::sync::atomic::{AtomicBool, Ordering};

use concurrency_17_compare_exchange_claim::claim;

#[test]
fn the_first_claim_wins() {
    let flag = AtomicBool::new(false);
    assert!(claim(&flag));
}

#[test]
fn the_second_claim_loses() {
    let flag = AtomicBool::new(false);
    assert!(claim(&flag));
    assert!(!claim(&flag));
}

#[test]
fn a_released_flag_can_be_claimed_again() {
    let flag = AtomicBool::new(true);
    flag.store(false, Ordering::SeqCst);
    assert!(claim(&flag));
}

#[test]
fn independent_flags_are_independent() {
    let first = AtomicBool::new(false);
    let second = AtomicBool::new(false);
    assert!(claim(&first));
    assert!(claim(&second));
}
