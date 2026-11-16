use std::sync::atomic::AtomicU32;

use sealed_concurrency_03_two_closures_unique_access::bump_both;

#[test]
fn both_closures_add_one() {
    assert_eq!(bump_both(&AtomicU32::new(0)), 2);
}

#[test]
fn the_starting_value_is_kept() {
    assert_eq!(bump_both(&AtomicU32::new(10)), 12);
}

#[test]
fn repeated_calls_advance() {
    let counter = AtomicU32::new(0);
    assert_eq!(bump_both(&counter), 2);
    assert_eq!(bump_both(&counter), 4);
}
