use borrow_07_overlapping_mut_borrows::Counter;

#[test]
fn bumps_a_fresh_counter_twice() {
    let mut counter = Counter::new(0);
    counter.bump_twice();
    assert_eq!(counter.value(), 2);
}

#[test]
fn bumps_from_the_current_value() {
    let mut counter = Counter::new(5);
    counter.bump_twice();
    assert_eq!(counter.value(), 7);
}

#[test]
fn repeated_calls_accumulate() {
    let mut counter = Counter::new(0);
    counter.bump_twice();
    counter.bump_twice();
    counter.bump_twice();
    assert_eq!(counter.value(), 6);
}

#[test]
fn the_mutable_accessor_still_works() {
    let mut counter = Counter::new(1);
    *counter.value_mut() = 9;
    assert_eq!(counter.value(), 9);
}
