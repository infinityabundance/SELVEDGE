use borrow_16_closure_borrows_local::make_counter;

#[test]
fn the_first_call_returns_seed_plus_one() {
    let mut counter = make_counter(3);
    assert_eq!(counter(), 4);
    assert_eq!(counter(), 5);
}

#[test]
fn counters_are_independent() {
    let mut first = make_counter(10);
    let mut second = make_counter(0);
    assert_eq!(first(), 11);
    assert_eq!(second(), 1);
    assert_eq!(first(), 12);
}

#[test]
fn the_counter_starts_above_the_seed_only() {
    let mut counter = make_counter(41);
    assert_eq!(counter(), 42);
}
