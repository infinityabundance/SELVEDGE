use migrate_09_mutable_receiver::count_to;

#[test]
fn ticks_the_exact_number_of_times() {
    assert_eq!(count_to(3), 3);
}

#[test]
fn zero_ticks_is_zero() {
    assert_eq!(count_to(0), 0);
}

#[test]
fn larger_counts_still_work() {
    assert_eq!(count_to(10), 10);
}
