use concurrency_04_deterministic_lost_update::increment_pair;

#[test]
fn both_increments_are_counted() {
    assert_eq!(increment_pair(), 2);
}

#[test]
fn the_result_is_stable_across_runs() {
    for _ in 0..20 {
        assert_eq!(increment_pair(), 2);
    }
}

#[test]
fn the_counter_is_not_one() {
    assert_ne!(increment_pair(), 1);
}
