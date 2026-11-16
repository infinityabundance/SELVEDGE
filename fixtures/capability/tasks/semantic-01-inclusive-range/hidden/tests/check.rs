use semantic_01_inclusive_range::range_sum;

#[test]
fn includes_the_upper_bound() {
    assert_eq!(range_sum(1, 4), 10);
}

#[test]
fn a_single_point_is_that_point() {
    assert_eq!(range_sum(3, 3), 3);
}

#[test]
fn reversed_bounds_are_empty() {
    assert_eq!(range_sum(4, 1), 0);
}

#[test]
fn negative_ranges_are_inclusive() {
    assert_eq!(range_sum(-2, 2), 0);
}
