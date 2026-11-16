use sealed_deps_01_missing_workspace_key::max_gap;

#[test]
fn the_largest_rise_wins() {
    assert_eq!(max_gap(&[1, 4, 2]), 3);
}

#[test]
fn a_single_value_has_no_gap() {
    assert_eq!(max_gap(&[5]), 0);
}

#[test]
fn a_falling_pair_has_no_gap() {
    assert_eq!(max_gap(&[9, 1]), 0);
}
