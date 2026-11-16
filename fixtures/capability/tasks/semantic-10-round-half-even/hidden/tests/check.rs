use semantic_10_round_half_even::round_half_even;

#[test]
fn positive_ties_go_to_even() {
    assert_eq!(round_half_even(2.5), 2.0);
    assert_eq!(round_half_even(3.5), 4.0);
}

#[test]
fn negative_ties_go_to_even() {
    assert_eq!(round_half_even(-2.5), -2.0);
    assert_eq!(round_half_even(-3.5), -4.0);
}

#[test]
fn non_ties_round_normally() {
    assert_eq!(round_half_even(2.4), 2.0);
    assert_eq!(round_half_even(2.6), 3.0);
    assert_eq!(round_half_even(-2.6), -3.0);
}

#[test]
fn integers_are_unchanged() {
    assert_eq!(round_half_even(7.0), 7.0);
}
