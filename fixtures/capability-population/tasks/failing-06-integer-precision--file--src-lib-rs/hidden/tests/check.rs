use failing_06_integer_precision::percent_of;

#[test]
fn keeps_the_remainder_for_an_odd_amount() {
    assert_eq!(percent_of(199, 50), 99);
}

#[test]
fn exact_hundreds_are_unchanged() {
    assert_eq!(percent_of(200, 50), 100);
}

#[test]
fn one_percent_of_one_cent_truncates_to_zero() {
    assert_eq!(percent_of(1, 1), 0);
}

#[test]
fn full_percent_is_the_whole_amount() {
    assert_eq!(percent_of(1, 100), 1);
}

#[test]
fn zero_percent_is_zero() {
    assert_eq!(percent_of(10_000, 0), 0);
}
