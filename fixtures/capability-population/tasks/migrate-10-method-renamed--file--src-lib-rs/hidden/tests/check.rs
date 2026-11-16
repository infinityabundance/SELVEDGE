use migrate_10_method_renamed::running_total;

#[test]
fn sums_every_value() {
    assert_eq!(running_total(&[1, 2, 3]), 6);
}

#[test]
fn an_empty_slice_sums_to_zero() {
    assert_eq!(running_total(&[]), 0);
}

#[test]
fn negative_values_are_included() {
    assert_eq!(running_total(&[-4, 4, -2]), -2);
}
