use sealed_semantic_01_floor_division::mean_floor;

#[test]
fn a_negative_half_floors_down() {
    assert_eq!(mean_floor(&[-3, -2]), -3);
}

#[test]
fn a_negative_exact_average_is_unchanged() {
    assert_eq!(mean_floor(&[-4, -2]), -3);
}

#[test]
fn positive_values_behave_as_before() {
    assert_eq!(mean_floor(&[2, 4]), 3);
}

#[test]
fn an_empty_slice_is_zero() {
    assert_eq!(mean_floor(&[]), 0);
}
