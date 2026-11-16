use sealed_failing_02_base36_table::to_base36;

#[test]
fn the_last_digit_is_z() {
    assert_eq!(to_base36(35), "z");
}

#[test]
fn zero_and_small_values() {
    assert_eq!(to_base36(0), "0");
    assert_eq!(to_base36(10), "a");
    assert_eq!(to_base36(34), "y");
}

#[test]
fn multi_digit_values() {
    assert_eq!(to_base36(36), "10");
    assert_eq!(to_base36(1295), "zz");
}
