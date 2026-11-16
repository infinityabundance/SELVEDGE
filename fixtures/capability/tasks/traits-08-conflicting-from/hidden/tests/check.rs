use traits_08_conflicting_from::Celsius;

#[test]
fn converts_from_f64() {
    assert_eq!(Celsius::from(21.0), Celsius(21.0));
}

#[test]
fn converts_from_i32() {
    assert_eq!(Celsius::from(7), Celsius(7.0));
}

#[test]
fn converts_negative_i32() {
    assert_eq!(Celsius::from(-3), Celsius(-3.0));
}
