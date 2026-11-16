use traits_18_duplicate_inherent_method::Meter;

#[test]
fn the_ratio_is_the_value() {
    assert_eq!(Meter { value: 5 }.ratio(), 5);
}

#[test]
fn zero_is_zero() {
    assert_eq!(Meter { value: 0 }.ratio(), 0);
}

#[test]
fn the_method_is_available_on_references() {
    let meter = Meter { value: 9 };
    assert_eq!((&meter).ratio(), 9);
}
