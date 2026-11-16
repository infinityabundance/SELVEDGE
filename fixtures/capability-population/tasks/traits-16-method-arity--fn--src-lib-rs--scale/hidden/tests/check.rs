use traits_16_method_arity::{scaled, Meter};

#[test]
fn a_factor_multiplies_the_value() {
    assert_eq!(scaled(&Meter { value: 3 }, 4), 12);
}

#[test]
fn a_zero_factor_is_zero() {
    assert_eq!(scaled(&Meter { value: 3 }, 0), 0);
}

#[test]
fn a_factor_of_one_is_the_identity() {
    assert_eq!(scaled(&Meter { value: 9 }, 1), 9);
}
