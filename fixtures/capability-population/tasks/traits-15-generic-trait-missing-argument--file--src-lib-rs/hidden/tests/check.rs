use traits_15_generic_trait_missing_argument::encode_byte;

#[test]
fn fifteen_is_lowercase_hex() {
    assert_eq!(encode_byte(15), "0f");
}

#[test]
fn zero_is_padded() {
    assert_eq!(encode_byte(0), "00");
}

#[test]
fn the_maximum_byte_is_two_digits() {
    assert_eq!(encode_byte(255), "ff");
}
