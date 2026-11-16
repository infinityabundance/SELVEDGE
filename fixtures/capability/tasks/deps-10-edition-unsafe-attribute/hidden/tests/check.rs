use deps_10_edition_unsafe_attribute::checksum;

#[test]
fn sums_the_bytes() {
    assert_eq!(checksum(b"ab"), 195u8);
}

#[test]
fn empty_input_is_zero() {
    assert_eq!(checksum(b""), 0u8);
}

#[test]
fn the_sum_wraps() {
    assert_eq!(checksum(&[255, 1]), 0u8);
}
