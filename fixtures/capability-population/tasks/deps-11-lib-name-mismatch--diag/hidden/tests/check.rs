use deps_11_lib_name_mismatch::codec_total;

#[test]
fn empty_input_is_zero() {
    assert_eq!(codec_total(""), 0);
}

#[test]
fn a_single_byte_is_its_value() {
    assert_eq!(codec_total("a"), 97);
}

#[test]
fn bytes_are_summed() {
    assert_eq!(codec_total("abc"), 294);
}
