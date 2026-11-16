use deps_16_missing_dependency_dir::codec_fold;

#[test]
fn empty_input_is_zero() {
    assert_eq!(codec_fold(""), 0);
}

#[test]
fn two_bytes_are_combined() {
    assert_eq!(codec_fold("ab"), 3105);
}

#[test]
fn three_bytes_are_combined() {
    assert_eq!(codec_fold("abc"), 96354);
}
