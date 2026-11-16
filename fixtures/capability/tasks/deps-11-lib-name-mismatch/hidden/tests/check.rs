use deps_11_lib_name_mismatch::codec_label;

#[test]
fn the_vendored_library_is_reached() {
    assert_eq!(codec_label(), "codec 2.0.0");
}
