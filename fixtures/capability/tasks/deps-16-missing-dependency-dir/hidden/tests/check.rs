use deps_16_missing_dependency_dir::codec_label;

#[test]
fn the_vendored_dependency_is_found() {
    assert_eq!(codec_label(), "codec 2.0.0");
}
