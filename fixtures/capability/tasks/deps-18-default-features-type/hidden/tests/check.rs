use deps_18_default_features_type::codec_label;

#[test]
fn the_dependency_is_built_without_default_features() {
    assert_eq!(codec_label(), "codec 2.0.0");
}
