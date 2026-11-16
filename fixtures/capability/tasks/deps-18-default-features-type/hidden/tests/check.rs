use deps_18_default_features_type::codec_label;

#[test]
fn the_label_is_wrapped() {
    assert_eq!(codec_label(), "[codec 2.0.0]");
}

#[test]
fn the_label_has_the_registered_length() {
    assert_eq!(codec_label().len(), 13);
}

#[test]
fn the_label_starts_with_the_opening_bracket() {
    assert!(codec_label().starts_with('['));
}
