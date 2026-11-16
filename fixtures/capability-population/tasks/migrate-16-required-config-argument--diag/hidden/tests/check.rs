use migrate_16_required_config_argument::render_label;

#[test]
fn short_text_is_padded_to_ten() {
    assert_eq!(render_label("ab"), "ab        ");
    assert_eq!(render_label("ab").len(), 10);
}

#[test]
fn long_text_is_not_truncated() {
    assert_eq!(render_label("abcdefghijk"), "abcdefghijk");
}

#[test]
fn empty_text_becomes_spaces() {
    assert_eq!(render_label(""), "          ");
}
