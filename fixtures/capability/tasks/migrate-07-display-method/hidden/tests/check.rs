use migrate_07_display_method::render;

#[test]
fn renders_a_value() {
    assert_eq!(render("alpha"), "alpha");
}

#[test]
fn preserves_spacing() {
    assert_eq!(render("a b"), "a b");
}

#[test]
fn renders_unicode() {
    assert_eq!(render("hé"), "hé");
}

#[test]
fn empty_renders_empty() {
    assert_eq!(render(""), "");
}
