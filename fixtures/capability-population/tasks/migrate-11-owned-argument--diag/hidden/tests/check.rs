use migrate_11_owned_argument::plate;

#[test]
fn wraps_the_text_in_brackets() {
    assert_eq!(plate("x"), "[x]");
}

#[test]
fn empty_text_is_still_wrapped() {
    assert_eq!(plate(""), "[]");
}

#[test]
fn spaces_are_preserved() {
    assert_eq!(plate("a b"), "[a b]");
}
