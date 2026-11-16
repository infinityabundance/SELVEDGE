use deps_14_workspace_inheritance::parse_flag;

#[test]
fn affirmative_forms_are_true() {
    assert_eq!(parse_flag(" yes "), Some(true));
    assert_eq!(parse_flag("ON"), Some(true));
}

#[test]
fn negative_forms_are_false() {
    assert_eq!(parse_flag("0"), Some(false));
}

#[test]
fn unknown_text_is_none() {
    assert_eq!(parse_flag("maybe"), None);
}
