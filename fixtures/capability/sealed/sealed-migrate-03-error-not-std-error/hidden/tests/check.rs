use sealed_migrate_03_error_not_std_error::parse_or_message;

#[test]
fn a_number_is_doubled() {
    assert_eq!(parse_or_message("21").unwrap(), 42);
}

#[test]
fn surrounding_space_is_ignored() {
    assert_eq!(parse_or_message(" 3 ").unwrap(), 6);
}

#[test]
fn text_is_an_error_with_the_vendored_message() {
    let error = parse_or_message("nope").unwrap_err();
    assert_eq!(error.to_string(), "not a number");
}
