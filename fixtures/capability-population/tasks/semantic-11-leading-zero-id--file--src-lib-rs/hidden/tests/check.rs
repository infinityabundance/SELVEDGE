use semantic_11_leading_zero_id::parse_id;

#[test]
fn a_leading_zero_is_rejected() {
    assert_eq!(parse_id("012"), None);
    assert_eq!(parse_id("0012"), None);
}

#[test]
fn zero_itself_is_allowed() {
    assert_eq!(parse_id("0"), Some(0));
}

#[test]
fn plain_numbers_parse() {
    assert_eq!(parse_id("12"), Some(12));
}

#[test]
fn empty_and_non_digits_are_rejected() {
    assert_eq!(parse_id(""), None);
    assert_eq!(parse_id("12a"), None);
}
