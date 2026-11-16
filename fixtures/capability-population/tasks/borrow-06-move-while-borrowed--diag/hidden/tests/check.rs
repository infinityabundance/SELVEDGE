use borrow_06_move_while_borrowed::split_first;

#[test]
fn returns_the_original_and_the_token_length() {
    assert_eq!(split_first("hello world".to_string()), ("hello world".to_string(), 5));
}

#[test]
fn leading_whitespace_is_not_part_of_the_token() {
    assert_eq!(split_first("   hi there".to_string()), ("   hi there".to_string(), 2));
}

#[test]
fn blank_input_has_length_zero() {
    assert_eq!(split_first("   ".to_string()), ("   ".to_string(), 0));
}

#[test]
fn empty_input_has_length_zero() {
    assert_eq!(split_first(String::new()), (String::new(), 0));
}
