use semantic_14_url_encode_plus::encode;

#[test]
fn a_literal_plus_is_encoded() {
    assert_eq!(encode("a+b"), "a%2Bb");
}

#[test]
fn spaces_and_pluses_together() {
    assert_eq!(encode("a b+c"), "a+b%2Bc");
}

#[test]
fn plain_text_is_unchanged() {
    assert_eq!(encode("abc"), "abc");
}

#[test]
fn empty_input_is_empty() {
    assert_eq!(encode(""), "");
}
