use sealed_deps_02_resolver_type::reverse_words;

#[test]
fn three_words_are_reversed() {
    assert_eq!(reverse_words("a b c"), "c b a");
}

#[test]
fn surrounding_space_is_ignored() {
    assert_eq!(reverse_words(" single "), "single");
}

#[test]
fn empty_input_is_empty() {
    assert_eq!(reverse_words(""), "");
}
