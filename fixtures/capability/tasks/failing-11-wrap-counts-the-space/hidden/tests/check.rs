use failing_11_wrap_counts_the_space::wrap;

#[test]
fn a_space_counts_against_the_width() {
    assert_eq!(wrap("aaa bbb", 6), vec!["aaa", "bbb"]);
}

#[test]
fn several_words_fill_one_line() {
    assert_eq!(wrap("aa bb cc", 5), vec!["aa bb", "cc"]);
}

#[test]
fn empty_input_has_no_lines() {
    assert!(wrap("", 4).is_empty());
}

#[test]
fn an_overlong_word_stays_whole() {
    assert_eq!(wrap("abcdef", 3), vec!["abcdef"]);
}
