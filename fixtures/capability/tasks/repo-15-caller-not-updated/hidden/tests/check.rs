use repo_15_caller_not_updated::line;

#[test]
fn short_text_is_padded_to_six() {
    assert_eq!(line("ab"), "ab    ");
    assert_eq!(line("ab").len(), 6);
}

#[test]
fn long_text_is_not_truncated() {
    assert_eq!(line("abcdefg"), "abcdefg");
}

#[test]
fn empty_text_becomes_six_spaces() {
    assert_eq!(line(""), "      ");
}
