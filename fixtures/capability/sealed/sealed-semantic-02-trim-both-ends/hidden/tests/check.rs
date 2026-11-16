use sealed_semantic_02_trim_both_ends::cleaned;

#[test]
fn trailing_whitespace_is_trimmed() {
    assert_eq!(cleaned(" x "), "x");
}

#[test]
fn internal_whitespace_is_kept() {
    assert_eq!(cleaned(" a b "), "a b");
}

#[test]
fn tabs_and_newlines_are_trimmed() {
    assert_eq!(cleaned("\tx\n"), "x");
}

#[test]
fn empty_and_blank_inputs_are_empty() {
    assert_eq!(cleaned(""), "");
    assert_eq!(cleaned("   "), "");
}
