use semantic_15_shorten_with_ellipsis::shorten;

#[test]
fn the_result_fits_the_budget() {
    let short = shorten("abcdef", 4);
    assert_eq!(short, "abc…");
    assert_eq!(short.chars().count(), 4);
}

#[test]
fn a_budget_of_one_is_just_the_ellipsis() {
    assert_eq!(shorten("abcdef", 1), "…");
}

#[test]
fn a_zero_budget_is_empty() {
    assert_eq!(shorten("abcdef", 0), "");
}

#[test]
fn text_that_fits_is_unchanged() {
    assert_eq!(shorten("abc", 3), "abc");
    assert_eq!(shorten("", 5), "");
}
