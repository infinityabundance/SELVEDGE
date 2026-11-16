use deps_06_edition_reserved_keyword::squares;

#[test]
fn squares_the_prefix() {
    assert_eq!(squares(4), vec![0, 1, 4, 9]);
}

#[test]
fn empty_prefix_is_empty() {
    assert!(squares(0).is_empty());
}

#[test]
fn one_element_is_zero() {
    assert_eq!(squares(1), vec![0]);
}
