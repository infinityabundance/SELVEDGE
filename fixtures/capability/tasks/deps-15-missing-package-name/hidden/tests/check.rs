use deps_15_missing_package_name::initials;

#[test]
fn every_word_contributes_a_capital() {
    assert_eq!(initials("alpha beta"), "AB");
}

#[test]
fn surrounding_space_is_ignored() {
    assert_eq!(initials(" single "), "S");
}

#[test]
fn empty_input_is_empty() {
    assert_eq!(initials(""), "");
}
