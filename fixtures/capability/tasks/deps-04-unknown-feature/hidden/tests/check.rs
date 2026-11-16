use deps_04_unknown_feature::encode;

#[test]
fn wraps_a_value_in_quotes() {
    assert_eq!(encode("alpha"), "\"alpha\"");
}

#[test]
fn empty_value_gets_quotes() {
    assert_eq!(encode(""), "\"\"");
}

#[test]
fn spacing_is_preserved() {
    assert_eq!(encode("a b"), "\"a b\"");
}
