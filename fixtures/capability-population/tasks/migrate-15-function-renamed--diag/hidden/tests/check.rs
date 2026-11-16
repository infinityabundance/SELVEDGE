use migrate_15_function_renamed::double;

#[test]
fn doubles_a_number() {
    assert_eq!(double("2.5"), Some(5.0));
}

#[test]
fn ignores_surrounding_space() {
    assert_eq!(double(" 3 "), Some(6.0));
}

#[test]
fn rejects_text() {
    assert_eq!(double("x"), None);
}
