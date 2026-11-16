use migrate_08_builder_result::configured;

#[test]
fn accepts_a_positive_level() {
    assert_eq!(configured(3), Ok(3));
}

#[test]
fn accepts_the_maximum_level() {
    assert_eq!(configured(255), Ok(255));
}

#[test]
fn rejects_zero_with_the_registered_text() {
    assert_eq!(configured(0), Err("invalid level".to_string()));
}

#[test]
fn accepts_level_one() {
    assert_eq!(configured(1), Ok(1));
}
