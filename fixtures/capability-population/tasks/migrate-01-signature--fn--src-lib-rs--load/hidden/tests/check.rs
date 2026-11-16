use migrate_01_signature::load;

#[test]
fn parses_a_value() {
    assert_eq!(load("12.5kg"), Ok(12.5));
}

#[test]
fn rejects_garbage_with_the_registered_text() {
    assert_eq!(load("nope"), Err("invalid measurement".to_string()));
}

#[test]
fn rejects_empty_input() {
    assert_eq!(load(""), Err("invalid measurement".to_string()));
}
