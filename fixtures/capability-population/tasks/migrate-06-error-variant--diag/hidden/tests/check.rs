use migrate_06_error_variant::read;

fn table() -> Vec<(&'static str, i32)> {
    vec![("alpha", 1), ("beta", 2)]
}

#[test]
fn returns_a_present_value() {
    assert_eq!(read(&table(), "beta"), Ok(2));
}

#[test]
fn reports_a_missing_key_with_its_name() {
    assert_eq!(read(&table(), "gamma"), Err("missing gamma".to_string()));
}

#[test]
fn reports_the_empty_variant_distinctly() {
    assert_eq!(read(&table(), ""), Err("empty key".to_string()));
}
