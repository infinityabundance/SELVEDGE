use borrow_15_return_local_table_entry::seed;

#[test]
fn returns_the_seeded_value() {
    assert_eq!(seed("k"), "value-k".to_string());
}

#[test]
fn successive_calls_are_independent() {
    let first = seed("one");
    let second = seed("two");
    assert_eq!(first, "value-one");
    assert_eq!(second, "value-two");
}

#[test]
fn a_long_key_is_included_verbatim() {
    assert_eq!(seed("alpha-beta"), "value-alpha-beta".to_string());
}
