use sealed_migrate_01_option_argument::{read, record, Store};

#[test]
fn a_recorded_value_can_be_read() {
    let mut store = Store::new();
    record(&mut store, "a", 7);
    assert_eq!(read(&store, "a"), Some(7));
}

#[test]
fn reading_a_missing_key_is_none() {
    let store = Store::new();
    assert_eq!(read(&store, "missing"), None);
}

#[test]
fn a_later_record_overwrites() {
    let mut store = Store::new();
    record(&mut store, "a", 1);
    record(&mut store, "a", 2);
    assert_eq!(read(&store, "a"), Some(2));
}
