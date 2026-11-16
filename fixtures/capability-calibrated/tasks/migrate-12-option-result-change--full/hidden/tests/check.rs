use migrate_12_option_result_change::{next_count, Registry};

fn registry() -> Registry {
    let mut registry = Registry::new();
    registry.insert("alpha", 4);
    registry.insert("zero", 0);
    registry
}

#[test]
fn a_present_key_is_incremented() {
    assert_eq!(next_count(&registry(), "alpha"), 5);
}

#[test]
fn a_zeroth_value_is_incremented() {
    assert_eq!(next_count(&registry(), "zero"), 1);
}

#[test]
fn a_missing_key_is_zero() {
    assert_eq!(next_count(&registry(), "missing"), 0);
}
