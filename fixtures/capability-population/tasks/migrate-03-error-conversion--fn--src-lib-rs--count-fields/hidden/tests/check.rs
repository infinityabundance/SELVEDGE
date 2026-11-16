use migrate_03_error_conversion::{count_fields, AppError};

#[test]
fn counts_fields() {
    assert_eq!(count_fields("1 2 3"), Ok(3));
}

#[test]
fn maps_the_empty_error() {
    assert_eq!(count_fields(""), Err(AppError::Empty));
}

#[test]
fn maps_the_malformed_field() {
    assert_eq!(count_fields("1 x"), Err(AppError::Invalid { field: 1 }));
}
