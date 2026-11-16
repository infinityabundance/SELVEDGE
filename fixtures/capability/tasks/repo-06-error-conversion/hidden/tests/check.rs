use repo_06_error_conversion::{error::AppError, load};

fn data() -> Vec<(&'static str, &'static str)> {
    vec![("alpha", "1"), ("blank", "")]
}

#[test]
fn returns_a_present_value() {
    assert_eq!(load("alpha", &data()), Ok("1".to_string()));
}

#[test]
fn maps_a_missing_key_to_missing() {
    assert_eq!(
        load("gamma", &data()),
        Err(AppError::Missing("gamma".to_string()))
    );
}

#[test]
fn maps_a_corrupt_key_to_corrupt() {
    assert_eq!(
        load("bad-key", &data()),
        Err(AppError::Corrupt("bad-key".to_string()))
    );
}

#[test]
fn reports_an_empty_value_as_a_parse_error() {
    assert_eq!(
        load("blank", &data()),
        Err(AppError::Parse("empty".to_string()))
    );
}
