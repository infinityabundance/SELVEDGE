use repo_10_missing_module_file::{retries, Settings};

#[test]
fn the_default_retry_count_is_three() {
    assert_eq!(retries(), 3);
}

#[test]
fn the_default_label_is_available() {
    assert_eq!(Settings::default().label, "default");
}

#[test]
fn settings_are_cloneable_and_comparable() {
    let settings = Settings { retries: 5, label: "x".to_string() };
    let copy = settings.clone();
    assert_eq!(settings, copy);
}
