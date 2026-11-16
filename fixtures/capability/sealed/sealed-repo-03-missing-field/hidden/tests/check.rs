use sealed_repo_03_missing_field::make;

#[test]
fn the_default_configuration_is_complete() {
    let config = make();
    assert_eq!(config.retries, 3);
    assert_eq!(config.label, "default");
}
