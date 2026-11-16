use repo_11_private_field::{make, Config};

#[test]
fn the_builder_uses_the_constructor() {
    assert_eq!(make().value(), 2);
}

#[test]
fn the_constructor_is_public() {
    assert_eq!(Config::new(9).value(), 9);
}

#[test]
fn the_field_stays_private() {
    let config = make();
    assert_eq!(config.value(), 2);
}
