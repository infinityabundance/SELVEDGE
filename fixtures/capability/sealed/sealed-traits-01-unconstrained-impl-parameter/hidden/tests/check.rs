use sealed_traits_01_unconstrained_impl_parameter::Registry;

#[test]
fn the_constructor_is_empty() {
    assert!(Registry::new().entries.is_empty());
}

#[test]
fn a_constructed_registry_can_be_used() {
    let mut registry = Registry::new();
    registry.entries.push("a".to_string());
    assert_eq!(registry.entries.len(), 1);
}
