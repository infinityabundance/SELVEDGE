use sealed_traits_03_dyn_associated_type::{describe, Map};

#[test]
fn the_trait_object_returns_the_value() {
    assert_eq!(describe(&Map), 0);
}

#[test]
fn the_concrete_type_agrees() {
    use sealed_traits_03_dyn_associated_type::Store;
    assert_eq!(Map.get(0), describe(&Map));
}
