use sealed_traits_02_missing_associated_item::{clone_value, Cell, Element};

#[test]
fn the_helper_returns_the_value() {
    let cell = Cell { item: 7 };
    assert_eq!(clone_value(&cell), 7);
}

#[test]
fn the_trait_method_agrees() {
    let cell = Cell { item: 3 };
    assert_eq!(cell.value(), clone_value(&cell));
}
