use traits_06_method_ambiguity::{left, right, Item};

#[test]
fn left_uses_the_left_trait() {
    assert_eq!(left(&Item(1)), 1);
}

#[test]
fn right_uses_the_right_trait() {
    assert_eq!(right(&Item(1)), 101);
}

#[test]
fn the_traits_are_independent() {
    let item = Item(7);
    assert_eq!(left(&item), 7);
    assert_eq!(right(&item), 107);
}
