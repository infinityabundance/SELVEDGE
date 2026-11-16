use traits_07_add_impl::{Money, total};

#[test]
fn adds_two_values() {
    assert_eq!(total(Money(2), Money(3)), Money(5));
}

#[test]
fn zero_is_the_identity() {
    assert_eq!(total(Money(0), Money(7)), Money(7));
    assert_eq!(total(Money(7), Money(0)), Money(7));
}

#[test]
fn large_values_are_added() {
    assert_eq!(total(Money(1_000_000), Money(2_000_000)), Money(3_000_000));
}

#[test]
fn the_operator_is_available_directly() {
    assert_eq!(Money(1) + Money(2), Money(3));
}
