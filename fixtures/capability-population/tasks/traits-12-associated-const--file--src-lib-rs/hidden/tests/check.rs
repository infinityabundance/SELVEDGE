use traits_12_associated_const::{Limits, Small};

#[test]
fn the_constant_is_eight() {
    assert_eq!(Small::MAX, 8);
}

#[test]
fn values_above_the_limit_are_capped() {
    assert_eq!(Small.capped(20), 8);
}

#[test]
fn values_below_the_limit_are_kept() {
    assert_eq!(Small.capped(3), 3);
}
