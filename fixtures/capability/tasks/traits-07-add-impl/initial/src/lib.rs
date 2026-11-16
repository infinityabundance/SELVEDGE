//! Money arithmetic without the Add impl.

#[derive(Debug, PartialEq)]
pub struct Money(pub u64);

pub fn total(left: Money, right: Money) -> Money {
    left + right
}
