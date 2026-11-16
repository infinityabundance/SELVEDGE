//! Money arithmetic.

use std::ops::Add;

#[derive(Debug, PartialEq)]
pub struct Money(pub u64);

impl Add for Money {
    type Output = Money;

    fn add(self, right: Money) -> Money {
        Money(self.0 + right.0)
    }
}

pub fn total(left: Money, right: Money) -> Money {
    left + right
}
