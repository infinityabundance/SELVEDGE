//! Sum a slice of anything convertible to i64.

use crate::traits::AsI64;

pub fn sum<T: AsI64>(items: &[T]) -> i64 {
    items.iter().map(|value| value.as_i64()).sum()
}
