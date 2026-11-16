//! Flooring mean.

use crate::traits::AsI64;

pub fn mean_floor<T: AsI64>(items: &[T]) -> i64 {
    if items.is_empty() {
        return 0;
    }
    let total: i64 = items.iter().map(|value| value.as_i64()).sum();
    total.div_euclid(items.len() as i64)
}
