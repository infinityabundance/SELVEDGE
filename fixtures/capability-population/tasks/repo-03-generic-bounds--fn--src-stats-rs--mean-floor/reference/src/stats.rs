//! Flooring mean.

use crate::traits::AsI64;

pub fn mean_floor<T: AsI64>(items: &[T]) -> i64 {
    if items.is_empty() {
        return 0;
    }
    let total: i64 = items.iter().map(|value| value.as_i64()).sum();
    total.div_euclid(items.len() as i64)
}

#[cfg(test)]
mod visible {
    use crate::{math, stats, total_u32, traits::AsI64};

    #[test]
    fn sums_u32() {
        assert_eq!(math::sum(&[1u32, 2, 3]), 6);
        assert_eq!(total_u32(&[4, 5]), 9);
    }

    #[test]
    fn sums_i64() {
        assert_eq!(math::sum(&[1i64, -2, 3]), 2);
    }

    #[test]
    fn mean_floor_floors() {
        assert_eq!(stats::mean_floor(&[1i32, 2, 2]), 1);
        assert_eq!(stats::mean_floor(&[-3i64, -2]), -3);
    }

    #[test]
    fn empty_mean_is_zero() {
        assert_eq!(stats::mean_floor(&[] as &[i64]), 0);
    }

    #[test]
    fn the_trait_is_usable() {
        assert_eq!(7u32.as_i64(), 7);
    }
}
