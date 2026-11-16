//! Sums and means over integer types.

pub mod math;
pub mod stats;
pub mod traits;

pub fn total_u32(items: &[u32]) -> i64 {
    math::sum(items)
}
