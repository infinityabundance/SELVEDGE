//! Saturating scaling.

pub fn scale(value: u32, factor: u32) -> u32 {
    value.saturating_mul(factor)
}
