//! Unit scaling.

pub fn scale_units(value: u64) -> f64 {
    scale::scale(value as f64)
}
