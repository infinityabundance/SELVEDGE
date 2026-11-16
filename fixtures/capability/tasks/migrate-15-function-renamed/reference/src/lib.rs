//! Doubling through the units crate.

pub fn double(input: &str) -> Option<f64> {
    units::parse_value(input).map(|value| value * 2.0)
}
