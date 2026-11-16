//! Doubling through the units crate. This does not build: the
//! function was renamed in version 2.

pub fn double(input: &str) -> Option<f64> {
    units::parse(input).map(|value| value * 2.0)
}
