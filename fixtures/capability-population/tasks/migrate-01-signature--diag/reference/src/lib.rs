//! Load a measurement through the unit crate.

pub fn load(input: &str) -> Result<f64, String> {
    units::parse(input)
        .map(|measurement| measurement.value)
        .ok_or_else(|| "invalid measurement".to_string())
}
