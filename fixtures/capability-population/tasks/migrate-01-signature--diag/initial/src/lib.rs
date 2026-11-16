//! Load a measurement through the unit crate. This does not build: the
//! crate's `parse` returned a `Result` in version 1 and returns an
//! `Option` in version 2.

pub fn load(input: &str) -> Result<f64, String> {
    let measurement = units::parse(input).map_err(|error| error.to_string())?;
    Ok(measurement.value)
}
