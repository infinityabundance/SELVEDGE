//! Parsing through the vendored crate.

pub fn parse_or_message(text: &str) -> Result<i64, Box<dyn std::error::Error>> {
    let value = parse::parse(text).map_err(|error| error.to_string())?;
    Ok(value * 2)
}
