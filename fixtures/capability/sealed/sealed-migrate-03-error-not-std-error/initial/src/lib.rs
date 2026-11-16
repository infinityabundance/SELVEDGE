//! Parsing through the vendored crate. This does not build: version 2's
//! error is not a `std::error::Error`.

pub fn parse_or_message(text: &str) -> Result<i64, Box<dyn std::error::Error>> {
    let value = parse::parse(text)?;
    Ok(value * 2)
}
