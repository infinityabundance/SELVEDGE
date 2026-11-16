//! Parse one CSV row.

#[derive(Debug, PartialEq)]
pub struct ParseError {
    pub field: usize,
}

impl core::fmt::Display for ParseError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(formatter, "malformed field {}", self.field)
    }
}

impl std::error::Error for ParseError {}

pub fn parse_row(text: &str) -> Result<(String, i64), ParseError> {
    let mut parts = text.split(',');
    let name = parts.next().ok_or(ParseError { field: 0 })?;
    let value = parts
        .next()
        .ok_or(ParseError { field: 1 })?
        .trim()
        .parse::<i64>()
        .map_err(|_| ParseError { field: 1 })?;
    Ok((name.to_string(), value))
}
