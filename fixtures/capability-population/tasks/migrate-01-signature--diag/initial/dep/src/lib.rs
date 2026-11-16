//! Unit parsing, version 2. `parse` no longer reports why it failed;
//! `parse_strict` does.

#[derive(Debug, Clone, PartialEq)]
pub struct Measurement {
    pub value: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ParseError {
    Empty,
    Malformed,
}

impl core::fmt::Display for ParseError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            ParseError::Empty => write!(formatter, "empty input"),
            ParseError::Malformed => write!(formatter, "malformed measurement"),
        }
    }
}

impl std::error::Error for ParseError {}

fn number(input: &str) -> Option<f64> {
    let digits: String = input
        .chars()
        .take_while(|character| character.is_ascii_digit() || *character == '.')
        .collect();
    digits.parse().ok()
}

pub fn parse(input: &str) -> Option<Measurement> {
    if input.is_empty() {
        return None;
    }
    number(input).map(|value| Measurement { value })
}

pub fn parse_strict(input: &str) -> Result<Measurement, ParseError> {
    if input.is_empty() {
        return Err(ParseError::Empty);
    }
    match number(input) {
        Some(value) => Ok(Measurement { value }),
        None => Err(ParseError::Malformed),
    }
}
