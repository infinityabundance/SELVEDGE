//! Tokenisation, version 2: errors carry the failing field index.

#[derive(Debug, Clone, PartialEq)]
pub enum Error {
    Empty,
    Malformed(usize),
}

impl core::fmt::Display for Error {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Error::Empty => write!(formatter, "empty input"),
            Error::Malformed(field) => write!(formatter, "malformed field {field}"),
        }
    }
}

impl std::error::Error for Error {}

pub fn tokenize(text: &str) -> Result<Vec<u32>, Error> {
    if text.is_empty() {
        return Err(Error::Empty);
    }
    let mut output = Vec::new();
    for (index, field) in text.split_whitespace().enumerate() {
        match field.parse::<u32>() {
            Ok(value) => output.push(value),
            Err(_) => return Err(Error::Malformed(index)),
        }
    }
    Ok(output)
}
