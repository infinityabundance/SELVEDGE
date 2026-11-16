//! The application error.

use crate::parse::ParseError;

#[derive(Debug, PartialEq)]
pub enum AppError {
    Missing,
    Parse(usize),
    Io,
}

impl core::fmt::Display for AppError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            AppError::Missing => write!(formatter, "missing"),
            AppError::Parse(field) => write!(formatter, "malformed field {field}"),
            AppError::Io => write!(formatter, "io error"),
        }
    }
}

impl std::error::Error for AppError {}

impl From<ParseError> for AppError {
    fn from(error: ParseError) -> Self {
        AppError::Parse(error.field)
    }
}

impl From<std::io::Error> for AppError {
    fn from(_error: std::io::Error) -> Self {
        AppError::Io
    }
}
