//! The application error.

#[derive(Debug, PartialEq)]
pub enum AppError {
    Missing,
}

impl core::fmt::Display for AppError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            AppError::Missing => write!(formatter, "missing"),
        }
    }
}

impl std::error::Error for AppError {}
