//! Application errors.

#[derive(Debug, PartialEq)]
pub enum AppError {
    Parse(String),
    Missing(String),
    Corrupt(String),
}
