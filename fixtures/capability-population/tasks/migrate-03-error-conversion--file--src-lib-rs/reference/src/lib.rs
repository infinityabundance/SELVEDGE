//! Count fields, converting the dependency error into the application
//! error.

#[derive(Debug, PartialEq)]
pub enum AppError {
    Empty,
    Invalid { field: usize },
}

impl From<tokens::Error> for AppError {
    fn from(error: tokens::Error) -> Self {
        match error {
            tokens::Error::Empty => AppError::Empty,
            tokens::Error::Malformed(field) => AppError::Invalid { field },
        }
    }
}

pub fn count_fields(text: &str) -> Result<usize, AppError> {
    let tokens = tokens::tokenize(text)?;
    Ok(tokens.len())
}
