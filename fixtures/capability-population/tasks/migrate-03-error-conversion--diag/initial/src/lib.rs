//! Count fields. This does not build: `AppError` has no conversion from
//! the version-2 dependency error, so `?` is rejected.

#[derive(Debug, PartialEq)]
pub enum AppError {
    Empty,
    Invalid { field: usize },
}

pub fn count_fields(text: &str) -> Result<usize, AppError> {
    let tokens = tokens::tokenize(text)?;
    Ok(tokens.len())
}
