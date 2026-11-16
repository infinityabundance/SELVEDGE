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

// TODO(FUNCTION)

#[cfg(test)]
mod visible {
    use crate::{count_fields, AppError};

    #[test]
    fn counts_fields() {
        assert_eq!(count_fields("1 2 3"), Ok(3));
    }

    #[test]
    fn maps_the_empty_error() {
        assert_eq!(count_fields(""), Err(AppError::Empty));
    }

    #[test]
    fn maps_the_malformed_field() {
        assert_eq!(count_fields("1 x"), Err(AppError::Invalid { field: 1 }));
    }
}
