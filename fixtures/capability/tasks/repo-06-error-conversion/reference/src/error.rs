//! Application errors.

use crate::store::StoreError;

#[derive(Debug, PartialEq)]
pub enum AppError {
    Parse(String),
    Missing(String),
    Corrupt(String),
}

impl From<StoreError> for AppError {
    fn from(error: StoreError) -> Self {
        match error {
            StoreError::Corrupt(key) => AppError::Corrupt(key),
            StoreError::NotFound(key) => AppError::Missing(key),
        }
    }
}
