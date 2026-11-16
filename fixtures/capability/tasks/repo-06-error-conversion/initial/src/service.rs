//! Service loading.

use crate::error::AppError;
use crate::store;

pub fn load(key: &str, data: &[(&str, &str)]) -> Result<String, AppError> {
    let value = store::read(key, data)?;
    if value.is_empty() {
        return Err(AppError::Parse("empty".to_string()));
    }
    Ok(value)
}
