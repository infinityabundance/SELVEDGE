//! Store reads.

#[derive(Debug, PartialEq)]
pub enum StoreError {
    Corrupt(String),
    NotFound(String),
}

pub fn read(key: &str, data: &[(&str, &str)]) -> Result<String, StoreError> {
    if key.starts_with("bad") {
        return Err(StoreError::Corrupt(key.to_string()));
    }
    data.iter()
        .find(|(name, _value)| *name == key)
        .map(|(_name, value)| value.to_string())
        .ok_or_else(|| StoreError::NotFound(key.to_string()))
}
