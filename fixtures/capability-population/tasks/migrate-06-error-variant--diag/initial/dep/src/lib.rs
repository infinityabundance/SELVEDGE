//! Table lookup, version 2. The error variants are structured.

#[derive(Debug, PartialEq)]
pub enum Error {
    Missing { key: String },
    Empty,
}

pub fn lookup(
    table: &[(&'static str, i32)],
    key: &str,
) -> Result<i32, Error> {
    if key.is_empty() {
        return Err(Error::Empty);
    }
    table
        .iter()
        .find(|(name, _value)| *name == key)
        .map(|(_name, value)| *value)
        .ok_or_else(|| Error::Missing { key: key.to_string() })
}
