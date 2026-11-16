//! Seed a table and read it back. The crate does not build: the
//! result borrows a table created inside the function.

use std::collections::BTreeMap;

pub fn seed<'a>(key: &str) -> &'a str {
    let mut table = BTreeMap::new();
    table.insert(key.to_string(), format!("value-{key}"));
    table.get(key).map(|value| value.as_str()).unwrap_or("")
}
