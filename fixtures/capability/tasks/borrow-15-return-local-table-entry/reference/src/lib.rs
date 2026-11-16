//! Seed a table and read it back.

use std::collections::BTreeMap;

pub fn seed(key: &str) -> String {
    let mut table = BTreeMap::new();
    table.insert(key.to_string(), format!("value-{key}"));
    table.get(key).cloned().unwrap_or_default()
}
