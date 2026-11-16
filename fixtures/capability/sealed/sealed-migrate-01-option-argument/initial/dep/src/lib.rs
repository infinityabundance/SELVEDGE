//! Store, version 2. Values are optional so a key can be cleared.

use std::collections::BTreeMap;

pub struct Store {
    values: BTreeMap<String, Option<i64>>,
}

impl Store {
    pub fn new() -> Self {
        Self { values: BTreeMap::new() }
    }

    pub fn set(&mut self, key: &str, value: Option<i64>) {
        self.values.insert(key.to_string(), value);
    }

    pub fn get(&self, key: &str) -> Option<i64> {
        self.values.get(key).copied().flatten()
    }
}
