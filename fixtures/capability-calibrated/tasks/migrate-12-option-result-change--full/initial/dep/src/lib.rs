//! Registry, version 2. `lookup` returns `Option` because a key may
//! be absent; version 1 returned the count directly.

use std::collections::BTreeMap;

pub struct Registry {
    items: BTreeMap<String, i64>,
}

impl Registry {
    pub fn new() -> Self {
        Self { items: BTreeMap::new() }
    }

    pub fn insert(&mut self, key: &str, value: i64) {
        self.items.insert(key.to_string(), value);
    }

    pub fn lookup(&self, key: &str) -> Option<i64> {
        self.items.get(key).copied()
    }
}
