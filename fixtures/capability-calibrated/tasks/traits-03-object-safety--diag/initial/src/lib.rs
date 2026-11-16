//! A small key-value store used through dynamic dispatch.

use std::collections::BTreeMap;

pub trait Store {
    fn get(&self, key: &str) -> Option<i64>;
    fn put<T: Into<i64>>(&mut self, key: &str, value: T);
    fn len(&self) -> usize;
}

#[derive(Default)]
pub struct MemoryStore {
    items: BTreeMap<String, i64>,
}

impl Store for MemoryStore {
    fn get(&self, key: &str) -> Option<i64> {
        self.items.get(key).copied()
    }

    fn put<T: Into<i64>>(&mut self, key: &str, value: T) {
        self.items.insert(key.to_string(), value.into());
    }

    fn len(&self) -> usize {
        self.items.len()
    }
}

pub fn count(store: &dyn Store) -> usize {
    store.len()
}
