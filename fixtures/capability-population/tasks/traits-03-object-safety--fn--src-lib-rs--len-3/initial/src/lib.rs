//! A small key-value store used through dynamic dispatch.

use std::collections::BTreeMap;

pub trait Store {
    fn get(&self, key: &str) -> Option<i64>;
    fn put(&mut self, key: &str, value: i64);
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

    fn put(&mut self, key: &str, value: i64) {
        self.items.insert(key.to_string(), value);
    }

// TODO(FUNCTION)
}

pub fn count(store: &dyn Store) -> usize {
    store.len()
}

#[cfg(test)]
mod visible {
    use crate::{count, MemoryStore, Store};

    #[test]
    fn put_and_get() {
        let mut store = MemoryStore::default();
        store.put("a", 3i64);
        assert_eq!(store.get("a"), Some(3));
    }

    #[test]
    fn dynamic_dispatch_works() {
        let mut store = MemoryStore::default();
        store.put("a", 1i64);
        store.put("b", 2i64);
        let dynamic: &dyn Store = &store;
        assert_eq!(count(dynamic), 2);
    }

    #[test]
    fn empty_store_counts_zero() {
        let store = MemoryStore::default();
        assert_eq!(count(&store), 0);
    }
}
