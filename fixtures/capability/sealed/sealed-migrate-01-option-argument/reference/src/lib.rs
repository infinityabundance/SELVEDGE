//! Record a value.

pub use store::Store;

pub fn record(store: &mut Store, key: &str, value: i64) {
    store.set(key, Some(value));
}

pub fn read(store: &Store, key: &str) -> Option<i64> {
    store.get(key)
}
