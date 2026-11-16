//! Record a value. This does not build: version 2 takes an Option.

pub use store::Store;

pub fn record(store: &mut Store, key: &str, value: i64) {
    store.set(key, value);
}

pub fn read(store: &Store, key: &str) -> Option<i64> {
    store.get(key)
}
