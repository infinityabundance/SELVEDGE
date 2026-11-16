use traits_03_object_safety::{count, MemoryStore, Store};

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
