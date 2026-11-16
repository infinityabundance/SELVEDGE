//! Append items to a store.

use crate::store::Store;

pub fn append_all(store: &mut Store, extra: &[String]) -> usize {
    let first = store.items.first().map(|item| item.len()).unwrap_or(0);
    for item in extra {
        store.items.push(item.clone());
    }
    first
}
