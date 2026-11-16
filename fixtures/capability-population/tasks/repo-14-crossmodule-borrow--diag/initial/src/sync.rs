//! Append items to a store. This does not compile: a read borrow is held
//! across the mutation.

use crate::store::Store;

pub fn append_all(store: &mut Store, extra: &[String]) -> usize {
    let first = &store.items[0];
    for item in extra {
        store.items.push(item.clone());
    }
    first.len()
}
