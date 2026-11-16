//! Append items to a store.

use crate::store::Store;

pub fn append_all(store: &mut Store, extra: &[String]) -> usize {
    let first = store.items.first().map(|item| item.len()).unwrap_or(0);
    for item in extra {
        store.items.push(item.clone());
    }
    first
}

#[cfg(test)]
mod visible {
    use crate::{append_all, Store};

    fn store(items: &[&str]) -> Store {
        let mut store = Store::new();
        for item in items {
            store.items.push(item.to_string());
        }
        store
    }

    #[test]
    fn returns_the_first_length_and_appends() {
        let mut store = store(&["a", "bb"]);
        assert_eq!(append_all(&mut store, &["c".to_string()]), 1);
        assert_eq!(store.items.len(), 3);
    }

    #[test]
    fn an_empty_store_reports_zero() {
        let mut store = Store::new();
        assert_eq!(append_all(&mut store, &["x".to_string()]), 0);
        assert_eq!(store.items.len(), 1);
    }

    #[test]
    fn appending_nothing_changes_nothing() {
        let mut store = store(&["abc"]);
        assert_eq!(append_all(&mut store, &[]), 3);
        assert_eq!(store.items.len(), 1);
    }
}
