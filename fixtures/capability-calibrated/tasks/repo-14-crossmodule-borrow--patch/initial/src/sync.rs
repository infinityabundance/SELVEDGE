//! Append items to a store.

use crate::store::Store;

// TODO(FUNCTION)

#[cfg(test)]
mod visible {
    use super::*;

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
}
