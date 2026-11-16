//! The registration flow.

use crate::registry::Registry;

pub fn register_and_count(mut registry: Registry, names: &[&str]) -> usize {
    for name in names {
        registry.add(name);
    }
    registry.count()
}

#[cfg(test)]
mod visible {
    use super::*;

    #[test]
    fn two_names_are_counted() {
        assert_eq!(register_and_count(Registry::new(), &["a", "b"]), 2);
    }
}
