//! Counts through the registry crate.

pub use registry::Registry;

pub fn next_count(registry: &Registry, key: &str) -> i64 {
    todo!()
}

#[cfg(test)]
mod visible {
    use super::*;

    fn registry() -> Registry {
        let mut registry = Registry::new();
        registry.insert("alpha", 4);
        registry.insert("zero", 0);
        registry
    }
    
    #[test]
    fn a_present_key_is_incremented() {
        assert_eq!(next_count(&registry(), "alpha"), 5);
    }
}
