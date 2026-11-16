//! Counts through the registry crate.

pub use registry::Registry;

// TODO(FUNCTION)

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
