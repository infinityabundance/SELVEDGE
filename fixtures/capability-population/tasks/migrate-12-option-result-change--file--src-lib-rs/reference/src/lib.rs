//! Counts through the registry crate.

pub use registry::Registry;

pub fn next_count(registry: &Registry, key: &str) -> i64 {
    registry.lookup(key).map(|count| count + 1).unwrap_or(0)
}
