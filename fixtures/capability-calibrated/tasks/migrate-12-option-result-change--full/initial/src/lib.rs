//! Counts through the registry crate. This does not build: version 2
//! returns `Option`.

pub use registry::Registry;

pub fn next_count(registry: &Registry, key: &str) -> i64 {
    registry.lookup(key) + 1
}
