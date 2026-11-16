//! The registration flow.

use crate::registry::Registry;

pub fn register_and_count(mut registry: Registry, names: &[&str]) -> usize {
    for name in names {
        registry.add(name);
    }
    registry.count()
}
