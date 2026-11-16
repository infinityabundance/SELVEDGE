//! The registration flow. This does not compile: the registry is used
//! after it is moved.

use crate::registry::Registry;

pub fn register_and_count(mut registry: Registry, names: &[&str]) -> usize {
    for name in names {
        registry.add(name);
    }
    let snapshot = registry;
    let count = registry.count();
    count + snapshot.count()
}
