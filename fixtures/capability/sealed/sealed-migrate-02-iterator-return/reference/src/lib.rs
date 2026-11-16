//! Catalog summary.

pub use catalog::Catalog;

pub fn summarise(catalog: &Catalog) -> (usize, Option<String>) {
    let entries: Vec<String> = catalog.items().collect();
    let first = entries.first().cloned();
    (entries.len(), first)
}
