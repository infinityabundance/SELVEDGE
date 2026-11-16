//! Catalog summary. This does not build: version 2 returns an iterator.

pub use catalog::Catalog;

pub fn summarise(catalog: &Catalog) -> (usize, Option<String>) {
    let first = catalog.items()[0].clone();
    (catalog.items().count(), Some(first))
}
