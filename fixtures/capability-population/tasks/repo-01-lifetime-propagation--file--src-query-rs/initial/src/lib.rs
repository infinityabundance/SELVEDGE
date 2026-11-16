//! Prices, queries and reports.

pub mod model;
pub mod query;
pub mod report;
pub mod store;

pub use model::Item;

pub fn cheapest_name<'a>(items: &'a [Item], fallback: &'a Item) -> &'a str {
    query::cheapest(items, fallback).name.as_str()
}
