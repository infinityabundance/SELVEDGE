//! Render a one-line report.

use crate::model::Item;
use crate::query;

pub fn describe_cheapest(items: &[Item], fallback: &Item) -> String {
    let item = query::cheapest(items, fallback).unwrap();
    format!("{} costs {}", item.name, item.price)
}
