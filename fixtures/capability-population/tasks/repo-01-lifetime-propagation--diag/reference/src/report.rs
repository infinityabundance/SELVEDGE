//! Render a one-line report.

use crate::model::Item;
use crate::query;

pub fn describe_cheapest<'a>(items: &'a [Item], fallback: &'a Item) -> String {
    let item = query::cheapest(items, fallback);
    format!("{} costs {}", item.name, item.price)
}
