//! The cheapest item.

use crate::model::Item;

pub fn cheapest<'a>(items: &'a [Item], fallback: &'a Item) -> &'a Item {
    items.iter().min_by_key(|item| item.price).unwrap_or(fallback)
}
