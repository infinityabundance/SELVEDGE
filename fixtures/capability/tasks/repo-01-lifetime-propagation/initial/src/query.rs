//! The cheapest item.

use crate::model::Item;

pub fn cheapest(items: &[Item], fallback: &Item) -> Option<&Item> {
    items.iter().min_by_key(|item| item.price).or(Some(fallback))
}
