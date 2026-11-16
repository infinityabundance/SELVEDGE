//! Lookup by name.

use crate::model::Item;

pub fn find<'a>(items: &'a [Item], name: &str) -> Option<&'a Item> {
    items.iter().find(|item| item.name == name)
}
