//! Collect item names through the item crate. This does not build:
//! version 2 replaced `collect` with the callback `scan`.

pub fn names(input: &str) -> Vec<String> {
    items::collect(input).into_iter().map(|item| item.name.to_string()).collect()
}
