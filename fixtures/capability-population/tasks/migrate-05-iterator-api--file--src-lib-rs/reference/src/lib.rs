//! Collect item names through the item crate.

pub fn names(input: &str) -> Vec<String> {
    let mut names = Vec::new();
    items::scan(input, &mut |item| names.push(item.name.to_string()));
    names
}
