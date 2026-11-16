//! Item scanning, version 2. The collection-returning API is gone.

pub struct Item<'a> {
    pub name: &'a str,
}

pub fn scan<'a>(input: &'a str, visit: &mut dyn FnMut(Item<'a>)) {
    for part in input.split(',') {
        if !part.is_empty() {
            visit(Item { name: part });
        }
    }
}
