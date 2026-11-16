//! The public slug API. This does not compile: the sibling module is
//! named as if it were an external crate.

pub fn slugify(text: &str) -> String {
    util::slug(text)
}
