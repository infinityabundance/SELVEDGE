//! The public slug API.

use crate::util::slug;

pub fn slugify(text: &str) -> String {
    slug(text)
}
