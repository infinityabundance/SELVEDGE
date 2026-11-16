//! File stems through the vendored crate.

use std::path::Path;

pub fn stem_of(name: &str) -> Option<String> {
    storage::stem(Path::new(name))
}
