//! Path stems, version 2. The API takes `&Path`.

use std::path::Path;

pub fn stem(path: &Path) -> Option<String> {
    path.file_stem().map(|stem| stem.to_string_lossy().into_owned())
}
