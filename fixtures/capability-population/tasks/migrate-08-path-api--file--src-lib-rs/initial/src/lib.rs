//! File stems through the vendored crate. This does not build: the
//! API now takes `&Path`.

pub fn stem_of(name: &str) -> Option<String> {
    storage::stem(name)
}
