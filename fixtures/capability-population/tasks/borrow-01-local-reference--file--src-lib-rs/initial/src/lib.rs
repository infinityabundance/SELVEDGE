//! Label construction. The crate does not build: `label` returns a
//! reference to a `String` it creates locally.

pub fn label(prefix: &str) -> &str {
    let text = format!("{prefix}:ready");
    &text
}
