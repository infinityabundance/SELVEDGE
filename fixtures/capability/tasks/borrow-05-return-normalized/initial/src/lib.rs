//! Whitespace normalisation. The crate does not build: it returns a
//! reference to a `String` it creates locally.

pub fn normalize(text: &str) -> &str {
    let normalized = text.split_whitespace().collect::<Vec<_>>().join(" ");
    &normalized
}
