//! A cache over the formatter.

pub fn cached(text: &str) -> String {
    crate::format::normalise(text)
}
