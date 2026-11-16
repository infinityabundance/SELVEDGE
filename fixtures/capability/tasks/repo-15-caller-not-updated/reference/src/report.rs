//! A report line.

pub fn line(text: &str) -> String {
    crate::format::label(text, 6)
}
