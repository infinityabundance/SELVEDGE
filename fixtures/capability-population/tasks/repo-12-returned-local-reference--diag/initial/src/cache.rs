//! A cache over the formatter. This does not compile: it returns a
//! reference to a local buffer.

pub fn cached<'a>(text: &str) -> &'a str {
    let value = crate::format::normalise(text);
    &value
}
