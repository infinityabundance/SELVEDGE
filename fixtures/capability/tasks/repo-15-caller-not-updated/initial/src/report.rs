//! A report line. This does not compile: the caller was not updated
//! after the formatter gained a width parameter.

pub fn line(text: &str) -> String {
    crate::format::label(text)
}
