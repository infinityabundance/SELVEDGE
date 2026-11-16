//! A report line. This does not compile: the caller was not updated
//! after the formatter gained its second argument.

pub fn line(part: u32) -> String {
    crate::format::ratio(part)
}
