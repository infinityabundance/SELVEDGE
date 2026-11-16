//! A report line.

pub fn line(part: u32) -> String {
    crate::format::ratio(part, 10)
}
