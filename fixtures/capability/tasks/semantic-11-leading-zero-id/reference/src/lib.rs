//! Canonical identifier parsing.

pub fn parse_id(text: &str) -> Option<u32> {
    if text.len() > 1 && text.starts_with('0') {
        return None;
    }
    text.parse().ok()
}
