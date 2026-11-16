//! Command detection.

pub fn is_command(text: &str, name: &str) -> bool {
    if name.is_empty() {
        return false;
    }
    text.split_whitespace()
        .next()
        .map(|word| word.eq_ignore_ascii_case(name))
        .unwrap_or(false)
}
