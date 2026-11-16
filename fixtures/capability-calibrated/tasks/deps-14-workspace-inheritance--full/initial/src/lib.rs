//! Boolean flag parsing.

pub fn parse_flag(text: &str) -> Option<bool> {
    match text.trim().to_ascii_lowercase().as_str() {
        "yes" | "on" | "1" => Some(true),
        "no" | "off" | "0" => Some(false),
        _ => None,
    }
}
