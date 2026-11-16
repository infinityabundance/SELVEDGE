//! Encoding, version 2.

pub fn to_json(value: &str) -> String {
    format!("\"{value}\"")
}
