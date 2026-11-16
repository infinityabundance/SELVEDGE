//! Encoding, version 2. JSON support is behind the `json` feature.

#[cfg(feature = "json")]
pub mod json {
    pub fn to_json(value: &str) -> String {
        format!("\"{value}\"")
    }
}
