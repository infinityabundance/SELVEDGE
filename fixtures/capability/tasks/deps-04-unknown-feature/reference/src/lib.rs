//! Encode through the vendored crate.

pub fn encode(value: &str) -> String {
    codec::json::to_json(value)
}
