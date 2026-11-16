//! Encode through the vendored crate.

pub fn encode(value: &str) -> String {
    codec::to_json(value)
}
