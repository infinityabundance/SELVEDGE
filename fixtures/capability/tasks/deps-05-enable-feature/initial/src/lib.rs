//! Encode through the vendored crate. This does not build: the
//! `json` feature is not enabled.

pub fn encode(value: &str) -> String {
    codec::json::to_json(value)
}
