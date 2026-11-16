//! Render through the vendored crate.

pub fn render(value: &str) -> String {
    codec::render(value)
}
