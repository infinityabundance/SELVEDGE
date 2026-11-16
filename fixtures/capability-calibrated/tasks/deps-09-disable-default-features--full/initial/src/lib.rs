//! Render through the vendored crate. This does not build: the
//! default feature enables the legacy API only.

pub fn render(value: &str) -> String {
    codec::render(value)
}
