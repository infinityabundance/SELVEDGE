//! Rendering, version 2. `render` requires `modern`; the default is
//! the legacy API.

#[cfg(feature = "modern")]
pub fn render(value: &str) -> String {
    value.trim().to_uppercase()
}

#[cfg(feature = "legacy")]
pub fn render_legacy(value: &str) -> String {
    value.to_string()
}
