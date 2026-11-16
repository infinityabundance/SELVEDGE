//! Select the vendored mode.

pub fn selected() -> &'static str {
    codec::mode()
}
