//! Codec, version 2. With default features disabled the same label is
//! returned; the defect is the manifest type.

pub fn label() -> &'static str {
    "codec 2.0.0"
}
