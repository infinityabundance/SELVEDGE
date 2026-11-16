//! A codec label. This does not build: `default-features` is a string.

pub fn codec_label() -> &'static str {
    codec::label()
}
