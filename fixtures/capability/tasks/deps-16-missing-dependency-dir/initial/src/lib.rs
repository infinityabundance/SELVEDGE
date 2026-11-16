//! A codec label. This does not build: the dependency path is wrong.

pub fn codec_label() -> &'static str {
    codec::label()
}
